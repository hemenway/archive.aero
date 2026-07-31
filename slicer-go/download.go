package main

// Downloading, ZIP extraction, and source-payload fixups (ZIP-under-.tif,
// PDF-under-.tif, 16-bit rescale, PDF rasterization). Port of download_file,
// extract_zip, _is_pdf_payload, _is_zip_payload, _extract_row_file_from_zip_payload,
// _ensure_8bit_tif, _convert_pdf_payload_to_tif, _pdf_is_chart_sheet,
// _materialize_tifs_from_pdf_dir, _materialize_tif_from_local_pdf and
// _download_missing_files.

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

var httpClient = &http.Client{
	Transport: &http.Transport{
		ResponseHeaderTimeout: 300 * time.Second,
		TLSHandshakeTimeout:   30 * time.Second,
	},
}

// downloadFile downloads a URL to targetPath with throttling and retry logic:
// throttling to avoid rate limits, exponential backoff on failure, progress
// reporting, and post-download payload fixups.
func (s *Slicer) downloadFile(url, targetPath string) bool {
	const maxRetries = 3

	// Throttle downloads to avoid rate limiting.
	elapsed := time.Since(s.lastDownloadTime).Seconds()
	if elapsed < s.downloadDelay {
		waitTime := s.downloadDelay - elapsed
		s.logf("    Waiting %.1fs to avoid rate limiting...", waitTime)
		time.Sleep(time.Duration(waitTime * float64(time.Second)))
	}

	retryCount := 0
	backoffDelay := 5 * time.Second

	for retryCount < maxRetries {
		err := s.tryDownload(url, targetPath)
		if err == nil {
			sizeMB := float64(fileSize(targetPath)) / (1024 * 1024)
			s.logf("    ✓ Downloaded: %s (%.1f MB)", filepath.Base(targetPath), sizeMB)

			// Update last download time for throttling.
			s.lastDownloadTime = time.Now()

			// Some links deliver a ZIP saved under the row's .pdf/.tif name
			// (simviation realvfrcharts_alaska_pt*.zip); swap in the real member.
			if !strings.EqualFold(filepath.Ext(targetPath), ".zip") && isZipPayload(targetPath) {
				if !s.extractRowFileFromZipPayload(targetPath) {
					return false
				}
			}

			// Some Wayback "TIF" links actually serve GeoPDFs; rasterize to a
			// real GeoTIFF before indexing.
			if strings.EqualFold(filepath.Ext(targetPath), ".tif") && isPDFPayload(targetPath) {
				if !s.convertPDFPayloadToTif(targetPath, "") {
					return false
				}
			}

			// 16-bit LOC scans get rescaled to 8-bit on arrival so future
			// mosaics stay Byte like the rest of the archive.
			if strings.EqualFold(filepath.Ext(targetPath), ".tif") && !s.ensure8bitTif(targetPath) {
				return false
			}

			s.filesByName[filepath.Base(targetPath)] = targetPath
			return true
		}

		retryCount++
		errStr := err.Error()
		if len(errStr) > 100 {
			errStr = errStr[:100]
		}
		s.logf("    ✗ Download failed (attempt %d/%d): %s", retryCount, maxRetries, errStr)

		// Connection reset errors: the server is rejecting us, slow down
		// aggressively for all future requests.
		lower := strings.ToLower(err.Error())
		if strings.Contains(lower, "connection reset") || strings.Contains(lower, "connection aborted") {
			oldDelay := s.downloadDelay
			s.downloadDelay = max(s.downloadDelay*1.5, s.downloadDelay+5)
			s.logf("    ⚠ Connection error detected - increasing delay from %.1fs to %.1fs",
				oldDelay, s.downloadDelay)
		}

		if retryCount < maxRetries {
			s.logf("    Retrying in %d seconds...", int(backoffDelay.Seconds()))
			time.Sleep(backoffDelay)
			backoffDelay *= 2
			safeUnlink(targetPath) // remove partial file before retry
		} else {
			s.logf("    Giving up after %d attempts", maxRetries)
			safeUnlink(targetPath)
			return false
		}
	}
	return false
}

func (s *Slicer) tryDownload(url, targetPath string) error {
	s.logf("    Downloading: %s", url)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "archive.aero-slicer/1.0")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	totalSize := resp.ContentLength
	out, err := os.Create(targetPath)
	if err != nil {
		return err
	}

	var downloaded int64
	lastLoggedPercent := -10
	buf := make([]byte, 1024*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				out.Close()
				return writeErr
			}
			downloaded += int64(n)
			if totalSize > 0 {
				currentPercent := int(float64(downloaded) / float64(totalSize) * 100)
				if currentPercent >= lastLoggedPercent+10 {
					s.logf("      Progress: %d%%", currentPercent)
					lastLoggedPercent = currentPercent
				}
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			out.Close()
			return readErr
		}
	}
	return out.Close()
}

// extractZip extracts a ZIP to a directory with the same name (minus .zip).
func (s *Slicer) extractZip(zipPath, extractDir string) bool {
	if dirExists(extractDir) {
		s.logf("    Directory already exists: %s", filepath.Base(extractDir))
		return true
	}
	s.logf("    Extracting: %s", filepath.Base(zipPath))

	if err := unzipTo(zipPath, extractDir); err != nil {
		s.logf("    ✗ Extraction failed: %v", err)
		_ = os.RemoveAll(extractDir)
		return false
	}

	// Count extracted files and update the indexes.
	tifFiles := globTifsUnder(extractDir)
	s.tifsByDir[filepath.Base(extractDir)] = tifFiles
	for _, tif := range tifFiles {
		s.filesByName[filepath.Base(tif)] = tif
	}

	s.logf("    ✓ Extracted %d files to: %s", len(tifFiles), filepath.Base(extractDir))
	return true
}

func unzipTo(zipPath, destDir string) error {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer zr.Close()

	for _, f := range zr.File {
		target := filepath.Join(destDir, f.Name)
		// Guard against zip-slip.
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(destDir)+string(os.PathSeparator)) &&
			filepath.Clean(target) != filepath.Clean(destDir) {
			return fmt.Errorf("zip entry escapes destination: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(target)
		if err != nil {
			rc.Close()
			return err
		}
		_, copyErr := io.Copy(out, rc)
		rc.Close()
		closeErr := out.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

// isPDFPayload is true if the file's content is PDF, regardless of extension.
func isPDFPayload(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 5)
	n, _ := io.ReadFull(f, head)
	return n >= 4 && strings.HasPrefix(string(head[:n]), "%PDF")
}

// isZipPayload is true if the file's content is a ZIP archive.
func isZipPayload(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 4)
	n, _ := io.ReadFull(f, head)
	return n == 4 && head[0] == 'P' && head[1] == 'K' && head[2] == 0x03 && head[3] == 0x04
}

// extractRowFileFromZipPayload handles a download link that delivered a ZIP
// saved under the row's .pdf/.tif filename. Extract the archive and replace
// `path` with the member whose stem matches it, converting PDF -> 300 dpi
// GeoTIFF when the row wants a .tif.
func (s *Slicer) extractRowFileFromZipPayload(path string) bool {
	stem := stemOf(path)
	payloadDir := filepath.Join(filepath.Dir(path), stem+"_zip_payload")

	s.logf("    %s is a ZIP payload; extracting to find its real file...", filepath.Base(path))
	if !dirExists(payloadDir) {
		if err := unzipTo(path, payloadDir); err != nil {
			s.logf("    ✗ ZIP payload extraction failed for %s: %v", filepath.Base(path), err)
			return false
		}
	}

	wantStem := strings.ToLower(stem)
	member := ""
	var allFiles []string
	_ = filepath.WalkDir(payloadDir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			allFiles = append(allFiles, p)
		}
		return nil
	})
	sort.Strings(allFiles)
	for _, cand := range allFiles {
		if strings.ToLower(stemOf(cand)) == wantStem {
			member = cand
			break
		}
	}
	if member == "" {
		s.logf("    ✗ No member named '%s.*' inside ZIP payload %s", stem, filepath.Base(path))
		return false
	}

	ok := false
	memberExt := strings.ToLower(filepath.Ext(member))
	pathExt := strings.ToLower(filepath.Ext(path))
	switch {
	case memberExt == pathExt:
		if err := os.Rename(member, path); err == nil {
			ok = true
		}
	case memberExt == ".pdf" && pathExt == ".tif":
		ok = s.convertPDFPayloadToTif(member, path)
	default:
		s.logf("    ✗ Don't know how to turn member %s into %s", filepath.Base(member), filepath.Base(path))
	}
	if ok {
		s.logf("    ✓ Replaced ZIP payload %s with member %s", filepath.Base(path), filepath.Base(member))
		s.filesByName[filepath.Base(path)] = path
		_ = os.RemoveAll(payloadDir)
	}
	return ok
}

// ensure8bitTif rescales 16-bit scans (LOC ca-series batches) to 8-bit in
// place (gdal_translate -ot Byte -scale 0 65535 0 255) so every mosaic stays
// Byte. Non-16-bit files pass through untouched. Returns false only on a
// failed conversion.
func (s *Slicer) ensure8bitTif(path string) bool {
	tmp := filepath.Join(filepath.Dir(path), stemOf(path)+"_8bit_tmp.tif")
	defer safeUnlink(tmp)

	info, err := gdalInfoJSON(s.globalEnv, path)
	if err != nil || len(info.Bands) == 0 {
		return true // unreadable file: pass through, matching gdal.Open->None
	}
	if info.Bands[0].Type != "UInt16" {
		return true
	}

	s.logf("    16-bit source detected: rescaling %s to 8-bit (0..65535 -> 0..255)", filepath.Base(path))
	_, err = runTool(s.globalEnv, "gdal_translate", "-q",
		"-ot", "Byte", "-scale", "0", "65535", "0", "255",
		"-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "COMPRESS=LZW",
		path, tmp)
	if err != nil {
		s.logf("    ✗ 8-bit rescale failed for %s: %v", filepath.Base(path), err)
		return false
	}
	if err := os.Rename(tmp, path); err != nil {
		s.logf("    ✗ 8-bit rescale failed for %s: %v", filepath.Base(path), err)
		return false
	}
	s.logf("    ✓ Rescaled %s to 8-bit", filepath.Base(path))
	return true
}

// convertPDFPayloadToTif rasterizes a PDF (a real .pdf or a PDF payload saved
// under a .tif name) to a GeoTIFF at 300 dpi. Writes to outPath when given
// (keeping the source), otherwise converts in place. Georeferencing embedded
// in the PDF carries over.
func (s *Slicer) convertPDFPayloadToTif(path, outPath string) bool {
	dest := outPath
	if dest == "" {
		dest = path
	}
	tmpOut := filepath.Join(filepath.Dir(dest),
		fmt.Sprintf("%s_pdfconv_%d.tif", stemOf(dest), s.nextTmpID()))
	defer safeUnlink(tmpOut)

	s.logf("    Converting PDF to GeoTIFF (300 dpi): %s -> %s", filepath.Base(path), filepath.Base(dest))

	info, err := gdalInfoJSON(s.globalEnv, path, "-if", "PDF", "-oo", "DPI=300")
	if err != nil {
		s.logf("    ✗ GDAL could not open PDF: %s", filepath.Base(path))
		return false
	}
	if !info.HasGeoTransform() && info.GCPCount() == 0 {
		s.logf("    ⚠ PDF has no georeferencing; %s will need georeferencing before it can be warped",
			filepath.Base(dest))
	}

	_, err = runTool(s.globalEnv, "gdal_translate", "-q",
		"-if", "PDF", "-oo", "DPI=300", "-of", "GTiff",
		"-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=1",
		path, tmpOut)
	if err != nil {
		s.logf("    ✗ PDF to TIF conversion failed for %s: %v", filepath.Base(path), err)
		return false
	}
	if err := os.Rename(tmpOut, dest); err != nil {
		s.logf("    ✗ PDF to TIF conversion failed for %s: %v", filepath.Base(path), err)
		return false
	}
	s.logf("    ✓ Converted PDF to GeoTIFF: %s", filepath.Base(dest))
	return true
}

// pdfIsChartSheet is true when a PDF page is a full chart sheet (>= ~14
// inches on a side at 300 dpi). Weeds out letter/A4 print sets that would
// rasterize into useless page tiles.
func (s *Slicer) pdfIsChartSheet(pdf string) bool {
	info, err := gdalInfoJSON(s.globalEnv, pdf, "-if", "PDF", "-oo", "DPI=300")
	if err != nil {
		return false
	}
	return max(info.XSize(), info.YSize()) >= 4200
}

// materializeTifsFromPDFDir rasterizes each chart-sheet PDF in an extracted
// zip dir to a sibling .tif (300 dpi convention) and returns the tifs. In
// preview mode the PDFs are returned as-is (counted, not converted).
func (s *Slicer) materializeTifsFromPDFDir(extractDir string) []string {
	var pdfs []string
	_ = filepath.WalkDir(extractDir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".pdf") {
			pdfs = append(pdfs, p)
		}
		return nil
	})
	sort.Strings(pdfs)

	var tifs []string
	for _, pdf := range pdfs {
		if !s.pdfIsChartSheet(pdf) {
			s.logf("    Skipping print-size PDF (not a chart sheet): %s", filepath.Base(pdf))
			continue
		}
		if s.previewMode {
			tifs = append(tifs, pdf)
			continue
		}
		tifPath := strings.TrimSuffix(pdf, filepath.Ext(pdf)) + ".tif"
		if !fileExists(tifPath) {
			if !s.convertPDFPayloadToTif(pdf, tifPath) {
				continue
			}
		}
		s.filesByName[filepath.Base(tifPath)] = tifPath
		tifs = append(tifs, tifPath)
	}
	if len(pdfs) > 0 && !s.previewMode {
		s.logf("    %s: materialized %d/%d PDFs as GeoTIFFs", filepath.Base(extractDir), len(tifs), len(pdfs))
	}
	return tifs
}

// materializeTifFromLocalPDF handles CSV rows for PDF-sourced charts that
// carry a .tif filename while the file on disk is the original .pdf:
// rasterize the local PDF next to itself instead of re-downloading.
func (s *Slicer) materializeTifFromLocalPDF(filename string) string {
	if !strings.HasSuffix(filename, ".tif") {
		return ""
	}
	pdfName := stemOf(filename) + ".pdf"
	pdfPath, ok := s.filesByName[pdfName]
	if !ok || !fileExists(pdfPath) {
		return ""
	}
	tifPath := filepath.Join(filepath.Dir(pdfPath), filename)
	if s.convertPDFPayloadToTif(pdfPath, tifPath) {
		s.filesByName[filename] = tifPath
		return tifPath
	}
	return ""
}

// downloadMissingFiles is the preparation phase: download all missing files
// upfront before processing begins, sequentially with proper throttling, to
// prevent burst requests to web.archive.org.
func (s *Slicer) downloadMissingFiles(allowedDates map[string]bool) {
	type missingFile struct {
		filename, downloadLink, location, date string
		isZip                                  bool
	}
	var missing []missingFile

	for _, date := range s.dateKeyOrder {
		if allowedDates != nil && !allowedDates[date] {
			continue
		}
		for _, record := range s.doleData[date] {
			filename := record["filename"]
			downloadLink := record["download_link"]
			location := record["location"]
			if filename == "" || downloadLink == "" {
				continue
			}

			// Check if the file exists locally. Consult the recursive source
			// index, not just source_dir top level.
			if strings.HasSuffix(filename, ".zip") {
				dirName := filename[:len(filename)-4]
				dirPath := filepath.Join(s.sourceDir, dirName)
				if _, ok := s.tifsByDir[dirName]; ok {
					continue
				}
				if dirExists(dirPath) {
					continue
				}
				// ZIP already on disk but never extracted: extract instead of
				// re-downloading.
				if zipPath, ok := s.filesByName[filename]; ok {
					if s.extractZip(zipPath, filepath.Join(filepath.Dir(zipPath), dirName)) {
						continue
					}
				}
				missing = append(missing, missingFile{filename, downloadLink, location, date, true})
			} else {
				filePath := filepath.Join(s.sourceDir, filename)
				if _, ok := s.filesByName[filename]; !ok && !fileExists(filePath) {
					// A matching local .pdf can be rasterized instead.
					if s.materializeTifFromLocalPDF(filename) != "" {
						continue
					}
					missing = append(missing, missingFile{filename, downloadLink, location, date, false})
				}
			}
		}
	}

	if len(missing) == 0 {
		s.log("No missing files to download.")
		return
	}

	// Sort by date, latest first, to match processing order.
	sort.SliceStable(missing, func(i, j int) bool {
		return s.dateKeySortKey(missing[i].date).After(s.dateKeySortKey(missing[j].date))
	})

	s.logf("Found %d missing files to download (sorted latest→oldest)", len(missing))
	s.logf("Download throttle: %.1fs between requests", s.downloadDelay)
	estimatedMins := float64(len(missing)) * s.downloadDelay / 60
	s.logf("Estimated time: ~%.0f minutes (assuming %.1fs per file)\n", estimatedMins, s.downloadDelay)

	downloadedCount := 0
	failedCount := 0

	for idx, mf := range missing {
		s.logf("  [%d/%d] %s - %s", idx+1, len(missing), mf.date, mf.location)
		targetPath := filepath.Join(s.sourceDir, mf.filename)

		if s.downloadFile(mf.downloadLink, targetPath) {
			downloadedCount++
			if mf.isZip {
				dirName := mf.filename[:len(mf.filename)-4]
				if s.extractZip(targetPath, filepath.Join(s.sourceDir, dirName)) {
					s.logf("    ✓ Downloaded and extracted: %s", mf.filename)
				} else {
					s.logf("    ⚠ Downloaded but extraction failed: %s", mf.filename)
					failedCount++
				}
			} else {
				s.logf("    ✓ Downloaded: %s", mf.filename)
			}
			// No extra pause here: downloadFile already throttles the next
			// request via lastDownloadTime.
		} else {
			s.logf("    ✗ Failed to download: %s", mf.filename)
			failedCount++
			if idx+1 < len(missing) {
				time.Sleep(time.Duration(s.downloadDelay * float64(time.Second)))
			}
		}
	}

	s.log("\n=== Download Preparation Complete ===")
	s.logf("Downloaded: %d/%d", downloadedCount, len(missing))
	if failedCount > 0 {
		s.logf("Failed: %d", failedCount)
	}
	s.log("Processing will now use these files as they're available locally.\n")
}
