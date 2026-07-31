package main

// Source-file and shapefile indexing plus filename resolution.
// Port of _build_source_index / _build_shapefile_index / resolve_filename /
// find_files / get_shapefile_srs / _pick_shapefile_for_source.

import (
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
)

// buildSourceIndex walks sourceDir once and caches all files for O(1) lookups.
func (s *Slicer) buildSourceIndex() {
	s.log("Building source file index...")

	_ = filepath.WalkDir(s.sourceDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		filename := d.Name()

		// Cache direct TIF, ZIP, PDF and JPG files by name (PDFs so a local
		// .pdf can be rasterized instead of re-downloading its .tif row;
		// JPGs for the simviation/archive-2004 chart rows).
		if hasAnySuffix(filename, ".tif", ".zip", ".pdf", ".jpg") {
			s.filesByName[filename] = path
		}

		// For TIFs inside extracted directories, index by parent directory name.
		if strings.HasSuffix(filename, ".tif") {
			parent := filepath.Dir(path)
			if parent != s.sourceDir && filepath.Clean(parent) != filepath.Clean(s.sourceDir) {
				parentName := filepath.Base(parent)
				s.tifsByDir[parentName] = append(s.tifsByDir[parentName], path)
			}
		}
		return nil
	})

	s.logf("  Indexed %d files and %d directories", len(s.filesByName), len(s.tifsByDir))
}

// buildShapefileIndex globs shapeDir/sectional once, keyed by normalized name.
func (s *Slicer) buildShapefileIndex() {
	s.log("Building shapefile index...")

	sectionalDir := filepath.Join(s.shapeDir, "sectional")
	if !dirExists(sectionalDir) {
		s.logf("  WARNING: Sectional shapefile directory not found at %s", sectionalDir)
		return
	}

	_ = filepath.WalkDir(sectionalDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(d.Name(), ".shp") {
			return nil
		}
		// Normalize the shapefile name once; used by pickShapefileForSource to
		// upgrade to directional (_east/_west/...) variants per source file.
		s.shapefileIndex[s.normalizeName(stemOf(d.Name()))] = path
		return nil
	})

	s.logf("  Indexed %d shapefiles", len(s.shapefileIndex))
}

func hasAnySuffix(name string, suffixes ...string) bool {
	for _, suf := range suffixes {
		if strings.HasSuffix(name, suf) {
			return true
		}
	}
	return false
}

// getShapefileSRS returns the SRS of a shapefile in Proj4 form, cached.
func (s *Slicer) getShapefileSRS(shapefile string) string {
	s.srsMu.Lock()
	if cached, ok := s.srsCache[shapefile]; ok {
		s.srsMu.Unlock()
		return cached
	}
	s.srsMu.Unlock()

	result := shapefileSRSProj4(s.globalEnv, shapefile)

	s.srsMu.Lock()
	s.srsCache[shapefile] = result
	s.srsMu.Unlock()
	return result
}

// rowShapefile returns the row's cutline shapefile (sectional or extents) for
// non-GCP warps, or "".
func (s *Slicer) rowShapefile(row Row) string {
	cutline := rowCutline(row, s.shapeDir)
	if cutline != nil && cutline.Kind == "shapefile" && fileExists(cutline.Path) {
		return cutline.Path
	}
	return ""
}

// pickShapefileForSource prefers directional shapefile variants
// (east/west/north/south) when the source filename encodes a directional
// split chart, e.g. Western Aleutian Islands East/West.
func (s *Slicer) pickShapefileForSource(location, srcFile, defaultShp string) string {
	if defaultShp == "" {
		return ""
	}
	normLoc := s.normalizeName(location)
	stemNorm := s.normalizeName(stemOf(srcFile))
	for _, suffix := range []string{"_east", "_west", "_north", "_south"} {
		if strings.Contains(stemNorm, suffix) {
			if variant, ok := s.shapefileIndex[normLoc+suffix]; ok {
				return variant
			}
		}
	}
	return defaultShp
}

// globTifsUnder collects **/*.tif under a directory (files only), sorted for
// deterministic behavior.
func globTifsUnder(dir string) []string {
	var tifs []string
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if strings.HasSuffix(d.Name(), ".tif") {
			tifs = append(tifs, path)
		}
		return nil
	})
	sort.Strings(tifs)
	return tifs
}

// resolveFilename resolves a CSV filename to actual file path(s):
//  1. Direct .tif files: "file.tif" -> finds file.tif
//  2. Unzipped .zip directories: "file.zip" -> all .tifs in file/ directory
//  3. Auto-download if not found locally (when downloadLink provided)
func (s *Slicer) resolveFilename(filename, downloadLink string) []string {
	if filename == "" {
		return nil
	}

	// Case 1: .zip file (after unzipping, this becomes a directory)
	if strings.HasSuffix(filename, ".zip") {
		dirName := filename[:len(filename)-4]
		extractDir := filepath.Join(s.sourceDir, dirName)

		// Index the extracted directory if it exists but wasn't indexed.
		if _, ok := s.tifsByDir[dirName]; !ok {
			directPath := extractDir
			if !dirExists(directPath) {
				// The extracted dir may live in a subdir next to its zip.
				if indexedZip, ok := s.filesByName[filename]; ok {
					candidate := filepath.Join(filepath.Dir(indexedZip), dirName)
					if dirExists(candidate) {
						directPath = candidate
						extractDir = candidate
					}
				}
			}
			if dirExists(directPath) {
				s.tifsByDir[dirName] = globTifsUnder(directPath)
			}
		}

		// A local zip that was never extracted (e.g. downloaded by the prep
		// phase, or a *_P.zip print bundle) can be extracted without a
		// re-download.
		if _, ok := s.tifsByDir[dirName]; !ok && !s.previewMode {
			zipPath, ok := s.filesByName[filename]
			if !ok {
				zipPath = filepath.Join(s.sourceDir, filename)
			}
			if fileExists(zipPath) {
				extractDir = filepath.Join(filepath.Dir(zipPath), dirName)
				if !s.extractZip(zipPath, extractDir) {
					s.logf("    Failed to extract: %s", filename)
				}
			}
		}

		// Still nothing local: download + extract (but not in preview mode).
		if _, ok := s.tifsByDir[dirName]; !ok && downloadLink != "" && !s.previewMode {
			s.logf("    File not found locally: %s", filename)
			zipPath := filepath.Join(s.sourceDir, filename)
			extractDir = filepath.Join(s.sourceDir, dirName)
			if s.downloadFile(downloadLink, zipPath) {
				if !s.extractZip(zipPath, extractDir) {
					s.logf("    Failed to extract: %s", filename)
				}
			} else {
				s.logf("    Failed to download: %s", downloadLink)
			}
		}

		tifFiles := s.tifsByDir[dirName]
		if len(tifFiles) == 0 && dirExists(extractDir) {
			// Zip dirs holding only PDFs (realcharts_sec*.zip, wayback *_P.zip
			// print bundles): rasterize each PDF to a sibling .tif (300 dpi
			// convention) and mosaic from those.
			tifFiles = s.materializeTifsFromPDFDir(extractDir)
			if len(tifFiles) > 0 {
				s.tifsByDir[dirName] = tifFiles
			}
		}
		return tifFiles
	}

	// Case 2: Direct .tif/.pdf/.jpg file — check index first (O(1) lookup).
	if found, ok := s.filesByName[filename]; ok {
		// A downloader can save ZIP bytes under a row's .pdf/.tif name
		// (simviation realvfrcharts_alaska_pt*.zip). Materialize the real
		// member before handing the path to GDAL.
		if !s.previewMode && !strings.EqualFold(filepath.Ext(found), ".zip") && isZipPayload(found) {
			if !s.extractRowFileFromZipPayload(found) {
				return nil
			}
		}
		// Rows that point straight at a PDF: warp from a 300 dpi GeoTIFF
		// conversion beside it, keeping resolution consistent with the
		// *_PDFs_* sources. Print-size PDFs are left alone.
		if !s.previewMode && strings.EqualFold(filepath.Ext(found), ".pdf") &&
			isPDFPayload(found) && s.pdfIsChartSheet(found) {
			tifPath := strings.TrimSuffix(found, filepath.Ext(found)) + ".tif"
			if fileExists(tifPath) || s.convertPDFPayloadToTif(found, tifPath) {
				s.filesByName[filepath.Base(tifPath)] = tifPath
				found = tifPath
			}
		}
		return []string{found}
	}

	// Not in index — check if file exists but wasn't indexed.
	directPath := filepath.Join(s.sourceDir, filename)
	if fileExists(directPath) {
		s.filesByName[filename] = directPath
		return []string{directPath}
	}

	// A matching local .pdf can be rasterized instead of downloading.
	if !s.previewMode {
		if localTif := s.materializeTifFromLocalPDF(filename); localTif != "" {
			return []string{localTif}
		}
	}

	// If not found, try to download (but not in preview mode).
	if downloadLink != "" && !s.previewMode {
		s.logf("    File not found locally: %s", filename)
		tifPath := filepath.Join(s.sourceDir, filename)
		if s.downloadFile(downloadLink, tifPath) {
			return []string{tifPath}
		}
		s.logf("    Failed to download: %s", downloadLink)
	}

	return nil
}

// findFiles finds files from the CSV filename (must be provided in v2 CSV).
func (s *Slicer) findFiles(filename, downloadLink string) []string {
	if filename == "" {
		return nil
	}
	return s.resolveFilename(filename, downloadLink)
}
