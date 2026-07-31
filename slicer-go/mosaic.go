package main

// Final stage: mosaic the individual warped chart rasters directly into one
// GeoTIFF per date. Port of create_mosaic_geotiff and
// _make_geotiff_progress_cb. gdalwarp composites straight from the per-chart
// TIFs/VRTs, so the mosaic is the only materialized artifact; overlapping
// charts composite via their alpha bands (last opaque source wins).

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// geotiffProgressLogger replicates the Python GDAL progress callback: log
// per-job progress + ETA every 5%, plus an overall ETA based on completed
// mosaics.
type geotiffProgressLogger struct {
	s         *Slicer
	startTime time.Time
	lastPct   int
}

func (p *geotiffProgressLogger) report(pct int) {
	if pct-p.lastPct < 5 {
		return
	}
	p.lastPct = pct
	elapsed := time.Since(p.startTime).Seconds()

	etaSec := 0.0
	if pct > 0 {
		estTotal := elapsed * 100 / float64(pct)
		etaSec = maxFloat(0, estTotal-elapsed)
		p.s.logf("      GeoTIFF progress: %d%% | ETA %.1f min", pct, etaSec/60)
	} else {
		p.s.logf("      GeoTIFF progress: %d%%", pct)
	}

	// Overall ETA (based on completed GeoTIFFs).
	completed := p.s.geotiffCompleted
	totalPlanned := p.s.geotiffTotalPlanned
	remainingJobs := max(0, totalPlanned-completed)
	durations := p.s.geotiffTimes
	if len(durations) > 0 && remainingJobs > 0 {
		sum := 0.0
		for _, d := range durations {
			sum += d
		}
		avgSec := sum / float64(len(durations))
		var overallSec float64
		if pct > 0 {
			overallSec = etaSec + float64(remainingJobs-1)*avgSec
		} else {
			overallSec = float64(remainingJobs) * avgSec
		}
		overallHours := overallSec / 3600
		if overallHours >= 1 {
			p.s.logf("      Overall ETA: %.1f hours (%d GeoTIFFs remaining, avg %.1f min each)",
				overallHours, remainingJobs, avgSec/60)
		} else {
			p.s.logf("      Overall ETA: %.1f min (%d GeoTIFFs remaining)", overallSec/60, remainingJobs)
		}
	}
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// parseWarpProgress consumes gdalwarp's default progress stream
// ("0...10...20...30 ... 100 - done.") and reports percent ticks. Number
// tokens outside 0..100 (e.g. the "Creating output file that is 15334P x
// 12742L" banner) are ignored; valid numbers resync the percentage and each
// '.' after the meter has started advances 2.5%.
func parseWarpProgress(r io.Reader, report func(pct int)) {
	br := bufio.NewReader(r)
	pct := 0.0
	meterStarted := false
	numBuf := strings.Builder{}
	flushNum := func() {
		if numBuf.Len() > 0 {
			if n, err := strconv.Atoi(numBuf.String()); err == nil && n >= 0 && n <= 100 && float64(n) >= pct {
				pct = float64(n)
				meterStarted = true
				report(int(pct))
			}
			numBuf.Reset()
		}
	}
	for {
		b, err := br.ReadByte()
		if err != nil {
			flushNum()
			return
		}
		switch {
		case b >= '0' && b <= '9':
			numBuf.WriteByte(b)
		case b == '.':
			flushNum()
			if meterStarted && pct+2.5 <= 100 {
				pct += 2.5
				report(int(pct))
			}
		default:
			flushNum()
		}
	}
}

// createMosaicGeoTIFF mosaics the warped chart rasters into one GeoTIFF.
// Returns success and the elapsed seconds (for the overall-ETA average).
func (s *Slicer) createMosaicGeoTIFF(inputFiles []string, outputTiff, compress string) (bool, float64) {
	// Validate inputs (skip missing/empty/corrupt-small files).
	var validFiles []string
	for _, f := range inputFiles {
		if !fileExists(f) {
			s.logf("      Warning: Input file not found: %s", filepath.Base(f))
			continue
		}
		if size := fileSize(f); size < 1024 {
			s.logf("      Warning: Skipping %s (too small: %d bytes)", filepath.Base(f), size)
			continue
		}
		validFiles = append(validFiles, f)
	}
	if len(validFiles) == 0 {
		s.log("      ✗ No valid input files for mosaic")
		return false, 0
	}

	if compress == "NONE" {
		s.logf("    Creating mosaic GeoTIFF with no compression (%s threads)...", s.numThreads)
	} else {
		s.logf("    Creating mosaic GeoTIFF with %s compression (%s threads)...", compress, s.numThreads)
	}

	// Configure GDAL for single-process mosaic creation (use all CPUs).
	s.geotiffEnv = gdalEnvForPhase("geotiff", 1, 0)

	creationOpts := []string{
		"TILED=YES",
		"BIGTIFF=YES",
		"SPARSE_OK=TRUE", // skip all-empty (alpha=0) tiles; geotiff2pmtiles reads them as empty
		"BLOCKXSIZE=1024",
		"BLOCKYSIZE=1024",
		"NUM_THREADS=" + s.numThreads,
	}
	if compress != "NONE" {
		creationOpts = append(creationOpts, "COMPRESS="+compress, "PREDICTOR=2")
		if compress == "DEFLATE" {
			creationOpts = append(creationOpts, "ZLEVEL=1")
		}
	}

	// Output resolution: left unset so gdalwarp matches the finest source
	// pixel size on its own (equivalent to BuildVRT resolution='highest').
	args := []string{"-overwrite", "-of", "GTiff",
		"-r", s.resampleAlg,
		"-multi",
		"-wm", "1024", // MB working buffer (default ~64) -> fewer, larger chunks
		"-wo", "NUM_THREADS=ALL_CPUS", // parallelize the resampling compute
	}
	for _, co := range creationOpts {
		args = append(args, "-co", co)
	}

	// Optional clip window: projWin is ULX ULY LRX LRY -> -te is minX minY maxX maxY.
	if s.clipProjwin != nil {
		ulx, uly, lrx, lry := s.clipProjwin[0], s.clipProjwin[1], s.clipProjwin[2], s.clipProjwin[3]
		s.logf("    Applying clip bounds (minX,minY,maxX,maxY): (%v, %v, %v, %v)", ulx, lry, lrx, uly)
		args = append(args, "-te", fstr(ulx), fstr(lry), fstr(lrx), fstr(uly))
	}

	// Write to a temp name and rename on success so an interrupted/failed
	// mosaic never leaves a partial file at the final path (the resume logic
	// treats an existing output as complete and skips the date).
	tmpOutput := filepath.Join(filepath.Dir(outputTiff), stemOf(outputTiff)+".partial.tif")
	args = append(args, validFiles...)
	args = append(args, tmpOutput)

	startTime := time.Now()
	progress := &geotiffProgressLogger{s: s, startTime: startTime, lastPct: -5}

	cmd := exec.Command("gdalwarp", args...)
	cmd.Env = s.geotiffEnv
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		s.logf("    Error creating mosaic GeoTIFF: %v", err)
		return false, 0
	}
	var stderrBuf strings.Builder
	cmd.Stderr = &stderrBuf
	if err := cmd.Start(); err != nil {
		s.logf("    Error creating mosaic GeoTIFF: %v", err)
		return false, 0
	}
	parseWarpProgress(stdout, progress.report)
	err = cmd.Wait()

	if err != nil {
		s.log("      ✗ Failed to create mosaic GeoTIFF")
		if msg := strings.TrimSpace(stderrBuf.String()); msg != "" {
			tail := msg
			if len(tail) > 400 {
				tail = tail[len(tail)-400:]
			}
			s.logf("      %s", tail)
		}
		safeUnlink(tmpOutput)
		return false, 0
	}

	// An "empty" mosaic (every source warp clipped to nothing) still writes a
	// valid sparse GeoTIFF of a few KB and would otherwise get logged as
	// COMPLETE. Fail loudly and leave no output so the date is retried after
	// the source rows are fixed.
	const emptyFloorBytes = 256 * 1024
	if fileSize(tmpOutput) < emptyFloorBytes {
		s.logf("      ✗ Mosaic wrote no pixels (%d bytes) - all sources clipped empty; "+
			"check cutlines/GCPs of the rows feeding %s", fileSize(tmpOutput), filepath.Base(outputTiff))
		safeUnlink(tmpOutput)
		return false, 0
	}
	if err := os.Rename(tmpOutput, outputTiff); err != nil {
		s.logf("    Error creating mosaic GeoTIFF: %v", err)
		safeUnlink(tmpOutput)
		return false, 0
	}

	elapsed := time.Since(startTime).Seconds()
	fileSizeMB := float64(fileSize(outputTiff)) / (1024 * 1024)
	rate := 0.0
	if elapsed > 0 {
		rate = fileSizeMB / elapsed
	}
	s.logf("      ✓ Mosaic GeoTIFF created: %s (%.1f MB in %.1fs, %.1f MB/s)",
		filepath.Base(outputTiff), fileSizeMB, elapsed, rate)
	return true, elapsed
}

var _ = fmt.Sprintf // keep fmt import if unused paths change
