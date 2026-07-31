package main

// FAA Chart Newslicer (Go port) — CLI tool for processing sectional charts.
// Pipeline: download -> warp/cut -> mosaic -> GeoTIFF (one per date).
//
// Feature-parity port of scripts/slicer.py. All raster work is delegated to
// the GDAL command-line tools (Homebrew GDAL), one subprocess per operation.

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func fatalf(format string, args ...any) {
	fmt.Printf(format+"\n", args...)
	os.Exit(1)
}

func parseProjwin(s string) (*[4]float64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == ',' })
	var vals []float64
	for _, f := range fields {
		if f == "" {
			continue
		}
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, fmt.Errorf("bad --clip-projwin value %q", f)
		}
		vals = append(vals, v)
	}
	if len(vals) != 4 {
		return nil, fmt.Errorf("--clip-projwin needs exactly 4 values (ULX ULY LRX LRY), got %d", len(vals))
	}
	out := [4]float64{vals[0], vals[1], vals[2], vals[3]}
	return &out, nil
}

func main() {
	var (
		sourceDir     string
		outputDir     string
		csvFile       string
		shapeDir      string
		tempDir       string
		resample      string
		warpOutput    string
		clipProjwin   string
		compression   string
		numThreads    string
		parallelWarp  int
		warpMulti     bool
		downloadDelay float64
		yes           bool
		allDates      bool
		startDateStr  string
		endDateStr    string
	)

	fs := flag.NewFlagSet("slicer", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "FAA Chart Slicer (Go) - Warp, cut, and mosaic sectional charts into GeoTIFFs\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\nOptions:\n", os.Args[0])
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n  %s\n  %s -s /path/to/charts -o /path/to/output -c master_dole_v2.csv -b shapefiles\n",
			os.Args[0], os.Args[0])
	}

	fs.StringVar(&sourceDir, "s", "/Volumes/projects/rawtiffs", "Source directory containing TIFF/ZIP files")
	fs.StringVar(&sourceDir, "source", "/Volumes/projects/rawtiffs", "Source directory containing TIFF/ZIP files")
	fs.StringVar(&outputDir, "o", "/Volumes/drive/sync", "Output directory for mosaicked GeoTIFFs (one {date}.tif per date)")
	fs.StringVar(&outputDir, "output", "/Volumes/drive/sync", "Output directory for mosaicked GeoTIFFs")
	fs.StringVar(&csvFile, "c", "/Users/ryanhemenway/archive.aero/master_dole_v2.csv", "Master Dole CSV file, v2 schema")
	fs.StringVar(&csvFile, "csv", "/Users/ryanhemenway/archive.aero/master_dole_v2.csv", "Master Dole CSV file, v2 schema")
	fs.StringVar(&shapeDir, "b", "/Users/ryanhemenway/archive.aero/shapefiles", "Directory containing shapefiles")
	fs.StringVar(&shapeDir, "shapefiles", "/Users/ryanhemenway/archive.aero/shapefiles", "Directory containing shapefiles")
	fs.StringVar(&tempDir, "t", "/Volumes/projects/.temp", "Temporary directory for intermediate files")
	fs.StringVar(&tempDir, "temp-dir", "/Volumes/projects/.temp", "Temporary directory for intermediate files")
	fs.StringVar(&resample, "r", "nearest", "Resampling algorithm: nearest|bilinear|cubic|cubicspline")
	fs.StringVar(&resample, "resample", "nearest", "Resampling algorithm: nearest|bilinear|cubic|cubicspline")
	fs.StringVar(&warpOutput, "warp-output", "tif", "Intermediate warp output format: tif (materialize each cutline warp; parallel pixel work) or vrt (defer warp compute into the largely-serial mosaic step)")
	fs.StringVar(&clipProjwin, "clip-projwin", "", "Optional projWin window in target CRS: \"ULX ULY LRX LRY\" (space or comma separated)")
	fs.StringVar(&compression, "compression", "LZW", "Mosaic GeoTIFF compression: LZW|DEFLATE|NONE (ZSTD intentionally omitted - geotiff2pmtiles cannot decode it)")
	fs.StringVar(&numThreads, "num-threads", "ALL_CPUS", "GTiff NUM_THREADS creation option for GeoTIFF writing")
	fs.IntVar(&parallelWarp, "parallel-warp", 0, "Number of parallel warp jobs (default: auto; Apple Silicon auto mode caps at 6)")
	fs.BoolVar(&warpMulti, "warp-multithread", false, "Enable GDAL internal multithreading inside each warp job (off by default on Apple Silicon)")
	fs.Float64Var(&downloadDelay, "download-delay", 8.0, "Seconds to wait between downloads to avoid rate limiting (try 12-20 if hitting connection errors)")
	fs.BoolVar(&yes, "y", false, "Skip the 'Proceed with processing?' prompt (non-interactive)")
	fs.BoolVar(&yes, "yes", false, "Skip the 'Proceed with processing?' prompt (non-interactive)")
	fs.BoolVar(&allDates, "all", false, "Process all dates without prompting (mutually exclusive with --start-date/--end-date)")
	fs.StringVar(&startDateStr, "start-date", "", "Start of date range to process (YYYY-MM-DD, use with --end-date)")
	fs.StringVar(&endDateStr, "end-date", "", "End of date range to process (YYYY-MM-DD, use with --start-date)")
	_ = fs.Parse(os.Args[1:])

	// The whole pipeline shells out to GDAL; fail fast if it's missing.
	for _, tool := range []string{"gdalinfo", "gdalwarp", "gdal_translate", "gdalbuildvrt", "gdaltransform", "gdalsrsinfo", "ogrinfo"} {
		if _, err := exec.LookPath(tool); err != nil {
			fatalf("ERROR: required GDAL tool %q not found on PATH. Install with: brew install gdal", tool)
		}
	}

	// Validate paths.
	if !dirExists(sourceDir) {
		fatalf("ERROR: Source directory not found: %s", sourceDir)
	}
	if !fileExists(csvFile) {
		fatalf("ERROR: CSV file not found: %s", csvFile)
	}
	if !dirExists(shapeDir) {
		fatalf("ERROR: Shapefiles directory not found: %s", shapeDir)
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		fatalf("ERROR: cannot create output directory: %v", err)
	}
	if err := os.MkdirAll(tempDir, 0o755); err != nil {
		fatalf("ERROR: cannot create temp directory: %v", err)
	}

	resampleMap := map[string]string{
		"nearest":     "near",
		"bilinear":    "bilinear",
		"cubic":       "cubic",
		"cubicspline": "cubicspline",
	}
	resampleAlg, ok := resampleMap[resample]
	if !ok {
		fatalf("ERROR: invalid --resample %q (choose nearest|bilinear|cubic|cubicspline)", resample)
	}
	warpOutput = strings.ToLower(warpOutput)
	if warpOutput != "tif" && warpOutput != "vrt" {
		fatalf("ERROR: invalid --warp-output %q (choose tif|vrt)", warpOutput)
	}
	compression = strings.ToUpper(strings.TrimSpace(compression))
	if compression != "LZW" && compression != "DEFLATE" && compression != "NONE" {
		fatalf("ERROR: invalid --compression %q (choose LZW|DEFLATE|NONE)", compression)
	}
	projwin, err := parseProjwin(clipProjwin)
	if err != nil {
		fatalf("ERROR: %v", err)
	}

	slicer := newSlicer(sourceDir, outputDir, csvFile, shapeDir, tempDir, downloadDelay)
	slicer.warpOutput = warpOutput
	slicer.clipProjwin = projwin
	slicer.parallelWarp = parallelWarp
	if warpMulti {
		slicer.warpMultithread = true
	}
	slicer.numThreads = numThreads
	slicer.geotiffCompress = compression
	slicer.resampleAlg = resampleAlg

	if err := slicer.loadDoleData(); err != nil {
		fatalf("ERROR: %v", err)
	}

	// Build indexes for O(1) lookups (after loading CSV data).
	slicer.buildSourceIndex()
	slicer.buildShapefileIndex()

	// During the review phase, disable downloads (don't try to validate by
	// downloading) to avoid rate limiting.
	slicer.previewMode = true

	// Validate non-interactive date-range args early (before the slow review pass).
	if allDates && (startDateStr != "" || endDateStr != "") {
		fatalf("ERROR: --all cannot be combined with --start-date/--end-date.")
	}
	if (startDateStr != "") != (endDateStr != "") {
		fatalf("ERROR: --start-date and --end-date must be provided together.")
	}

	var cliDateRange *[2]time.Time
	if startDateStr != "" && endDateStr != "" {
		startDt := slicer.dateFromStr(startDateStr)
		endDt := slicer.dateFromStr(endDateStr)
		if startDt == nil || endDt == nil {
			fatalf("ERROR: --start-date/--end-date must be valid YYYY-MM-DD dates.")
		}
		sd, ed := *startDt, *endDt
		if sd.After(ed) {
			sd, ed = ed, sd
			fmt.Printf("Swapped range to %s - %s\n", sd.Format("2006-01-02"), ed.Format("2006-01-02"))
		}
		cliDateRange = &[2]time.Time{sd, ed}
	}

	// Generate review report and get user confirmation (without downloading).
	if !slicer.generateReviewReport(yes) {
		fmt.Println("Processing cancelled by user.")
		os.Exit(0)
	}

	// Now enable downloads for actual processing.
	slicer.previewMode = false

	var dateRange *[2]time.Time
	switch {
	case allDates:
		dateRange = nil
	case cliDateRange != nil:
		dateRange = cliDateRange
	default:
		dateRange = promptDateRange(slicer)
	}
	slicer.processAllDates(dateRange)
}

func promptDateRange(slicer *Slicer) *[2]time.Time {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Process all dates or a range? (all/range): ")
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil
		}
		choice := strings.ToLower(strings.TrimSpace(line))
		if choice == "all" || choice == "a" || choice == "" {
			return nil
		}
		if choice == "range" || choice == "r" {
			fmt.Print("Start date (YYYY-MM-DD): ")
			startStr, _ := reader.ReadString('\n')
			fmt.Print("End date (YYYY-MM-DD): ")
			endStr, _ := reader.ReadString('\n')
			startDt := slicer.dateFromStr(strings.TrimSpace(startStr))
			endDt := slicer.dateFromStr(strings.TrimSpace(endStr))
			if startDt == nil || endDt == nil {
				fmt.Println("Please enter valid dates in YYYY-MM-DD format.")
				continue
			}
			sd, ed := *startDt, *endDt
			if sd.After(ed) {
				sd, ed = ed, sd
				fmt.Printf("Swapped range to %s - %s\n", sd.Format("2006-01-02"), ed.Format("2006-01-02"))
			}
			return &[2]time.Time{sd, ed}
		}
		fmt.Println("Please enter 'all' or 'range'.")
	}
}
