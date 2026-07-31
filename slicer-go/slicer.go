package main

// ChartSlicer core: state, naming/date helpers and the v2 CSV intake.
// Direct port of the ChartSlicer class in scripts/slicer.py.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

type Slicer struct {
	sourceDir string
	outputDir string
	csvFile   string
	shapeDir  string
	tempDir   string

	doleData     map[string][]Row // date_key -> rows
	dateKeyOrder []string         // insertion order (Python dict order)
	dateKeyMap   map[string][2]string

	srsCache map[string]string
	srsMu    sync.Mutex

	previewMode      bool
	downloadDelay    float64 // seconds between downloads
	lastDownloadTime time.Time

	dateCache map[string]*time.Time

	// GeoTIFF output settings (final stage: warped sources -> mosaic GeoTIFF).
	// LZW/DEFLATE only — geotiff2pmtiles' TIFF reader cannot decode ZSTD.
	geotiffCompress string
	numThreads      string
	clipProjwin     *[4]float64 // ULX, ULY, LRX, LRY

	// GeoTIFF progress/ETA tracking (populated in processAllDates).
	geotiffTotalPlanned int
	geotiffCompleted    int
	geotiffTimes        []float64

	parallelWarp    int
	warpMultithread bool // extra internal GDAL threads per warp (off on Apple Silicon)
	warpOutput      string
	resampleAlg     string // gdalwarp -r name: near|bilinear|cubic|cubicspline

	// Use default PROJ transformer selection to avoid strict-op failures on
	// Alaska/dateline charts (nil = no -to options; kept for parity).
	transformerOptions []string

	abbreviations map[string]string

	// File index caches for O(1) lookups.
	filesByName    map[string]string   // direct TIFs/ZIPs/PDFs/JPGs by basename
	tifsByDir      map[string][]string // extracted folder name -> TIF list
	shapefileIndex map[string]string   // normalized location -> shapefile path

	// Per-phase GDAL subprocess environments (rebuilt when phases start).
	globalEnv  []string
	warpEnv    []string
	geotiffEnv []string

	logMu   sync.Mutex
	tmpSeq  int64
	tmpSeqM sync.Mutex
}

func newSlicer(sourceDir, outputDir, csvFile, shapeDir, tempDir string, downloadDelay float64) *Slicer {
	if tempDir == "" {
		tempDir = "/Volumes/projects/.temp"
	}
	availableRAMMB := getAvailableRAMMB(4096)
	cacheMB := max(512, min(8192, availableRAMMB*3/4))
	s := &Slicer{
		sourceDir:       sourceDir,
		outputDir:       outputDir,
		csvFile:         csvFile,
		shapeDir:        shapeDir,
		tempDir:         tempDir,
		doleData:        map[string][]Row{},
		dateKeyMap:      map[string][2]string{},
		srsCache:        map[string]string{},
		downloadDelay:   downloadDelay,
		dateCache:       map[string]*time.Time{},
		geotiffCompress: "LZW",
		numThreads:      "ALL_CPUS",
		warpMultithread: !isAppleSilicon,
		warpOutput:      "tif",
		resampleAlg:     "near",
		abbreviations: map[string]string{
			"hawaiian_is":           "hawaiian_islands",
			"dallas_ft_worth":       "dallas_ft_worth",
			"mariana_islands_inset": "mariana_islands",
		},
		filesByName:    map[string]string{},
		tifsByDir:      map[string][]string{},
		shapefileIndex: map[string]string{},
	}
	s.globalEnv = baseGDALEnv(cacheMB)
	s.warpEnv = s.globalEnv
	s.geotiffEnv = s.globalEnv
	return s
}

func (s *Slicer) log(msg string) {
	s.logMu.Lock()
	defer s.logMu.Unlock()
	fmt.Printf("[%s] %s\n", time.Now().Format("15:04:05"), msg)
}

func (s *Slicer) logf(format string, args ...any) { s.log(fmt.Sprintf(format, args...)) }

// nextTmpID replaces Python's threading.get_ident() in temp-file names.
func (s *Slicer) nextTmpID() int64 {
	s.tmpSeqM.Lock()
	defer s.tmpSeqM.Unlock()
	s.tmpSeq++
	return s.tmpSeq
}

// normalizeName normalizes a chart/location name consistently.
func (s *Slicer) normalizeName(name string) string {
	norm := strings.ToLower(strings.TrimSpace(name))
	norm = strings.NewReplacer(" ", "_", "-", "_", ".", "_", ",", "").Replace(norm)
	for strings.Contains(norm, "__") {
		norm = strings.ReplaceAll(norm, "__", "_")
	}
	if abbr, ok := s.abbreviations[norm]; ok {
		return abbr
	}
	return norm
}

// sanitizeFilename removes spaces and special characters from a filename stem.
func (s *Slicer) sanitizeFilename(name string) string {
	name = strings.NewReplacer(" ", "_", "-", "_", ".", "_", "(", "", ")", "", ",", "").Replace(name)
	for strings.Contains(name, "__") {
		name = strings.ReplaceAll(name, "__", "_")
	}
	return strings.ToLower(name)
}

// --- date helpers ------------------------------------------------------------

// dateFromStr parses YYYY-MM-DD date strings with caching.
func (s *Slicer) dateFromStr(dateStr string) *time.Time {
	if cached, ok := s.dateCache[dateStr]; ok {
		return cached
	}
	var result *time.Time
	if t, err := time.ParseInLocation("2006-01-02", dateStr, time.UTC); err == nil {
		result = &t
	}
	s.dateCache[dateStr] = result
	return result
}

// dateRangeKey builds a stable date key including both start and end dates.
func (s *Slicer) dateRangeKey(startDate, endDate string) string {
	if startDate == "" {
		return ""
	}
	if endDate == "" || endDate == startDate {
		return startDate
	}
	return startDate + "_to_" + endDate
}

// dateKeySortKey sorts by start date for range keys (zero time for undated).
func (s *Slicer) dateKeySortKey(dateKey string) time.Time {
	startDate := dateKey
	if pair, ok := s.dateKeyMap[dateKey]; ok {
		startDate = pair[0]
	}
	if parsed := s.dateFromStr(startDate); parsed != nil {
		return *parsed
	}
	return time.Time{}
}

// --- CSV intake ---------------------------------------------------------------

var waybackTSRe = regexp.MustCompile(`/web/(\d{14})/`)
var candidateTokenRe = regexp.MustCompile(`[^a-zA-Z0-9_]+`)
var underscoreRunRe = regexp.MustCompile(`_+`)

// loadDoleData loads v2 CSV data into date-keyed groups.
func (s *Slicer) loadDoleData() error {
	s.logf("Loading CSV: %s...", filepath.Base(s.csvFile))
	s.doleData = map[string][]Row{}
	s.dateKeyOrder = nil
	s.dateKeyMap = map[string][2]string{}

	rows, err := loadRows(s.csvFile)
	if err != nil {
		return err
	}

	totalRows := 0
	acceptedRows := 0
	skippedMissingFilename := 0
	undatedRows := 0

	for i, row := range rows {
		rowIdx := i + 1
		totalRows++

		filename := strings.TrimSpace(row["filename"])
		if filename == "" {
			skippedMissingFilename++
			continue
		}

		link := row["download_link"]
		timestamp := ""
		if strings.Contains(link, "web.archive.org/web/") {
			if m := waybackTSRe.FindStringSubmatch(link); m != nil {
				timestamp = m[1]
			}
		}
		row["wayback_ts"] = timestamp

		startDateRaw := strings.TrimSpace(row["date"])
		endDateRaw := strings.TrimSpace(row["end_date"])
		if endDateRaw == "" {
			endDateRaw = startDateRaw
		}

		var parsedStart, parsedEnd *time.Time
		if startDateRaw != "" {
			parsedStart = s.dateFromStr(startDateRaw)
		}
		if endDateRaw != "" {
			parsedEnd = s.dateFromStr(endDateRaw)
		} else {
			parsedEnd = parsedStart
		}

		startDate := startDateRaw
		if parsedStart != nil {
			startDate = parsedStart.Format("2006-01-02")
		}
		endDate := endDateRaw
		if parsedEnd != nil {
			endDate = parsedEnd.Format("2006-01-02")
		} else if endDateRaw == "" {
			endDate = startDate
		}

		var dateKey string
		if startDate != "" {
			dateKey = s.dateRangeKey(startDate, endDate)
			if dateKey == "" {
				continue
			}
		} else {
			undatedRows++
			stem := stemOf(filename)
			if stem == "" {
				stem = fmt.Sprintf("row%d", rowIdx)
			}
			dateKey = "undated_" + stem
		}

		// Duplicate rows for the same date|location are warp alternatives;
		// rows with complete pixel GCPs are tried first.
		normLoc := s.normalizeName(row["location"])
		if parsedStart != nil && normLoc != "" {
			row["candidate_group"] = startDate + "|" + normLoc
		} else {
			token := candidateTokenRe.ReplaceAllString(strings.ToLower(strings.TrimSpace(stemOf(filename))), "_")
			token = strings.Trim(underscoreRunRe.ReplaceAllString(token, "_"), "_")
			if token == "" {
				token = fmt.Sprintf("row%d", rowIdx+1)
			}
			row["candidate_group"] = "undated|" + token
		}
		if rowGCPPixels(row) != nil {
			row["candidate_rank"] = "1"
		} else {
			row["candidate_rank"] = "2"
		}

		row["start_date"] = startDate
		if startDate != "" {
			row["end_date"] = endDate
		} else {
			row["end_date"] = ""
		}
		row["date_key"] = dateKey
		if _, ok := s.doleData[dateKey]; !ok {
			s.dateKeyOrder = append(s.dateKeyOrder, dateKey)
		}
		s.doleData[dateKey] = append(s.doleData[dateKey], row)
		if _, ok := s.dateKeyMap[dateKey]; !ok {
			s.dateKeyMap[dateKey] = [2]string{startDate, endDate}
		}
		acceptedRows++
	}

	s.logf("Loaded %d date ranges.", len(s.doleData))
	s.logf("Accepted %d/%d rows (undated=%d, missing filename skipped=%d).",
		acceptedRows, totalRows, undatedRows, skippedMissingFilename)
	return nil
}

// --- small path/file helpers ----------------------------------------------------

func stemOf(name string) string {
	base := filepath.Base(name)
	ext := filepath.Ext(base)
	return strings.TrimSuffix(base, ext)
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func dirExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

func fileSize(path string) int64 {
	st, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return st.Size()
}

func safeUnlink(path string) {
	if fileExists(path) {
		_ = os.Remove(path)
	}
}
