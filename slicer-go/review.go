package main

// Pre-flight review: a dates × locations matrix of what would be processed.
// Port of generate_review_report and _save_review_csv.

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type reportEntry struct {
	date           string
	location       string
	edition        string
	shapefileFound string
	sourceFiles    string
	numFiles       int
}

// saveReviewCSV writes the review report as a matrix (dates × locations).
func (s *Slicer) saveReviewCSV(reportData []reportEntry, outputPath string) {
	dateSet := map[string]bool{}
	locSet := map[string]bool{}
	for _, e := range reportData {
		dateSet[e.date] = true
		locSet[e.location] = true
	}
	var dates, locations []string
	for d := range dateSet {
		dates = append(dates, d)
	}
	for l := range locSet {
		locations = append(locations, l)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(dates)))
	sort.Strings(locations)

	// date -> location -> cell content
	matrix := map[string]map[string]string{}
	for _, entry := range reportData {
		cell := "---"
		if entry.numFiles > 0 {
			files := strings.Split(entry.sourceFiles, ",")
			if len(files) > 1 {
				cell = fmt.Sprintf("(%d files)", len(files))
			} else {
				filename := filepath.Base(strings.TrimSpace(files[0]))
				if len(filename) > 12 {
					filename = filename[:9] + "..."
				}
				cell = filename
			}
		}
		if matrix[entry.date] == nil {
			matrix[entry.date] = map[string]string{}
		}
		matrix[entry.date][entry.location] = cell
	}

	f, err := os.Create(outputPath)
	if err != nil {
		s.logf("Warning: Could not save review CSV: %v", err)
		return
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	header := append([]string{"Date"}, locations...)
	if err := w.Write(header); err != nil {
		s.logf("Warning: Could not save review CSV: %v", err)
		return
	}
	for _, date := range dates {
		row := []string{date}
		for _, loc := range locations {
			cell, ok := matrix[date][loc]
			if !ok {
				cell = "---"
			}
			row = append(row, cell)
		}
		if err := w.Write(row); err != nil {
			s.logf("Warning: Could not save review CSV: %v", err)
			return
		}
	}
	s.logf("Review report saved to: %s", outputPath)
}

// generateReviewReport shows file matches for all dates, saves the CSV report
// and asks for confirmation. Returns true if the user confirms (or
// autoConfirm is set).
func (s *Slicer) generateReviewReport(autoConfirm bool) bool {
	s.log("\n=== Generating Processing Review ===")

	// Get all locations from the CSV (not shapefiles).
	locationSet := map[string]bool{}
	for _, records := range s.doleData {
		for _, rec := range records {
			if normLoc := s.normalizeName(rec["location"]); normLoc != "" {
				locationSet[normLoc] = true
			}
		}
	}
	var allLocations []string
	for loc := range locationSet {
		allLocations = append(allLocations, loc)
	}
	sort.Strings(allLocations)

	// CSV records lookup: date -> location -> record.
	csvLookup := map[string]map[string]Row{}
	for date, records := range s.doleData {
		for _, rec := range records {
			normLoc := s.normalizeName(rec["location"])
			if csvLookup[date] == nil {
				csvLookup[date] = map[string]Row{}
			}
			csvLookup[date][normLoc] = rec
		}
	}

	sortedDates := append([]string(nil), s.dateKeyOrder...)
	sort.SliceStable(sortedDates, func(i, j int) bool {
		return s.dateKeySortKey(sortedDates[i]).After(s.dateKeySortKey(sortedDates[j]))
	})

	var reportData []reportEntry
	for _, date := range sortedDates {
		for _, location := range allLocations {
			rec, hasRecord := csvLookup[date][location]
			if hasRecord {
				edition := rec["edition"]

				// Row cutline: shapefile ref or inline WKT.
				var shapefileFound string
				if strings.TrimSpace(rec["cutline_wkt"]) != "" {
					shapefileFound = "WKT"
				} else if s.rowShapefile(rec) != "" {
					shapefileFound = "YES"
				} else {
					shapefileFound = "NO"
				}

				filename := rec["filename"]
				downloadLink := rec["download_link"]
				foundFiles := s.findFiles(filename, downloadLink)

				var sourceFilesStr string
				if len(foundFiles) > 0 {
					names := make([]string, 0, len(foundFiles))
					for _, f := range foundFiles {
						names = append(names, filepath.Base(f))
					}
					sourceFilesStr = strings.Join(names, ", ")
				} else if filename != "" {
					// In the CSV but not on disk — will be downloaded during
					// processing.
					sourceFilesStr = "📥 WILL DOWNLOAD: " + filename
				} else {
					sourceFilesStr = "NOT IN CSV"
				}

				reportData = append(reportData, reportEntry{
					date:           date,
					location:       location,
					edition:        edition,
					shapefileFound: shapefileFound,
					sourceFiles:    sourceFilesStr,
					numFiles:       len(foundFiles),
				})
			} else {
				reportData = append(reportData, reportEntry{
					date:           date,
					location:       location,
					edition:        "N/A",
					shapefileFound: "N/A",
					sourceFiles:    "NOT FOUND",
					numFiles:       0,
				})
			}
		}
	}

	timestamp := time.Now().Format("20060102_150405")
	csvPath := filepath.Join(s.outputDir, "processing_review_"+timestamp+".csv")
	s.saveReviewCSV(reportData, csvPath)

	if autoConfirm {
		s.log("Auto-confirm enabled (--yes): proceeding with processing.")
		return true
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nProceed with processing? (y/n): ")
		line, err := reader.ReadString('\n')
		if err != nil {
			return false
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return true
		case "n", "no":
			return false
		default:
			fmt.Println("Please enter 'y' or 'n'")
		}
	}
}
