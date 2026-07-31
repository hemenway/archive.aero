package main

// The per-date processing engine: candidate-group fallback rounds, the
// parallel warp pool with memory/throughput autotuning, VRT-library resume
// and mosaic assembly. Port of _rebuild_vrt_library and process_all_dates.

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// rebuildVRTLibrary scans tempDir for previously created warp outputs so that
// resuming an interrupted run can reuse existing intermediates.
// Returns location -> date -> list of VRT/TIF paths.
func (s *Slicer) rebuildVRTLibrary(allLocations []string) map[string]map[string][]string {
	vrtLibrary := map[string]map[string][]string{}
	if !dirExists(s.tempDir) {
		return vrtLibrary
	}

	s.log("Rebuilding VRT library from existing temp files...")

	locationSet := map[string]bool{}
	for _, loc := range allLocations {
		locationSet[loc] = true
	}

	entries, err := os.ReadDir(s.tempDir)
	if err != nil {
		return vrtLibrary
	}

	addEntry := func(loc, date, path string) {
		if vrtLibrary[loc] == nil {
			vrtLibrary[loc] = map[string][]string{}
		}
		vrtLibrary[loc][date] = append(vrtLibrary[loc][date], path)
	}

	foundCount := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		date := entry.Name()
		// Skip non-date directories (simple validation: should contain hyphen).
		if !strings.Contains(date, "-") {
			continue
		}
		dateDir := filepath.Join(s.tempDir, date)

		// Collect per-location warped outputs (TIF/VRT) and fall back to
		// combined VRTs.
		combinedVRTByLoc := map[string]string{}

		dirFiles, err := os.ReadDir(dateDir)
		if err != nil {
			continue
		}
		var vrtFiles, tifFiles []string
		for _, f := range dirFiles {
			if f.IsDir() {
				continue
			}
			name := f.Name()
			if strings.HasSuffix(name, ".vrt") {
				vrtFiles = append(vrtFiles, filepath.Join(dateDir, name))
			} else if strings.HasSuffix(name, ".tif") {
				tifFiles = append(tifFiles, filepath.Join(dateDir, name))
			}
		}
		sort.Strings(vrtFiles)
		sort.Strings(tifFiles)

		for _, vrtFile := range vrtFiles {
			name := filepath.Base(vrtFile)
			// Skip mosaic VRTs and RGBA intermediate VRTs.
			if strings.HasPrefix(name, "mosaic_") || strings.Contains(name, "_rgba.vrt") {
				continue
			}
			// NOTE: world-width (pathological) filtering is deferred to
			// mosaic-input assembly so this rebuild stays pure filesystem
			// globbing with no GDAL opens.
			stem := stemOf(name)

			// Track per-location combined VRTs: {location}_{date}.vrt
			if strings.HasSuffix(stem, "_"+date) {
				location := stem[:len(stem)-len("_"+date)]
				if locationSet[location] || locationSet[s.normalizeName(location)] {
					combinedVRTByLoc[s.normalizeName(location)] = vrtFile
				}
				continue
			}

			// Treat other VRTs as individual warped outputs: {location}_{stem}.vrt
			for _, loc := range allLocations {
				if strings.HasPrefix(stem, loc+"_") {
					addEntry(loc, date, vrtFile)
					foundCount++
					break
				}
			}
		}

		// Collect warped TIFs: {location}_{original_stem}.tif
		for _, tifFile := range tifFiles {
			stem := stemOf(filepath.Base(tifFile))
			for _, loc := range allLocations {
				if strings.HasPrefix(stem, loc+"_") {
					addEntry(loc, date, tifFile)
					foundCount++
					break
				}
			}
		}

		// If no individual warped outputs were found, fall back to the
		// per-location combined VRTs.
		var combinedLocs []string
		for loc := range combinedVRTByLoc {
			combinedLocs = append(combinedLocs, loc)
		}
		sort.Strings(combinedLocs)
		for _, normLoc := range combinedLocs {
			if vrtLibrary[normLoc] != nil && len(vrtLibrary[normLoc][date]) > 0 {
				continue
			}
			addEntry(normLoc, date, combinedVRTByLoc[normLoc])
			foundCount++
		}
	}

	if foundCount > 0 {
		totalPairs := 0
		for _, dates := range vrtLibrary {
			totalPairs += len(dates)
		}
		s.logf("  Restored %d location-date entries from %d locations", totalPairs, len(vrtLibrary))
	} else {
		s.log("  No existing VRTs found (fresh run)")
	}
	return vrtLibrary
}

type groupState struct {
	location       string
	edition        string
	candidateGroup string
	normLoc        string
	records        []Row
	nextIndex      int
	done           bool
}

// processAllDates processes date ranges in the CSV into mosaicked GeoTIFFs,
// optionally limited by start date.
func (s *Slicer) processAllDates(dateRange *[2]time.Time) {
	s.log("\n=== Starting Chart Processing ===")

	// Get all locations from the CSV (not shapefiles).
	locationSet := map[string]bool{}
	for _, records := range s.doleData {
		for _, rec := range records {
			if normLoc := s.normalizeName(rec["location"]); normLoc != "" {
				locationSet[normLoc] = true
			}
		}
	}
	allLocations := make([]string, 0, len(locationSet))
	for loc := range locationSet {
		allLocations = append(allLocations, loc)
	}
	sort.Strings(allLocations)

	// Count how many locations have at least one row with a cutline.
	locsWithCutline := map[string]bool{}
	for _, records := range s.doleData {
		for _, rec := range records {
			if strings.TrimSpace(rec["cutline"]) != "" || strings.TrimSpace(rec["cutline_wkt"]) != "" {
				if normLoc := s.normalizeName(rec["location"]); normLoc != "" {
					locsWithCutline[normLoc] = true
				}
			}
		}
	}
	s.logf("Found %d locations from CSV (%d with cutlines).", len(allLocations), len(locsWithCutline))

	// Build the processing date list (optionally filtered), earliest first.
	sortedDates := append([]string(nil), s.dateKeyOrder...)
	sort.SliceStable(sortedDates, func(i, j int) bool {
		return s.dateKeySortKey(sortedDates[i]).Before(s.dateKeySortKey(sortedDates[j]))
	})
	if dateRange != nil {
		startDate, endDate := dateRange[0], dateRange[1]
		if startDate.After(endDate) {
			startDate, endDate = endDate, startDate
		}
		var filtered []string
		for _, dateKey := range sortedDates {
			startStr := dateKey
			if pair, ok := s.dateKeyMap[dateKey]; ok {
				startStr = pair[0]
			}
			dateObj := s.dateFromStr(startStr)
			if dateObj == nil {
				continue
			}
			if !dateObj.Before(startDate) && !dateObj.After(endDate) {
				filtered = append(filtered, dateKey)
			}
		}
		sortedDates = filtered
		if len(sortedDates) == 0 {
			s.log("No dates found in the requested range. Nothing to process.")
			return
		}
		s.logf("Limiting processing to %d date ranges (%s to %s)",
			len(sortedDates), startDate.Format("2006-01-02"), endDate.Format("2006-01-02"))
	}

	// === PREPARATION PHASE: download all missing files upfront ===
	s.log("\n=== Preparing: Downloading missing files ===")
	allowed := map[string]bool{}
	for _, d := range sortedDates {
		allowed[d] = true
	}
	s.downloadMissingFiles(allowed)

	// Track all warped outputs per location-date; rebuild from existing temp
	// files to support resume.
	vrtLibrary := s.rebuildVRTLibrary(allLocations)

	totalDates := len(sortedDates)
	s.geotiffTotalPlanned = totalDates
	s.geotiffCompleted = 0
	s.geotiffTimes = nil

	for dateIdx, date := range sortedDates {
		s.logf("\n[%d/%d] Processing range: %s", dateIdx+1, totalDates, date)

		tempDir := filepath.Join(s.tempDir, date)
		_ = os.MkdirAll(tempDir, 0o755)

		outputTiff := filepath.Join(s.outputDir, date+".tif")

		// Resume: skip dates whose final mosaic GeoTIFF already exists.
		if fileSize(outputTiff) > 0 {
			s.logf("  ✓ GeoTIFF already exists - skipping %s", filepath.Base(outputTiff))
			s.geotiffCompleted++
			continue
		}

		records := s.doleData[date]

		// Group records by candidate_group so early/master overlap rows are
		// alternatives.
		type recordGroup struct {
			location, edition, candidateGroup string
			records                           []Row
		}
		var recordGroups []*recordGroup
		recordGroupMap := map[string]*recordGroup{}
		for _, rec := range records {
			location := strings.TrimSpace(rec["location"])
			edition := strings.TrimSpace(rec["edition"])
			candidateGroup := strings.TrimSpace(rec["candidate_group"])
			if candidateGroup == "" {
				normLoc := s.normalizeName(location)
				if normLoc != "" {
					candidateGroup = date + "|" + normLoc
				} else {
					candidateGroup = date + "|row"
				}
				rec["candidate_group"] = candidateGroup
			}
			group, ok := recordGroupMap[candidateGroup]
			if !ok {
				group = &recordGroup{location: location, edition: edition, candidateGroup: candidateGroup}
				recordGroupMap[candidateGroup] = group
				recordGroups = append(recordGroups, group)
			}
			group.records = append(group.records, rec)
		}

		// Track warped outputs selected per location for this date.
		tempWarpedByLocation := map[string][]string{}
		var warpedLocationOrder []string

		// Prepare group state for fallback attempts (GCP-ready rows first).
		var groupStates []*groupState
		for _, group := range recordGroups {
			sort.SliceStable(group.records, func(i, j int) bool {
				return candidateRankOf(group.records[i]) < candidateRankOf(group.records[j])
			})
			groupStates = append(groupStates, &groupState{
				location:       group.location,
				edition:        group.edition,
				candidateGroup: group.candidateGroup,
				normLoc:        s.normalizeName(group.location),
				records:        group.records,
			})
		}

		// Execute warp jobs in parallel, with fallback to duplicate rows on
		// failure.
		if len(groupStates) > 0 {
			var warpWorkerUpperBound int
			if s.parallelWarp > 0 {
				warpWorkerUpperBound = max(1, min(s.parallelWarp, cpuCount))
			} else {
				warpWorkerUpperBound = max(2, cpuCount-2)
				if isAppleSilicon {
					// Keep some headroom for system + upload threads on M-series SoCs.
					warpWorkerUpperBound = min(warpWorkerUpperBound, 6)
				}
			}

			warpWorkerTarget := warpWorkerUpperBound
			var warpRoundRateEMA *float64
			warpMemPerWorkerMB := 900
			if strings.ToLower(s.warpOutput) == "vrt" {
				warpMemPerWorkerMB = 1200
			}
			if s.warpMultithread {
				warpMemPerWorkerMB += 256
			}

			roundIdx := 0
			for {
				var warpJobs []WarpJob
				attemptMap := map[attemptID]*groupState{}
				madeProgress := false

				for _, state := range groupStates {
					if state.done || state.nextIndex >= len(state.records) {
						continue
					}

					rec := state.records[state.nextIndex]
					location := strings.TrimSpace(rec["location"])
					edition := strings.TrimSpace(rec["edition"])
					filename := rec["filename"]

					attemptNum := state.nextIndex + 1
					attemptTotal := len(state.records)
					attemptLabel := ""
					if attemptTotal > 1 {
						attemptLabel = " (option " + strconv.Itoa(attemptNum) + "/" + strconv.Itoa(attemptTotal) + ")"
					}

					// Prefer per-row GCP context when available; fall back to
					// the row's cutline shapefile for pre-georeferenced sources.
					gcpContext := s.buildGCPWarpContext(rec)
					shpPath := ""
					if gcpContext == nil {
						shpPath = s.rowShapefile(rec)
						if shpPath == "" {
							s.logf("    %s (edition %s)%s: no valid GCP context and no cutline shapefile",
								location, edition, attemptLabel)
							state.done = true
							madeProgress = true
							continue
						}
					}

					// Find source files (use filename from CSV if available).
					downloadLink := rec["download_link"]
					foundFiles := s.findFiles(filename, downloadLink)
					if len(foundFiles) == 0 {
						csvLabel := filename
						if csvLabel == "" {
							csvLabel = "no match"
						}
						s.logf("    %s (edition %s)%s: No source files found [CSV: %s]",
							location, edition, attemptLabel, csvLabel)
						state.nextIndex++
						madeProgress = true
						if state.nextIndex < len(state.records) {
							s.logf("    %s (edition %s): trying next duplicate option", location, edition)
						} else {
							s.logf("    ✗ %s (edition %s): all %d option(s) exhausted - location will be missing from %s",
								location, edition, len(state.records), date)
						}
						continue
					}

					fileLabel := filename
					if fileLabel == "" {
						fileLabel = "pattern match"
					}
					s.logf("    %s (edition %s)%s: %s", location, edition, attemptLabel, fileLabel)

					attempt := attemptID{state.candidateGroup, state.nextIndex}
					attemptMap[attempt] = state

					for _, srcFile := range foundFiles {
						jobs := s.prepareWarpJob(srcFile, location, shpPath, tempDir, date, rec)
						for i := range jobs {
							jobs[i].attempt = attempt
						}
						warpJobs = append(warpJobs, jobs...)
					}
				}

				if len(warpJobs) == 0 {
					if madeProgress {
						continue
					}
					break
				}

				roundIdx++
				defAvail, defTotal := 4096, 8192
				if isAppleSilicon {
					defAvail, defTotal = 8192, 16384
				}
				warpMemSnapshot := getMemorySnapshotMB(defAvail, defTotal)
				warpReserveMB := recommendedFreeRAMFloorMB(warpMemSnapshot.TotalMB)
				warpMemBudgetMB := max(0, warpMemSnapshot.AvailableMB-warpReserveMB)
				memCapWorkers := 1
				if warpMemBudgetMB > 0 {
					memCapWorkers = warpMemBudgetMB / warpMemPerWorkerMB
				}
				warpMemoryCap := max(1, min(warpWorkerUpperBound, memCapWorkers))
				currentWarpWorkers := max(1, min(min(warpWorkerTarget, warpMemoryCap), len(warpJobs)))
				if currentWarpWorkers != warpWorkerTarget {
					s.logf("  ↺ Autotune warp: workers %d->%d (free=%dMB, floor=%dMB)",
						warpWorkerTarget, currentWarpWorkers, warpMemSnapshot.AvailableMB, warpReserveMB)
				}
				warpWorkerTarget = currentWarpWorkers
				s.warpEnv = gdalEnvForPhase("warp", currentWarpWorkers, warpMemSnapshot.AvailableMB)
				s.logf("  Processing %d warp jobs with %d parallel workers (round %d, free RAM %dMB)...",
					len(warpJobs), currentWarpWorkers, roundIdx, warpMemSnapshot.AvailableMB)

				roundStart := time.Now()
				results := make([]WarpResult, 0, len(warpJobs))
				var resultsMu sync.Mutex
				sem := make(chan struct{}, currentWarpWorkers)
				var wg sync.WaitGroup
				for _, job := range warpJobs {
					wg.Add(1)
					go func(j WarpJob) {
						defer wg.Done()
						sem <- struct{}{}
						defer func() { <-sem }()
						result := s.warpWorker(j)
						resultsMu.Lock()
						results = append(results, result)
						resultsMu.Unlock()
					}(job)
				}
				wg.Wait()

				roundElapsed := maxFloat(0.001, time.Since(roundStart).Seconds())
				roundSuccesses := 0
				for _, r := range results {
					if r.success {
						roundSuccesses++
					}
				}
				roundRate := float64(roundSuccesses) / roundElapsed

				if warpRoundRateEMA == nil {
					warpRoundRateEMA = &roundRate
				} else {
					switch {
					case roundSuccesses == 0 && warpWorkerTarget > 1:
						next := max(1, warpWorkerTarget-1)
						s.logf("  ↺ Autotune warp: workers %d->%d (no successes in round %d)",
							warpWorkerTarget, next, roundIdx)
						warpWorkerTarget = next
					case roundRate < (*warpRoundRateEMA*0.75) && warpWorkerTarget > 1:
						next := max(1, warpWorkerTarget-1)
						s.logf("  ↺ Autotune warp: workers %d->%d (throughput drop to %.3f jobs/s)",
							warpWorkerTarget, next, roundRate)
						warpWorkerTarget = next
					case roundRate >= (*warpRoundRateEMA*1.05) &&
						warpWorkerTarget < warpWorkerUpperBound &&
						len(warpJobs) > warpWorkerTarget:
						projected := warpWorkerTarget + 1
						projectedNeedMB := warpReserveMB + projected*warpMemPerWorkerMB
						if warpMemSnapshot.AvailableMB >= projectedNeedMB {
							s.logf("  ↺ Autotune warp: workers %d->%d (throughput stable at %.3f jobs/s)",
								warpWorkerTarget, projected, roundRate)
							warpWorkerTarget = projected
						}
					}
					ema := (*warpRoundRateEMA * 0.7) + (roundRate * 0.3)
					warpRoundRateEMA = &ema
				}

				// Group results by attempt (validate output files).
				outputsByAttempt := map[attemptID][]string{}
				for _, result := range results {
					if !result.success || result.outputTiff == "" {
						continue
					}
					outputFile := result.outputTiff
					if fileExists(outputFile) && fileSize(outputFile) > 1024 {
						if s.isWorldWidthWarp(outputFile) {
							s.logf("      Warning: Skipping pathological full-world warp: %s", filepath.Base(outputFile))
							continue
						}
						outputsByAttempt[result.attempt] = append(outputsByAttempt[result.attempt], outputFile)
					} else {
						s.logf("      Warning: Skipping invalid warped file: %s", filepath.Base(outputFile))
					}
				}

				// Apply results and schedule fallback attempts as needed.
				for attempt, state := range attemptMap {
					attemptOutputs := outputsByAttempt[attempt]
					if len(attemptOutputs) > 0 {
						if _, seen := tempWarpedByLocation[state.normLoc]; !seen {
							warpedLocationOrder = append(warpedLocationOrder, state.normLoc)
						}
						tempWarpedByLocation[state.normLoc] = append(tempWarpedByLocation[state.normLoc], attemptOutputs...)
						state.done = true
					} else {
						state.nextIndex++
						if state.nextIndex < len(state.records) {
							s.logf("    %s (edition %s): warp failed; trying next duplicate option",
								state.location, state.edition)
						} else {
							s.logf("    ✗ %s (edition %s): all %d option(s) exhausted - location will be missing from %s",
								state.location, state.edition, len(state.records), date)
						}
					}
				}
			}

			// Register warped outputs directly for mosaic (no per-location
			// VRT combining).
			locationCount := 0
			for _, normLoc := range warpedLocationOrder {
				tempWarped := tempWarpedByLocation[normLoc]
				if len(tempWarped) == 0 {
					continue
				}
				locationCount++
				if len(tempWarped) > 1 {
					s.logf("      Using %d warped files for %s", len(tempWarped), normLoc)
				} else {
					s.log("      Single file, using directly")
				}

				var combined []string
				if vrtLibrary[normLoc] != nil {
					combined = append(combined, vrtLibrary[normLoc][date]...)
				}
				combined = append(combined, tempWarped...)

				// De-duplicate while preserving order.
				seen := map[string]bool{}
				var deduped []string
				for _, p := range combined {
					if seen[p] {
						continue
					}
					seen[p] = true
					deduped = append(deduped, p)
				}
				if vrtLibrary[normLoc] == nil {
					vrtLibrary[normLoc] = map[string][]string{}
				}
				vrtLibrary[normLoc][date] = deduped
			}

			if len(tempWarpedByLocation) > 0 {
				s.logf("  Processed %d locations with data", locationCount)
			} else {
				s.log("  No valid locations found with source files")
			}
		} else {
			s.log("  No valid locations found with source files")
		}

		// Build mosaic inputs from exact date range matches only. Validate
		// every source with the full readability probe: restored temp VRTs
		// from prior runs can reference source rasters that no longer exist,
		// which opens fine but explodes with IReadBlock errors mid-mosaic.
		var mosaicSources []string
		for _, location := range allLocations {
			if vrtLibrary[location] == nil {
				continue
			}
			for _, entry := range vrtLibrary[location][date] {
				if !s.isReusableOutput(entry) {
					s.logf("      Warning: Skipping stale/unreadable mosaic source: %s", filepath.Base(entry))
					// Remove the broken temp so future runs re-warp instead of
					// re-probing.
					safeUnlink(entry)
					continue
				}
				mosaicSources = append(mosaicSources, entry)
			}
		}

		// Mosaic the per-chart rasters directly into the final GeoTIFF.
		if len(mosaicSources) > 0 {
			s.logf("  Mosaicking %d sources -> %s...", len(mosaicSources), filepath.Base(outputTiff))
			ok, elapsed := s.createMosaicGeoTIFF(mosaicSources, outputTiff, s.geotiffCompress)
			if ok {
				s.geotiffCompleted++
				if elapsed > 0 {
					s.geotiffTimes = append(s.geotiffTimes, elapsed)
				}
				s.logf("  ✓✓✓ %s GEOTIFF COMPLETE", strings.ToUpper(date))
			} else {
				s.log("  ✗ Mosaic GeoTIFF creation failed")
			}
		} else {
			s.logf("  Skipping %s - no data available", date)
		}
	}

	s.log("\n=== Processing Complete ===")
}

func candidateRankOf(rec Row) int {
	v := strings.TrimSpace(rec["candidate_rank"])
	if n, err := strconv.Atoi(v); err == nil {
		return n
	}
	return 9999
}
