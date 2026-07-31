package main

// Warp/cut stage: per-chart reprojection into EPSG:3857 with a cutline.
// Port of _build_gcp_warp_context, _gcp_cutline_split_bounds_3857,
// _gcp_cutline_mismatch, _antimeridian_split_bounds_3857, _is_world_width_warp,
// _vrt_sources_exist, _is_reusable_output, _prepare_warp_job, _warp_worker,
// warp_and_cut and warp_and_cut_from_row_gcps.

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type attemptID struct {
	group string
	index int
}

type WarpJob struct {
	inputTiff  string
	shapefile  string
	outputTiff string
	location   string
	date       string
	record     Row
	attempt    attemptID
}

type WarpResult struct {
	success    bool
	outputTiff string
	location   string
	date       string
	attempt    attemptID
}

// GCPContext carries per-row GCP warp inputs built from the v2 columns.
type GCPContext struct {
	Pixels    []Pt // pixel corners gcp1..gcp4 (TL, TR, BR, BL)
	Projected []Pt // GCP coordinates projected into the row's LCC CRS
	Cutline   *Cutline
	SourceCRS string
}

func fstr(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

// cutlineRingOf reads the outer ring of a cutline as (lon, lat) points.
func (s *Slicer) cutlineRingOf(c *Cutline) ([]Pt, error) {
	if c == nil {
		return nil, fmt.Errorf("no cutline")
	}
	if c.Kind == "wkt" {
		return parseWKTPolygonRing(c.WKT)
	}
	return firstFeaturePolygonRing(s.globalEnv, c.Path)
}

// buildGCPWarpContext builds per-row GCP warp inputs from the v2 columns:
// pixel corners, projected GCP coordinates in the row's LCC CRS, and the
// cutline. Returns nil when the row lacks complete GCP metadata.
func (s *Slicer) buildGCPWarpContext(row Row) *GCPContext {
	pixels := rowGCPPixels(row)
	gcpGeo := rowGCPLonLat(row)
	crsStr := rowLCCCRS(row)
	cutline := rowCutline(row, s.shapeDir)
	if pixels == nil || gcpGeo == nil || crsStr == "" || cutline == nil {
		return nil
	}
	if cutline.Kind == "shapefile" && !fileExists(cutline.Path) {
		s.logf("    ✗ Cutline shapefile missing: %s", cutline.Path)
		return nil
	}

	// GCP lat/lon (NAD83) -> the row's LCC CRS. A malformed CRS fails here;
	// skip GCP warp and fall back to the shapefile path, like the original.
	projected, err := transformPoints(s.globalEnv, "EPSG:4269", crsStr, "", gcpGeo)
	if err != nil {
		s.logf("    invalid CRS, falling back to shapefile warp: %q (%v)", crsStr, err)
		return nil
	}

	return &GCPContext{
		Pixels:    pixels,
		Projected: projected,
		Cutline:   cutline,
		SourceCRS: crsStr,
	}
}

// gcpCutlineSplitBounds3857 returns two EPSG:3857 output windows
// (east-of-dateline, west-of-dateline) for a GCP-warp cutline that straddles
// the antimeridian, else nil. Driven by the cutline geometry because the GCP
// VRT has no geotransform to derive windows from.
func (s *Slicer) gcpCutlineSplitBounds3857(cutlineDS, cutlineSRS string) [][4]float64 {
	ring, err := firstFeaturePolygonRing(s.globalEnv, cutlineDS)
	if err != nil || len(ring) == 0 {
		return nil
	}
	srs := cutlineSRS
	if srs == "" {
		srs = "EPSG:4269"
	}
	pts, err := transformPoints(s.globalEnv, srs, "EPSG:4326", "", ring)
	if err != nil {
		return nil
	}
	lons := make([]float64, len(pts))
	lats := make([]float64, len(pts))
	for i, p := range pts {
		lons[i], lats[i] = p.X, p.Y
	}
	if maxOf(lons)-minOf(lons) <= 180.0 {
		return nil // no straddle — normal single warp
	}

	y0, y1 := mercY(minOf(lats)), mercY(maxOf(lats))
	var eastLons, westLons []float64
	for _, l := range lons {
		if l < 0 { // just east of the dateline
			eastLons = append(eastLons, l)
		} else { // just west of the dateline
			westLons = append(westLons, l)
		}
	}
	var bounds [][4]float64
	if len(westLons) > 0 {
		bounds = append(bounds, [4]float64{mercX(minOf(westLons)), y0, worldMerc, y1})
	}
	if len(eastLons) > 0 {
		bounds = append(bounds, [4]float64{-worldMerc, y0, mercX(maxOf(eastLons)), y1})
	}
	if len(bounds) != 2 {
		return nil
	}
	return bounds
}

// gcpCutlineMismatch returns a reason string when a row's cutline cannot
// overlap its GCP footprint (bounding boxes disjoint in the row's LCC space),
// else "". Guard only — any error while checking returns "" so a warp is
// never blocked by the check itself.
func (s *Slicer) gcpCutlineMismatch(ctx *GCPContext) string {
	ring, err := s.cutlineRingOf(ctx.Cutline)
	if err != nil || len(ring) == 0 {
		return ""
	}
	var cutlineSRS string
	if ctx.Cutline.Kind == "shapefile" {
		cutlineSRS = s.getShapefileSRS(ctx.Cutline.Path)
	} else {
		cutlineSRS = cutlineDefaultSRS
	}
	cut, err := transformPoints(s.globalEnv, cutlineSRS, ctx.SourceCRS, "", ring)
	if err != nil {
		return ""
	}
	var gxs, gys, cxs, cys []float64
	for _, p := range ctx.Projected {
		gxs = append(gxs, p.X)
		gys = append(gys, p.Y)
	}
	for _, p := range cut {
		cxs = append(cxs, p.X)
		cys = append(cys, p.Y)
	}
	if minOf(cxs) > maxOf(gxs) || maxOf(cxs) < minOf(gxs) ||
		minOf(cys) > maxOf(gys) || maxOf(cys) < minOf(gys) {
		return fmt.Sprintf("cutline does not intersect the GCP footprint "+
			"(cutline x %.0f..%.0f y %.0f..%.0f vs GCPs x %.0f..%.0f y %.0f..%.0f)",
			minOf(cxs), maxOf(cxs), minOf(cys), maxOf(cys),
			minOf(gxs), maxOf(gxs), minOf(gys), maxOf(gys))
	}
	return ""
}

func minOf(v []float64) float64 {
	m := math.Inf(1)
	for _, x := range v {
		m = math.Min(m, x)
	}
	return m
}

func maxOf(v []float64) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, x)
	}
	return m
}

// antimeridianSplitBounds3857 returns split output bounds (EPSG:3857) for
// georeferenced sources that cross the antimeridian. When a source spans
// +180/-180 a single warp can balloon to nearly world width; splitting into
// east/west output windows keeps extents narrow.
func (s *Slicer) antimeridianSplitBounds3857(inputTiff string, srcCRS string) [][4]float64 {
	info, err := gdalInfoJSON(s.globalEnv, inputTiff)
	if err != nil || !info.HasGeoTransform() {
		return nil
	}
	fileCRS := info.CoordinateSystem != nil && strings.TrimSpace(info.CoordinateSystem.WKT) != ""
	if !fileCRS && srcCRS == "" {
		return nil
	}
	if fileCRS {
		srcCRS = "" // the file's own CRS wins; only fill the gap
	}
	xSize, ySize := float64(info.XSize()), float64(info.YSize())

	// gdaltransform with a positional raster argument takes PIXEL/LINE input
	// and applies the file's geotransform + CRS itself, so the corners go in
	// as pixel coordinates directly.
	cornersPx := []Pt{{0, 0}, {xSize, 0}, {xSize, ySize}, {0, ySize}}
	geoPts, err := transformPoints(s.globalEnv, srcCRS, "EPSG:4326", inputTiff, cornersPx)
	if err != nil {
		return nil
	}

	lons := make([]float64, len(geoPts))
	lats := make([]float64, len(geoPts))
	for i, p := range geoPts {
		lon := p.X
		for lon > 180.0 {
			lon -= 360.0
		}
		for lon < -180.0 {
			lon += 360.0
		}
		lons[i], lats[i] = lon, p.Y
	}
	if len(lons) == 0 {
		return nil
	}

	// Detect dateline crossing by a large longitudinal jump between
	// neighboring corners (wraparound pair included).
	maxJump := 0.0
	for i := range lons {
		prev := lons[(i+len(lons)-1)%len(lons)]
		if jump := math.Abs(lons[i] - prev); jump > maxJump {
			maxJump = jump
		}
	}
	if maxJump < 180.0 {
		return nil
	}

	var posLons, negLons []float64
	for _, lon := range lons {
		if lon >= 0 {
			posLons = append(posLons, lon)
		} else {
			negLons = append(negLons, lon)
		}
	}
	if len(posLons) == 0 || len(negLons) == 0 {
		return nil
	}

	const latMargin, lonMargin = 0.25, 0.25
	latMin := math.Max(-85.0, minOf(lats)-latMargin)
	latMax := math.Min(85.0, maxOf(lats)+latMargin)
	eastLonMin := math.Max(-180.0, minOf(posLons)-lonMargin)
	westLonMax := math.Min(180.0, maxOf(negLons)+lonMargin)

	lonWindows := [][2]float64{
		{eastLonMin, 180.0},
		{-180.0, westLonMax},
	}
	var splitBounds [][4]float64
	for _, w := range lonWindows {
		lonMin, lonMax := w[0], w[1]
		if lonMax <= lonMin {
			continue
		}
		minX, minY := mercX(lonMin), mercY(latMin)
		maxX, maxY := mercX(lonMax), mercY(latMax)
		bounds := [4]float64{
			math.Min(minX, maxX), math.Min(minY, maxY),
			math.Max(minX, maxX), math.Max(minY, maxY),
		}
		if bounds[2]-bounds[0] > 1000 && bounds[3]-bounds[1] > 1000 {
			splitBounds = append(splitBounds, bounds)
		}
	}
	return splitBounds
}

// isWorldWidthWarp detects pathological outputs that balloon to near
// full-world width in EPSG:3857.
func (s *Slicer) isWorldWidthWarp(outputRaster string) bool {
	info, err := gdalInfoJSON(s.globalEnv, outputRaster)
	if err != nil || !info.HasGeoTransform() {
		return false
	}
	widthM := math.Abs(info.GeoTransform[1]) * float64(info.XSize())
	return widthM >= worldWidthM
}

// vrtSourcesExist is true if every SourceFilename referenced by a VRT exists
// on disk (recursing one level into nested VRT sources). A bottom-strip
// checksum probe is not enough for VRTs: blocks outside a missing source's
// window read fine, so the IReadBlock explosion can hide anywhere.
func (s *Slicer) vrtSourcesExist(path string, depth int) bool {
	if depth > 2 {
		return true
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	dec := xml.NewDecoder(strings.NewReader(string(data)))
	for {
		tok, err := dec.Token()
		if err != nil {
			break // io.EOF ends the scan; malformed XML counts as "checked what we could"
		}
		start, ok := tok.(xml.StartElement)
		if !ok {
			continue
		}
		// Plain VRT sources use SourceFilename; warped VRTs use SourceDataset.
		if start.Name.Local != "SourceFilename" && start.Name.Local != "SourceDataset" {
			continue
		}
		relative := false
		for _, attr := range start.Attr {
			if attr.Name.Local == "relativeToVRT" && attr.Value == "1" {
				relative = true
			}
		}
		var content string
		if err := dec.DecodeElement(&content, &start); err != nil {
			return false
		}
		fn := strings.TrimSpace(content)
		if fn == "" {
			continue
		}
		src := fn
		if relative {
			src = filepath.Join(filepath.Dir(path), fn)
		}
		if !fileExists(src) {
			return false
		}
		if strings.EqualFold(filepath.Ext(src), ".vrt") && !s.vrtSourcesExist(src, depth+1) {
			return false
		}
	}
	return true
}

// isReusableOutput is true only if `path` is a fully-readable,
// non-pathological warp output. Used to decide whether an existing
// intermediate can be reused on resume instead of re-warping. A
// corrupt/truncated leftover fails gdalinfo or the bottom-strip read probe
// and returns false, so it gets re-warped rather than silently poisoning the
// date's mosaic. Near-world-width outputs (bad georeferencing) are rejected.
func (s *Slicer) isReusableOutput(path string) bool {
	if !fileExists(path) {
		return false
	}
	// Size floor applies to rasters only: a legitimate VRT can be <1KB of XML.
	if strings.EqualFold(filepath.Ext(path), ".vrt") {
		if !s.vrtSourcesExist(path, 0) {
			return false
		}
	} else if fileSize(path) <= 1024 {
		return false
	}
	info, err := gdalInfoJSON(s.globalEnv, path)
	if err != nil || !info.HasGeoTransform() {
		return false
	}
	if math.Abs(info.GeoTransform[1])*float64(info.XSize()) >= worldWidthM {
		return false
	}
	// Force a read of the last rows so a truncated/partially-written raster
	// (pixel data missing at EOF) fails here instead of at mosaic time.
	strip := min(8, info.YSize())
	if strip <= 0 {
		return false
	}
	probe := filepath.Join(os.TempDir(), fmt.Sprintf("slicer_probe_%d_%d.tif", os.Getpid(), s.nextTmpID()))
	defer safeUnlink(probe)
	_, err = runTool(s.globalEnv, "gdal_translate", "-q",
		"-srcwin", "0", strconv.Itoa(info.YSize()-strip), strconv.Itoa(info.XSize()), strconv.Itoa(strip),
		path, probe)
	return err == nil
}

// prepareWarpJob prepares warp job parameters for parallel processing.
func (s *Slicer) prepareWarpJob(srcFile, location, shpPath, tempDir, date string, record Row) []WarpJob {
	normLoc := s.normalizeName(location)

	// Decide intermediate format for warp output (TIFF default, VRT optional).
	suffix := ".tif"
	if strings.ToLower(s.warpOutput) == "vrt" {
		suffix = ".vrt"
	}

	sanitizedStem := s.sanitizeFilename(stemOf(srcFile))
	tempOut := filepath.Join(tempDir, normLoc+"_"+sanitizedStem+suffix)
	selectedShp := ""
	if shpPath != "" {
		selectedShp = s.pickShapefileForSource(location, srcFile, shpPath)
	}
	return []WarpJob{{
		inputTiff:  srcFile,
		shapefile:  selectedShp,
		outputTiff: tempOut,
		location:   normLoc,
		date:       date,
		record:     record,
	}}
}

// warpWorker is the worker function for parallel warping operations.
func (s *Slicer) warpWorker(job WarpJob) WarpResult {
	success := s.warpAndCut(job.inputTiff, job.shapefile, job.outputTiff, job.record)
	out := ""
	if success {
		out = job.outputTiff
	}
	return WarpResult{
		success:    success,
		outputTiff: out,
		location:   job.location,
		date:       job.date,
		attempt:    job.attempt,
	}
}

// warpAndCut warps and cuts a TIFF using the row's GCPs (preferred) or a
// shapefile cutline. If warpOutput is "vrt", writes a VRT instead of a TIFF.
func (s *Slicer) warpAndCut(inputTiff, shapefile, outputTiff string, record Row) bool {
	// Resume: reuse a fully-readable existing output instead of re-warping.
	if s.isReusableOutput(outputTiff) {
		s.logf("      ↻ Reusing existing warp output %s", filepath.Base(outputTiff))
		return true
	}

	// Safety net for sources downloaded by earlier runs: a .tif that is
	// really a GeoPDF payload gets rasterized in place before warping.
	if isPDFPayload(inputTiff) {
		if !s.convertPDFPayloadToTif(inputTiff, "") {
			return false
		}
	}

	if record != nil {
		if ctx := s.buildGCPWarpContext(record); ctx != nil {
			if s.warpAndCutFromRowGCPs(inputTiff, record, outputTiff) {
				return true
			}
			if shapefile != "" {
				s.logf("      ⚠ GCP warp failed for %s; falling back to shapefile warp", filepath.Base(inputTiff))
			}
		} else if shapefile == "" {
			s.logf("      ✗ Missing/invalid GCP metadata and no shapefile fallback for %s", filepath.Base(inputTiff))
			return false
		}
	}

	if shapefile == "" {
		s.logf("      ✗ No shapefile available for %s", filepath.Base(inputTiff))
		return false
	}

	// Row-declared source CRS (jpg + world-file sources carry a geotransform
	// but no CRS; without -s_srs the warp silently mis-places them).
	srcCRS := ""
	if record != nil {
		srcCRS = rowSrcCRS(record)
	}

	toVRT := strings.ToLower(s.warpOutput) == "vrt"
	if toVRT {
		s.logf("    Warping %s to VRT...", filepath.Base(inputTiff))
	} else {
		s.logf("    Warping %s to TIFF...", filepath.Base(inputTiff))
	}

	// Step 1: build an RGBA VRT source only for paletted inputs. For
	// non-paletted rasters, warp directly from the source TIFF.
	sanitizedStem := s.sanitizeFilename(stemOf(inputTiff))
	warpSource := inputTiff
	tempVRTName := ""

	srcInfo, err := gdalInfoJSON(s.warpEnv, inputTiff)
	if err != nil {
		s.logf("      ✗ Could not open %s", filepath.Base(inputTiff))
		return false
	}
	hasColorTable := len(srcInfo.Bands) > 0 && srcInfo.Bands[0].ColorTable != nil &&
		string(srcInfo.Bands[0].ColorTable) != "null"

	if hasColorTable {
		// (The Python version kept this in /vsimem for TIFF mode; subprocesses
		// can't share vsimem, so it always lands on disk and is deleted below
		// unless the downstream output is a VRT that references it.)
		tempVRTName = filepath.Join(filepath.Dir(outputTiff), sanitizedStem+"_src.vrt")
		_, err := runTool(s.warpEnv, "gdal_translate", "-q", "-of", "VRT", "-expand", "rgba",
			inputTiff, tempVRTName)
		if err != nil {
			s.logf("      ✗ Failed to prepare RGBA source VRT for %s", filepath.Base(inputTiff))
			return false
		}
		warpSource = tempVRTName
	}

	// Get the actual SRS of the shapefile (don't hardcode EPSG:4326).
	shapefileSRS := s.getShapefileSRS(shapefile)
	splitBounds := s.antimeridianSplitBounds3857(inputTiff, srcCRS)
	if len(splitBounds) > 0 {
		s.logf("      ↺ Antimeridian source detected; splitting warp into %d windows", len(splitBounds))
	}

	// Step 2: warp with cutline. Write GTiff (default) or VRT when requested.
	// Not specifying resolution lets GDAL maintain source resolution.
	runWarp := func(transformerOpts []string, outputPath string, outputBounds *[4]float64, forceVRTOutput bool) bool {
		outputFormat := "GTiff"
		if toVRT || forceVRTOutput {
			outputFormat = "VRT"
		}
		safeUnlink(outputPath)
		args := []string{"-q", "-overwrite",
			"-of", outputFormat,
			"-t_srs", "EPSG:3857",
			"-cutline", shapefile,
			"-cutline_srs", shapefileSRS,
			"-dstalpha",
			"-r", s.resampleAlg,
			"-wo", "CUTLINE_ALL_TOUCHED=TRUE",
		}
		if srcCRS != "" {
			args = append(args, "-s_srs", srcCRS)
		}
		if outputBounds == nil {
			args = append(args, "-crop_to_cutline")
		} else {
			args = append(args, "-te",
				fstr(outputBounds[0]), fstr(outputBounds[1]), fstr(outputBounds[2]), fstr(outputBounds[3]))
		}
		if outputFormat == "GTiff" {
			// Intermediates use DEFLATE level 1 to cut temp-dir I/O.
			args = append(args, "-co", "TILED=YES", "-co", "BIGTIFF=YES",
				"-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=1")
		}
		if s.warpMultithread {
			args = append(args, "-multi")
		}
		for _, to := range transformerOpts {
			args = append(args, "-to", to)
		}
		args = append(args, warpSource, outputPath)
		if _, err := runTool(s.warpEnv, "gdalwarp", args...); err != nil {
			return false
		}
		return fileExists(outputPath) && fileSize(outputPath) > 1024
	}

	translateVRTToTiff := func(srcVRT, dstTiff string) bool {
		safeUnlink(dstTiff)
		_, err := runTool(s.warpEnv, "gdal_translate", "-q", "-of", "GTiff",
			"-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=1",
			srcVRT, dstTiff)
		if err != nil {
			return false
		}
		return fileExists(dstTiff) && fileSize(dstTiff) > 1024
	}

	runSplitWarp := func(transformerOpts []string, boundsList [][4]float64) bool {
		if len(boundsList) == 0 {
			return false
		}
		type partItem struct {
			vrt    string
			bounds [4]float64
		}
		var partVRTs []string
		var partItems []partItem
		mergedVRT := ""
		defer func() {
			for _, part := range partVRTs {
				safeUnlink(part)
			}
			if mergedVRT != "" && !toVRT {
				safeUnlink(mergedVRT)
			}
		}()

		outStem := stemOf(outputTiff)
		outDir := filepath.Dir(outputTiff)
		for idx, bounds := range boundsList {
			partVRT := filepath.Join(outDir, fmt.Sprintf("%s_am_part%d.vrt", outStem, idx))
			b := bounds
			if runWarp(transformerOpts, partVRT, &b, true) {
				partVRTs = append(partVRTs, partVRT)
				partItems = append(partItems, partItem{partVRT, bounds})
			} else {
				safeUnlink(partVRT)
			}
		}
		if len(partVRTs) == 0 {
			return false
		}

		if len(partVRTs) == 1 {
			if toVRT {
				safeUnlink(outputTiff)
				if err := os.Rename(partVRTs[0], outputTiff); err != nil {
					return false
				}
				partVRTs = nil
				return fileExists(outputTiff) && fileSize(outputTiff) > 0
			}
			return translateVRTToTiff(partVRTs[0], outputTiff)
		}

		// If parts are on opposite sides of the WebMercator antimeridian,
		// merging them into one VRT yields a full-world canvas. Pick one side.
		hasPos, hasNeg := false, false
		for _, item := range partItems {
			if item.bounds[0] >= 0 {
				hasPos = true
			}
			if item.bounds[2] <= 0 {
				hasNeg = true
			}
		}
		if hasPos && hasNeg {
			stemNorm := s.normalizeName(stemOf(inputTiff))
			preferred := ""
			if strings.Contains(stemNorm, "_east") {
				preferred = "neg"
			} else if strings.Contains(stemNorm, "_west") {
				preferred = "pos"
			}

			var selected *partItem
			pickLargest := func(items []partItem) *partItem {
				var best *partItem
				var bestArea int64 = -1
				for i := range items {
					area := rasterPixelArea(s.warpEnv, items[i].vrt)
					if area > bestArea {
						bestArea = area
						best = &items[i]
					}
				}
				return best
			}
			if preferred == "neg" {
				var negItems []partItem
				for _, item := range partItems {
					if item.bounds[2] <= 0 {
						negItems = append(negItems, item)
					}
				}
				selected = pickLargest(negItems)
			} else if preferred == "pos" {
				var posItems []partItem
				for _, item := range partItems {
					if item.bounds[0] >= 0 {
						posItems = append(posItems, item)
					}
				}
				selected = pickLargest(posItems)
			}
			if selected == nil {
				selected = pickLargest(partItems)
			}

			side := "east-hemisphere"
			if selected.bounds[2] <= 0 {
				side = "west-hemisphere"
			}
			s.logf("      ↺ Antimeridian split spans both world edges; using %s window for %s",
				side, filepath.Base(inputTiff))

			if toVRT {
				safeUnlink(outputTiff)
				if err := os.Rename(selected.vrt, outputTiff); err != nil {
					return false
				}
				remaining := partVRTs[:0]
				for _, p := range partVRTs {
					if p != selected.vrt {
						remaining = append(remaining, p)
					}
				}
				partVRTs = remaining
				return fileExists(outputTiff) && fileSize(outputTiff) > 0
			}
			return translateVRTToTiff(selected.vrt, outputTiff)
		}

		if toVRT {
			mergedVRT = outputTiff
		} else {
			mergedVRT = filepath.Join(outDir, outStem+"_am_merge.vrt")
			safeUnlink(mergedVRT)
		}
		buildArgs := append([]string{"-q", mergedVRT}, partVRTs...)
		if _, err := runTool(s.warpEnv, "gdalbuildvrt", buildArgs...); err != nil {
			return false
		}
		if toVRT {
			return fileExists(mergedVRT) && fileSize(mergedVRT) > 0
		}
		return translateVRTToTiff(mergedVRT, outputTiff)
	}

	runOnce := func(transformerOpts []string) bool {
		if len(splitBounds) > 0 {
			return runSplitWarp(transformerOpts, splitBounds)
		}
		return runWarp(transformerOpts, outputTiff, nil, false)
	}

	success := runOnce(s.transformerOptions)
	if !success && len(s.transformerOptions) > 0 {
		s.log("      ⚠ Warp returned no dataset with strict transformer options; retrying with relaxed options...")
		success = runOnce([]string{"ALLOW_BALLPARK=YES"})
	}

	// Safety guard: if output still spans near full-world width, retry with
	// split bounds.
	if success && s.isWorldWidthWarp(outputTiff) {
		s.log("      ⚠ Detected near-world-width warp output; forcing antimeridian split retry...")
		forcedBounds := splitBounds
		if len(forcedBounds) == 0 {
			forcedBounds = s.antimeridianSplitBounds3857(inputTiff, srcCRS)
		}
		if len(forcedBounds) > 0 {
			success = runSplitWarp([]string{"ALLOW_BALLPARK=YES"}, forcedBounds)
		}
		if success && s.isWorldWidthWarp(outputTiff) {
			s.log("      ✗ Warp still produced pathological full-world width after retry")
			success = false
		}
	}

	// Cleanup the RGBA source VRT unless the output VRT still references it.
	if tempVRTName != "" && !toVRT {
		safeUnlink(tempVRTName)
	}

	if success {
		if toVRT {
			s.logf("      ✓ Created %s (%d bytes)", filepath.Base(outputTiff), fileSize(outputTiff))
		} else {
			s.logf("      ✓ Created %s (%.1f MB)", filepath.Base(outputTiff),
				float64(fileSize(outputTiff))/(1024*1024))
		}
		return true
	}

	safeUnlink(outputTiff)
	s.logf("      ✗ Warp failed for %s", filepath.Base(inputTiff))
	return false
}

// warpAndCutFromRowGCPs warps and clips using row GCP metadata and inferred
// bounds.
func (s *Slicer) warpAndCutFromRowGCPs(inputTiff string, row Row, outputTiff string) bool {
	ctx := s.buildGCPWarpContext(row)
	if ctx == nil {
		s.logf("      ✗ Missing/invalid GCP metadata for %s", filepath.Base(inputTiff))
		return false
	}

	// Rows scanned sideways store a display rotation; their GCP pixels are
	// authored in the rotated frame. Map them back to the raw file's frame
	// (the order-1 GCP fit absorbs the rotation, so no raster resampling).
	if rotation := rowRotation(row); rotation != 0 {
		srcInfo, err := gdalInfoJSON(s.warpEnv, inputTiff)
		if err != nil {
			s.logf("      ✗ Cannot open %s to apply rotation %d", filepath.Base(inputTiff), rotation)
			return false
		}
		rawW, rawH := float64(srcInfo.XSize()), float64(srcInfo.YSize())
		remapped := make([]Pt, len(ctx.Pixels))
		for i, p := range ctx.Pixels {
			remapped[i] = pxDisplayToRaw(p.X, p.Y, rotation, rawW, rawH)
		}
		ctx.Pixels = remapped
		s.logf("    Rotation %d° CW: GCP pixels remapped to raw frame", rotation)
	}

	// A wrong cutline ref or wrong GCP lat/lons otherwise "succeed" into an
	// all-alpha raster and ship an empty mosaic (Tulsa/Milwaukee, 2026-07 run).
	if mismatch := s.gcpCutlineMismatch(ctx); mismatch != "" {
		s.logf("      ✗ %s for %s - fix the dole row (cutline ref or GCP lat/lon)",
			mismatch, filepath.Base(inputTiff))
		return false
	}

	toVRT := strings.ToLower(s.warpOutput) == "vrt"
	if toVRT {
		s.logf("    Warping %s from row GCPs to VRT...", filepath.Base(inputTiff))
	} else {
		s.logf("    Warping %s from row GCPs to TIFF...", filepath.Base(inputTiff))
	}

	sanitizedStem := s.sanitizeFilename(stemOf(inputTiff))
	outDir := filepath.Dir(outputTiff)
	tempGCPVRT := filepath.Join(outDir, sanitizedStem+"_srcgcp.vrt")
	cutlineJSON := ""
	defer func() {
		if !toVRT {
			safeUnlink(tempGCPVRT)
		}
		if cutlineJSON != "" {
			safeUnlink(cutlineJSON)
		}
	}()

	// Build a VRT that assigns the four GCPs in the row's LCC CRS.
	translateArgs := []string{"-q", "-of", "VRT", "-a_srs", ctx.SourceCRS}
	for i := 0; i < 4; i++ {
		px, py := ctx.Pixels[i].X, ctx.Pixels[i].Y
		mx, my := ctx.Projected[i].X, ctx.Projected[i].Y
		translateArgs = append(translateArgs, "-gcp", fstr(px), fstr(py), fstr(mx), fstr(my))
	}
	translateArgs = append(translateArgs, inputTiff, tempGCPVRT)
	if _, err := runTool(s.warpEnv, "gdal_translate", translateArgs...); err != nil {
		s.logf("      ✗ Failed to build GCP VRT for %s", filepath.Base(inputTiff))
		return false
	}

	// Cutline: shapefile refs (extents/ or sectional/) pass through directly;
	// inline WKT overrides become a temp GeoJSON.
	var cutlineDS, cutlineSRS string
	if ctx.Cutline.Kind == "shapefile" {
		cutlineDS = ctx.Cutline.Path
		cutlineSRS = s.getShapefileSRS(ctx.Cutline.Path)
	} else {
		ring, err := s.cutlineRingOf(ctx.Cutline)
		if err != nil {
			s.logf("      ✗ Error warping %s from row GCPs: %v", filepath.Base(inputTiff), err)
			return false
		}
		cutlineJSON = filepath.Join(outDir, sanitizedStem+"_cutline.geojson")
		coords := make([][]float64, 0, len(ring))
		for _, p := range ring {
			coords = append(coords, []float64{p.X, p.Y})
		}
		doc := map[string]any{
			"type": "FeatureCollection",
			"features": []any{map[string]any{
				"type":       "Feature",
				"properties": map[string]any{},
				"geometry": map[string]any{
					"type":        "Polygon",
					"coordinates": []any{coords},
				},
			}},
		}
		data, err := json.Marshal(doc)
		if err != nil {
			return false
		}
		if err := os.WriteFile(cutlineJSON, data, 0o644); err != nil {
			return false
		}
		cutlineDS = cutlineJSON
		cutlineSRS = cutlineDefaultSRS
	}

	// Antimeridian handling: a cutline straddling +/-180 balloons a single
	// 3857 warp to world width (Western Aleutian Islands). Warp each side
	// into its own clamped window and keep the larger side.
	splitBounds := s.gcpCutlineSplitBounds3857(cutlineDS, cutlineSRS)

	runWarp := func(transformerOpts []string) bool {
		commonArgs := func() []string {
			args := []string{
				"-t_srs", "EPSG:3857",
				"-cutline", cutlineDS,
				"-cutline_srs", cutlineSRS,
				"-dstalpha",
				"-r", s.resampleAlg,
				"-order", "1",
				"-wo", "CUTLINE_ALL_TOUCHED=TRUE",
			}
			if s.warpMultithread {
				args = append(args, "-multi")
			}
			for _, to := range transformerOpts {
				args = append(args, "-to", to)
			}
			return args
		}

		if len(splitBounds) > 0 {
			var parts []string
			for sideIdx, bounds := range splitBounds {
				part := filepath.Join(outDir, fmt.Sprintf("%s_am%d.tif", sanitizedStem, sideIdx))
				args := []string{"-q", "-overwrite", "-of", "GTiff",
					"-te", fstr(bounds[0]), fstr(bounds[1]), fstr(bounds[2]), fstr(bounds[3]),
					"-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=1",
				}
				args = append(args, commonArgs()...)
				args = append(args, tempGCPVRT, part)
				if _, err := runTool(s.warpEnv, "gdalwarp", args...); err == nil {
					parts = append(parts, part)
				}
			}
			if len(parts) == 0 {
				return false
			}
			// Merging opposite world edges makes a full-world canvas, so keep
			// the side with the most pixels and drop the sliver.
			best := parts[0]
			var bestArea int64 = -1
			for _, p := range parts {
				if area := rasterPixelArea(s.warpEnv, p); area > bestArea {
					bestArea = area
					best = p
				}
			}
			s.logf("      ↺ Antimeridian split: keeping larger side (%s, %d side(s) warped)",
				filepath.Base(best), len(parts))
			for _, p := range parts {
				if p != best {
					safeUnlink(p)
				}
			}
			safeUnlink(outputTiff)
			return os.Rename(best, outputTiff) == nil
		}

		outputFormat := "GTiff"
		if toVRT {
			outputFormat = "VRT"
		}
		args := []string{"-q", "-overwrite", "-of", outputFormat, "-crop_to_cutline"}
		if outputFormat == "GTiff" {
			args = append(args, "-co", "TILED=YES", "-co", "BIGTIFF=YES",
				"-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=1")
		}
		args = append(args, commonArgs()...)
		args = append(args, tempGCPVRT, outputTiff)
		_, err := runTool(s.warpEnv, "gdalwarp", args...)
		return err == nil
	}

	ok := runWarp(s.transformerOptions)
	if !ok && len(s.transformerOptions) > 0 {
		s.log("      ⚠ Warp failed with strict transformer options; retrying with relaxed options...")
		ok = runWarp([]string{"ALLOW_BALLPARK=YES"})
	}

	if ok && fileExists(outputTiff) && fileSize(outputTiff) > 1024 {
		if s.isWorldWidthWarp(outputTiff) {
			s.logf("      ✗ GCP warp produced pathological full-world width: %s", filepath.Base(outputTiff))
			return false
		}
		s.logf("      ✓ Warp complete: %s", filepath.Base(outputTiff))
		return true
	}

	s.logf("      ✗ Warp failed for %s", filepath.Base(inputTiff))
	return false
}
