package main

// Thin wrappers around the GDAL command-line tools. Every raster/vector
// operation the Python version did through osgeo bindings maps to one CLI
// invocation here:
//
//	gdal.Open metadata probes        -> gdalinfo -json
//	gdal.Translate                   -> gdal_translate
//	gdal.Warp                        -> gdalwarp
//	gdal.BuildVRT                    -> gdalbuildvrt
//	osr.CoordinateTransformation     -> gdaltransform (points on stdin)
//	layer.GetSpatialRef().ExportToProj4 -> gdalsrsinfo -o proj4
//	ogr feature/ring reads           -> ogrinfo -json -features

import (
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// Warp outputs whose EPSG:3857 width reaches ~full-world are pathological.
const worldWidthM = 35_000_000.0

// EPSG:3857 half-world extent in metres.
const worldMerc = 20037508.342789244

type cmdError struct {
	name   string
	args   []string
	stderr string
	err    error
}

func (e *cmdError) Error() string {
	tail := e.stderr
	if len(tail) > 400 {
		tail = tail[len(tail)-400:]
	}
	return fmt.Sprintf("%s failed: %v: %s", e.name, e.err, strings.TrimSpace(tail))
}

// runTool executes a GDAL CLI tool and returns its stdout.
func runTool(env []string, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	if env != nil {
		cmd.Env = env
	}
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), &cmdError{name: name, args: args, stderr: stderr.String(), err: err}
	}
	return stdout.String(), nil
}

// --- gdalinfo ---------------------------------------------------------------

type GDALBand struct {
	Type       string          `json:"type"`
	ColorTable json.RawMessage `json:"colorTable"`
}

type GDALInfo struct {
	Size             []int           `json:"size"`
	GeoTransform     []float64       `json:"geoTransform"`
	Bands            []GDALBand      `json:"bands"`
	CoordinateSystem *struct {
		WKT string `json:"wkt"`
	} `json:"coordinateSystem"`
	GCPs *struct {
		GCPList []json.RawMessage `json:"gcpList"`
	} `json:"gcps"`
}

func (gi *GDALInfo) HasGeoTransform() bool { return len(gi.GeoTransform) == 6 }
func (gi *GDALInfo) GCPCount() int {
	if gi.GCPs == nil {
		return 0
	}
	return len(gi.GCPs.GCPList)
}
func (gi *GDALInfo) XSize() int {
	if len(gi.Size) > 0 {
		return gi.Size[0]
	}
	return 0
}
func (gi *GDALInfo) YSize() int {
	if len(gi.Size) > 1 {
		return gi.Size[1]
	}
	return 0
}

// gdalInfoJSON probes a raster's metadata. extraArgs come before the path
// (e.g. "-if", "PDF", "-oo", "DPI=300").
func gdalInfoJSON(env []string, path string, extraArgs ...string) (*GDALInfo, error) {
	args := append([]string{"-json"}, extraArgs...)
	args = append(args, path)
	out, err := runTool(env, "gdalinfo", args...)
	if err != nil {
		return nil, err
	}
	var info GDALInfo
	if err := json.Unmarshal([]byte(out), &info); err != nil {
		return nil, fmt.Errorf("gdalinfo -json parse for %s: %w", path, err)
	}
	return &info, nil
}

// rasterPixelArea returns XSize*YSize, or 0 on any error.
func rasterPixelArea(env []string, path string) int64 {
	info, err := gdalInfoJSON(env, path)
	if err != nil {
		return 0
	}
	return int64(info.XSize()) * int64(info.YSize())
}

// --- gdaltransform ----------------------------------------------------------

// transformPoints reprojects points via gdaltransform (traditional GIS axis
// order: x=lon, y=lat for geographic CRS, matching the Python code's
// OAMS_TRADITIONAL_GIS_ORDER setting). srcSRS may be "" when srcFile is a
// raster whose CRS should be used instead.
func transformPoints(env []string, srcSRS, dstSRS, srcFile string, pts []Pt) ([]Pt, error) {
	args := []string{"-output_xy"}
	if srcSRS != "" {
		args = append(args, "-s_srs", srcSRS)
	}
	args = append(args, "-t_srs", dstSRS)
	if srcFile != "" {
		args = append(args, srcFile)
	}

	var input strings.Builder
	for _, p := range pts {
		fmt.Fprintf(&input, "%.17g %.17g\n", p.X, p.Y)
	}

	cmd := exec.Command("gdaltransform", args...)
	if env != nil {
		cmd.Env = env
	}
	cmd.Stdin = strings.NewReader(input.String())
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, &cmdError{name: "gdaltransform", args: args, stderr: stderr.String(), err: err}
	}

	var result []Pt
	for _, line := range strings.Split(strings.TrimSpace(stdout.String()), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		x, errX := strconv.ParseFloat(fields[0], 64)
		y, errY := strconv.ParseFloat(fields[1], 64)
		if errX != nil || errY != nil {
			return nil, fmt.Errorf("gdaltransform output parse: %q", line)
		}
		result = append(result, Pt{x, y})
	}
	if len(result) != len(pts) {
		return nil, fmt.Errorf("gdaltransform returned %d points for %d inputs (stderr: %s)",
			len(result), len(pts), strings.TrimSpace(stderr.String()))
	}
	return result, nil
}

// --- gdalsrsinfo ------------------------------------------------------------

// shapefileSRSProj4 reads a datasource's SRS as a Proj4 string, falling back
// to EPSG:4326 like the Python original.
func shapefileSRSProj4(env []string, path string) string {
	out, err := runTool(env, "gdalsrsinfo", "-o", "proj4", "--single-line", path)
	if err != nil {
		return "EPSG:4326"
	}
	for _, line := range strings.Split(out, "\n") {
		line = strings.Trim(strings.TrimSpace(line), "'\"")
		if strings.HasPrefix(line, "+") {
			return line
		}
	}
	return "EPSG:4326"
}

// --- ogrinfo ----------------------------------------------------------------

type ogrJSON struct {
	Layers []struct {
		Features []struct {
			Geometry *struct {
				Type        string          `json:"type"`
				Coordinates json.RawMessage `json:"coordinates"`
			} `json:"geometry"`
		} `json:"features"`
	} `json:"layers"`
}

// firstFeaturePolygonRing reads the outer ring of the first feature of the
// first layer of a vector datasource (shapefile or GeoJSON). Only plain
// POLYGON geometries are accepted, matching dole_v2.cutline_ring.
func firstFeaturePolygonRing(env []string, path string) ([]Pt, error) {
	out, err := runTool(env, "ogrinfo", "-json", "-features", path)
	if err != nil {
		return nil, err
	}
	var doc ogrJSON
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		return nil, fmt.Errorf("ogrinfo -json parse for %s: %w", path, err)
	}
	if len(doc.Layers) == 0 || len(doc.Layers[0].Features) == 0 {
		return nil, fmt.Errorf("no features in %s", path)
	}
	geom := doc.Layers[0].Features[0].Geometry
	if geom == nil {
		return nil, fmt.Errorf("first feature of %s has no geometry", path)
	}
	if !strings.EqualFold(geom.Type, "Polygon") {
		return nil, fmt.Errorf("cutline is not a polygon: %s", geom.Type)
	}
	var rings [][][]float64
	if err := json.Unmarshal(geom.Coordinates, &rings); err != nil {
		return nil, fmt.Errorf("polygon coordinates parse for %s: %w", path, err)
	}
	if len(rings) == 0 {
		return nil, fmt.Errorf("polygon with no rings in %s", path)
	}
	ring := make([]Pt, 0, len(rings[0]))
	for _, c := range rings[0] {
		if len(c) < 2 {
			continue
		}
		ring = append(ring, Pt{c[0], c[1]})
	}
	return ring, nil
}

// --- WKT --------------------------------------------------------------------

var wktPolygonRe = regexp.MustCompile(`(?is)^\s*POLYGON\s*\(\(\s*(.*?)\s*\)`)

// parseWKTPolygonRing extracts the outer ring of an inline POLYGON WKT.
func parseWKTPolygonRing(wkt string) ([]Pt, error) {
	m := wktPolygonRe.FindStringSubmatch(wkt)
	if m == nil {
		return nil, fmt.Errorf("unparseable cutline_wkt: %.80q", wkt)
	}
	var ring []Pt
	for _, pair := range strings.Split(m[1], ",") {
		fields := strings.Fields(strings.TrimSpace(pair))
		if len(fields) < 2 {
			return nil, fmt.Errorf("bad WKT coordinate pair: %q", pair)
		}
		x, errX := strconv.ParseFloat(fields[0], 64)
		y, errY := strconv.ParseFloat(fields[1], 64)
		if errX != nil || errY != nil {
			return nil, fmt.Errorf("bad WKT coordinate pair: %q", pair)
		}
		ring = append(ring, Pt{x, y})
	}
	if len(ring) == 0 {
		return nil, fmt.Errorf("empty WKT polygon ring")
	}
	return ring, nil
}

// --- spherical mercator ------------------------------------------------------

// mercX/mercY implement the exact EPSG:4326 -> EPSG:3857 forward transform
// (the same formulas the Python code inlined for split-bounds math).
func mercX(lon float64) float64 { return lon / 180.0 * worldMerc }

func mercY(lat float64) float64 {
	lat = math.Max(math.Min(lat, 85.0511), -85.0511)
	return math.Log(math.Tan(math.Pi/4+lat*math.Pi/360)) * worldMerc / math.Pi
}
