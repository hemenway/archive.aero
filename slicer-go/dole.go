package main

// Port of scripts/dole_v2.py — canonical loader for the v2 dole schema
// (master_dole_v2.csv).
//
// Schema (one row per map):
//   identity:   filename, download_link, location, date, end_date, edition, note
//   gcps:       gcp1..gcp4 (TL, TR, BR, BL) x (px, py, lat, lon)
//   cutline:    cutline (shapefile ref relative to shapefiles/) or cutline_wkt
//               (inline POLYGON, lon/lat NAD83; overrides cutline)
//   projection: lcc_lat1, lcc_lat2, lcc_lat0, lcc_lon0
//   rotation:   optional degrees CLOCKWISE (0/90/180/270) the raw scan must be
//               turned to read upright; GCP pixels are stored in that rotated
//               display frame.

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var v2Fields = []string{
	"filename", "download_link", "location", "date", "end_date", "edition", "note",
	"gcp1_px", "gcp1_py", "gcp1_lat", "gcp1_lon",
	"gcp2_px", "gcp2_py", "gcp2_lat", "gcp2_lon",
	"gcp3_px", "gcp3_py", "gcp3_lat", "gcp3_lon",
	"gcp4_px", "gcp4_py", "gcp4_lat", "gcp4_lon",
	"cutline", "cutline_wkt",
	"lcc_lat1", "lcc_lat2", "lcc_lat0", "lcc_lon0",
	"rotation", "src_crs",
}

// Columns that may be absent from a CSV on disk (added after the v2 freeze).
var v2OptionalFields = map[string]bool{"rotation": true, "src_crs": true}

const lccTemplate = "+proj=lcc +lat_1=%s +lat_2=%s +lat_0=%s +lon_0=%s +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

// Cutline geometry is authored in NAD83 lon/lat.
const cutlineDefaultSRS = "EPSG:4269"

// Row mirrors csv.DictReader: column name -> raw string value, plus the
// runtime keys the slicer adds (wayback_ts, candidate_group, ...).
type Row map[string]string

// Pt is a 2D point (pixel or geographic/projected coordinate).
type Pt struct{ X, Y float64 }

// Cutline is the resolved cutline of a row: inline WKT or a shapefile ref.
type Cutline struct {
	Kind string // "wkt" | "shapefile"
	WKT  string
	Path string
	Ref  string
}

func fnum(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	f, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

// missingRequiredFields returns required v2 columns absent from a CSV header.
func missingRequiredFields(fieldnames []string) []string {
	present := map[string]bool{}
	for _, f := range fieldnames {
		present[f] = true
	}
	var missing []string
	for _, k := range v2Fields {
		if !present[k] && !v2OptionalFields[k] {
			missing = append(missing, k)
		}
	}
	return missing
}

// loadRows reads a v2 dole CSV into Row maps (DictReader semantics: short
// records pad with "", long records drop extras).
func loadRows(csvPath string) ([]Row, error) {
	f, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Strip a UTF-8 BOM if present (Python used encoding="utf-8-sig").
	br := newBOMSkippingReader(f)
	reader := csv.NewReader(br)
	reader.FieldsPerRecord = -1

	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", csvPath, err)
	}
	if missing := missingRequiredFields(header); len(missing) > 0 {
		return nil, fmt.Errorf("%s: not a v2 dole CSV, missing %v", csvPath, missing)
	}

	var rows []Row
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", csvPath, err)
		}
		row := Row{}
		for i, name := range header {
			if i < len(rec) {
				row[name] = rec[i]
			} else {
				row[name] = ""
			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

type bomSkippingReader struct {
	r       io.Reader
	skipped bool
}

func newBOMSkippingReader(r io.Reader) io.Reader { return &bomSkippingReader{r: r} }

func (b *bomSkippingReader) Read(p []byte) (int, error) {
	if !b.skipped {
		b.skipped = true
		head := make([]byte, 3)
		n, err := io.ReadFull(b.r, head)
		if n == 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF {
			return b.r.Read(p)
		}
		copied := copy(p, head[:n])
		if err != nil && err != io.ErrUnexpectedEOF {
			return copied, err
		}
		return copied, nil
	}
	return b.r.Read(p)
}

// rowGCPPixels returns the 4 pixel-space corners (TL, TR, BR, BL) or nil.
func rowGCPPixels(row Row) []Pt {
	pixels := make([]Pt, 0, 4)
	for corner := 1; corner <= 4; corner++ {
		px, okX := fnum(row[fmt.Sprintf("gcp%d_px", corner)])
		py, okY := fnum(row[fmt.Sprintf("gcp%d_py", corner)])
		if !okX || !okY {
			return nil
		}
		pixels = append(pixels, Pt{px, py})
	}
	return pixels
}

// rowGCPLonLat returns the 4 geographic corners as (lon, lat), NAD83, or nil.
func rowGCPLonLat(row Row) []Pt {
	coords := make([]Pt, 0, 4)
	for corner := 1; corner <= 4; corner++ {
		lat, okLat := fnum(row[fmt.Sprintf("gcp%d_lat", corner)])
		lon, okLon := fnum(row[fmt.Sprintf("gcp%d_lon", corner)])
		if !okLat || !okLon {
			return nil
		}
		coords = append(coords, Pt{lon, lat})
	}
	return coords
}

// rowLCCCRS synthesizes a Proj4 LCC string from the lcc_* columns, or "".
func rowLCCCRS(row Row) string {
	lat1 := strings.TrimSpace(row["lcc_lat1"])
	lat2 := strings.TrimSpace(row["lcc_lat2"])
	lat0 := strings.TrimSpace(row["lcc_lat0"])
	lon0 := strings.TrimSpace(row["lcc_lon0"])
	if lat1 == "" || lat2 == "" || lat0 == "" || lon0 == "" {
		return ""
	}
	return fmt.Sprintf(lccTemplate, lat1, lat2, lat0, lon0)
}

// rowCutline resolves the row's cutline. cutline_wkt wins over cutline.
func rowCutline(row Row, shapeDir string) *Cutline {
	if wkt := strings.TrimSpace(row["cutline_wkt"]); wkt != "" {
		return &Cutline{Kind: "wkt", WKT: wkt}
	}
	if ref := strings.TrimSpace(row["cutline"]); ref != "" {
		return &Cutline{
			Kind: "shapefile",
			Path: filepath.Join(shapeDir, ref+".shp"),
			Ref:  ref,
		}
	}
	return nil
}

// rowSrcCRS returns the declared CRS of a pre-georeferenced source file, or
// "" (trust the file's own embedded CRS). Used for jpg + world-file sources:
// GDAL reads the .JGW but never the .prj sidecar, and a warp without -s_srs
// silently passes degrees through as target-CRS meters.
func rowSrcCRS(row Row) string {
	return strings.TrimSpace(row["src_crs"])
}

// rowRotation returns the display rotation in degrees clockwise: 0/90/180/270.
func rowRotation(row Row) int {
	value := strings.TrimSpace(row["rotation"])
	if value == "" {
		return 0
	}
	f, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0
	}
	rot := ((int(f) % 360) + 360) % 360
	if rot == 90 || rot == 180 || rot == 270 {
		return rot
	}
	return 0
}

// pxDisplayToRaw maps a display-frame pixel back into the raw file's frame.
// rawW/rawH are always the RAW file's dimensions (pre-rotation).
func pxDisplayToRaw(dx, dy float64, rotation int, rawW, rawH float64) Pt {
	switch rotation {
	case 90:
		return Pt{dy, rawH - dx}
	case 180:
		return Pt{rawW - dx, rawH - dy}
	case 270:
		return Pt{rawW - dy, dx}
	}
	return Pt{dx, dy}
}
