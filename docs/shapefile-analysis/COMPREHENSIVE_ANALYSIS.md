# Comprehensive Shapefile Quality Analysis

**Analysis Date:** 2026-01-17
**Comparison:** N129BZ/chartmaker vs shanet/pirep sectional chart shapefiles
**Sample Size:** 8 diverse sectional charts across United States

---

## Executive Summary

**🏆 WINNER: CHARTMAKER (N129BZ)**

After comprehensive analysis of 8 sectional chart shapefiles from different geographic regions, **CHARTMAKER is the clear winner** for use in the archive.aero processing pipeline due to 100% coordinate system consistency and complete file sets.

**Critical Finding:** PIREP repository contains **2 out of 8 files (25%) with projected coordinates** instead of geographic coordinates, which will cause processing failures in GDAL georeferencing workflows.

---

## Methodology

### Shapefiles Analyzed

| Chart Name | Region | Coverage |
|-----------|---------|----------|
| Albuquerque | Southwest | New Mexico |
| Atlanta | Southeast | Georgia, Alabama |
| Seattle | Northwest | Washington, British Columbia |
| Chicago | Midwest | Illinois, Wisconsin, Indiana |
| Denver | Mountain West | Colorado, Wyoming |
| Miami | Southeast Coast | Florida, Bahamas |
| Phoenix | Southwest | Arizona |
| Anchorage | Alaska | South-central Alaska |

### Analysis Criteria

1. **Coordinate System Consistency** - All files must use geographic coordinates (WGS84)
2. **File Completeness** - Presence of DBF, PRJ, CPG, and other support files
3. **Vertex Density** - Number of vertices defining polygon boundaries
4. **Bounding Box Accuracy** - Coverage area precision
5. **Production Reliability** - Consistency across all files

---

## Critical Findings

### 1. Coordinate System Consistency

| Repository | Valid Files | Projected Files | Success Rate |
|-----------|------------|-----------------|--------------|
| **CHARTMAKER** | **8/8** | **0** | **100%** ✓ |
| PIREP | 6/8 | 2 | 75% ✗ |

**Problematic PIREP Files:**
- `seattle.shp` - Lambert Conformal Conic projection (X: -318037 to 346677)
- `denver.shp` - Lambert Conformal Conic projection (X: -316796 to 322761)

**Impact:** These projected coordinate files will cause:
- GDAL processing failures
- Incorrect georeferencing results
- Web map rendering errors
- Coordinate transformation issues

### 2. File Completeness

**CHARTMAKER** (6 files per shapefile):
- ✓ `.shp` - Main geometry file
- ✓ `.shx` - Shape index file
- ✓ `.dbf` - Attribute database
- ✓ `.prj` - Projection definition
- ✓ `.cpg` - Character encoding (UTF-8)
- ✓ `.qpj` - QGIS projection file

**PIREP** (3 files per shapefile):
- ✓ `.shp` - Main geometry file
- ✓ `.shx` - Shape index file
- ✓ `.prj` - Projection definition
- ✗ `.dbf` - Missing (no attributes)
- ✗ `.cpg` - Missing (no encoding info)
- ✗ `.qpj` - Missing

### 3. Vertex Density Analysis (Valid Files Only)

| Chart | CHARTMAKER Vertices | PIREP Vertices | Difference | Winner |
|-------|-------------------|----------------|------------|--------|
| Albuquerque | 14 | 18 | +4 (+28.6%) | PIREP |
| Atlanta | 9 | 18 | +9 (+100.0%) | PIREP |
| Chicago | 29 | 20 | -9 (-31.0%) | CHARTMAKER |
| Miami | 11 | 15 | +4 (+36.4%) | PIREP |
| Phoenix | 29 | 7 | -22 (-75.9%) | CHARTMAKER |
| Anchorage | 8 | 26 | +18 (+225.0%) | PIREP |

**Statistics:**
- CHARTMAKER: Mean=16.7, Median=12.5, StdDev=9.8
- PIREP: Mean=17.3, Median=18.0, StdDev=6.3

**Winner:** PIREP has higher vertex density in 4/6 valid files (67%), but the difference is not significant enough to overcome coordinate system issues.

### 4. Bounding Box Coverage

Average total bounding box expansion between repositories: **0.1012°**

PIREP generally has slightly larger coverage (extends boundaries by 100-2000m), but this varies by chart and may indicate less precise boundary definition.

---

## Detailed Comparison: Albuquerque (Example)

### File Composition
- **CHARTMAKER**: 6 files (380 bytes .shp)
- **PIREP**: 3 files (444 bytes .shp)

### Geometry
- **CHARTMAKER**: 14 vertices, irregular spacing
- **PIREP**: 18 vertices, regular ~0.5° spacing

### Coverage Differences
| Direction | Difference |
|-----------|-----------|
| West | 0.0072° (~800m) |
| East | 0.0147° (~1.6km) |
| South | 0.0011° (~127m) |
| North | 0.0097° (~1.1km) |

PIREP extends further in all directions.

### Projection
Both use identical WGS84 geographic coordinates:
```
GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]
```

---

## Quality Score Comparison

| Criterion | CHARTMAKER | PIREP | Winner |
|-----------|-----------|-------|---------|
| Coordinate Consistency | 8/8 (100%) | 6/8 (75%) | **CHARTMAKER** |
| File Completeness | 6/6 with DBF (100%) | 0/6 with DBF (0%) | **CHARTMAKER** |
| Vertex Density | Mean 16.7 | Mean 17.3 | PIREP |
| Coverage Precision | Standard | +0.1° average | Neutral |
| Production Reliability | ✓ Consistent | ✗ Inconsistent | **CHARTMAKER** |

**Overall Winner: CHARTMAKER** (3 major criteria vs 1 minor criteria)

---

## Strengths and Weaknesses

### CHARTMAKER Strengths
1. ✓ **100% geographic coordinate consistency** - Critical for GDAL workflows
2. ✓ **Complete file sets** - DBF attributes, character encoding, QGIS support
3. ✓ **Production reliability** - Consistent format across all files
4. ✓ **No surprises** - Predictable structure for automated processing

### CHARTMAKER Weaknesses
1. ✗ Generally lower vertex density (simpler polygons)
2. ✗ Slightly smaller coverage areas in some cases

### PIREP Strengths
1. ✓ **Higher vertex density** in most valid files (more detailed polygons)
2. ✓ Regular vertex spacing (algorithmic generation)
3. ✓ Slightly larger coverage areas in most cases

### PIREP Weaknesses
1. ✗ **25% of files use projected coordinates** - CRITICAL FAILURE
2. ✗ Missing DBF attribute files
3. ✗ Missing character encoding files
4. ✗ Inconsistent coordinate system usage
5. ✗ Will cause processing failures in standard workflows

---

## Impact on archive.aero Processing Pipeline

### Using CHARTMAKER (Recommended)
- ✓ All files will process correctly through GDAL
- ✓ Consistent coordinate transformation behavior
- ✓ No unexpected failures
- ✓ Complete metadata preservation via DBF
- ✓ Proper character encoding handling

### Using PIREP (Not Recommended)
- ✗ Seattle and Denver files will fail or produce incorrect results
- ✗ Requires manual inspection and filtering of each file
- ✗ No attribute data available
- ✗ Potential character encoding issues
- ⚠ Would need custom code to detect and handle projected vs geographic files

---

## Recommendations

### Primary Recommendation

**Use CHARTMAKER shapefiles** as the standard source for archive.aero sectional chart processing.

**Rationale:**
1. Coordinate system consistency is **non-negotiable** for GDAL georeferencing
2. The 25% failure rate in PIREP is unacceptable for production
3. Complete file sets provide better long-term maintainability
4. Vertex density differences are minor and don't justify the risks

### Alternative Approach (If Needed)

For specific charts where higher vertex density is desired:
1. Use CHARTMAKER as the baseline
2. Individually verify PIREP files for geographic coordinates
3. Only substitute PIREP files after verification
4. Document any substitutions clearly

**Example verification code:**
```python
def is_geographic(shp_path):
    sf = shapefile.Reader(shp_path)
    bbox = sf.bbox
    return (-180 <= bbox[0] <= 180 and
            -180 <= bbox[2] <= 180 and
            -90 <= bbox[1] <= 90 and
            -90 <= bbox[3] <= 90)
```

### Implementation for archive.aero

Update processing scripts to:
1. Source shapefiles from `https://github.com/N129BZ/chartmaker/tree/main/clipshapes/sectional`
2. Download all 6 file components (.shp, .shx, .dbf, .prj, .cpg, .qpj)
3. Verify geographic coordinates before processing
4. Use DBF attributes if needed for metadata

---

## Conclusion

While PIREP shapefiles offer higher vertex density in some cases, the **critical coordinate system inconsistencies** (25% failure rate) and **missing attribute files** make CHARTMAKER the clear choice for production use.

**CHARTMAKER provides:**
- ✓ 100% reliability
- ✓ Complete metadata
- ✓ Consistent processing
- ✓ Adequate geometric precision

**For archive.aero:** Use CHARTMAKER shapefiles to ensure robust, reliable, and maintainable georeferencing workflows.

---

## Appendix: Test Results Summary

```
COORDINATE SYSTEM CONSISTENCY TEST:
  CHARTMAKER: 8/8 files PASSED (100%)
  PIREP:      6/8 files PASSED (75%)

FILE COMPLETENESS TEST:
  CHARTMAKER: 6/6 files with complete file sets (100%)
  PIREP:      0/6 files with DBF attributes (0%)

VERTEX DENSITY TEST:
  PIREP has higher density in 4/6 cases
  Average difference: 3.5% higher in PIREP

RELIABILITY SCORE:
  CHARTMAKER: 100% (no failures)
  PIREP:      75% (2 coordinate system failures)
```

---

## References

- CHARTMAKER Repository: https://github.com/N129BZ/chartmaker
- PIREP Repository: https://github.com/shanet/pirep
- Analysis Scripts: `/docs/shapefile-analysis/`
- Test Data: 8 sectional charts across US geographic regions
