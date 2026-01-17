# Albuquerque Shapefile Comparison

Comparison between two Albuquerque sectional chart shapefiles from different GitHub repositories.

## Sources

1. **CHARTMAKER (N129BZ)**: https://github.com/N129BZ/chartmaker/blob/main/clipshapes/sectional/albuquerque.shp
2. **PIREP (shanet)**: https://github.com/shanet/pirep/blob/master/lib/faa/charts_crop_shapefiles/sectional/albuquerque.shp

## File Composition

### CHARTMAKER (N129BZ) - 6 files
- `albuquerque.cpg` - Character encoding file (UTF-8)
- `albuquerque.dbf` - Attribute database file
- `albuquerque.prj` - Projection file
- `albuquerque.qpj` - QGIS projection file
- `albuquerque.shp` - Main shapefile (380 bytes)
- `albuquerque.shx` - Shape index file (108 bytes)

### PIREP (shanet) - 3 files
- `albuquerque.prj` - Projection file
- `albuquerque.shp` - Main shapefile (444 bytes)
- `albuquerque.shx` - Shape index file (108 bytes)
- **Missing**: DBF file (no attributes)

## Key Differences

### 1. Number of Vertices
- **CHARTMAKER**: 14 points
- **PIREP**: 18 points
- **Impact**: PIREP has 4 more vertices, creating a more detailed polygon boundary

### 2. Bounding Box
| Metric | CHARTMAKER | PIREP | Difference |
|--------|-----------|-------|------------|
| Min X (West) | -108.992623° | -108.999820° | 0.007197° (~800m) |
| Max X (East) | -101.795662° | -101.780962° | 0.014700° (~1.6km) |
| Min Y (South) | 32.001544° | 32.000399° | 0.001145° (~127m) |
| Max Y (North) | 36.215727° | 36.225452° | 0.009725° (~1.1km) |

**PIREP coverage is larger** in all directions, extending the boundary by approximately:
- 800m further west
- 1.6km further east
- 127m further south
- 1.1km further north

### 3. Projection System
Both shapefiles use **identical projection**:
- Coordinate System: GCS_WGS_1984
- Datum: WGS_1984
- Spheroid: WGS_1984 (6378137.0, 298.257223563)
- Prime Meridian: Greenwich
- Units: Degrees

### 4. Attributes
- **CHARTMAKER**: Has DBF file with 1 field (`id: N` - numeric field, size 10), but value is NULL
- **PIREP**: No DBF file, no attributes

### 5. Polygon Detail

#### CHARTMAKER (14 vertices)
The polygon has fewer vertices, with points concentrated at:
- Northern boundary: 2 points
- Southern boundary: 12 points (spaced ~0.5-0.7 degrees apart)
- Creates a simpler polygon with longer straight segments

#### PIREP (18 vertices)
The polygon has more vertices, with points at:
- Northern boundary: 2 points
- Southern boundary: 16 points (spaced ~0.5 degrees apart)
- Creates a more detailed polygon with more vertices along the southern edge

### 6. Geometry Pattern

Both polygons follow the same general pattern:
1. Start at northwest corner
2. Draw line to northeast corner
3. Follow southern boundary from east to west with intermediate points
4. Close polygon back to northwest corner

**PIREP appears to have intermediate points at every 0.5° longitude** along the southern boundary (approximately), while **CHARTMAKER uses a less regular pattern** with fewer points.

## Coordinate Comparison

### Northern Edge
| Source | West Point | East Point |
|--------|-----------|-----------|
| CHARTMAKER | (-108.981499, 36.215727) | (-101.795662, 36.211787) |
| PIREP | (-108.998229, 36.225452) | (-101.780962, 36.219491) |

### Southern Edge Pattern
- **CHARTMAKER**: Irregular spacing, 12 points along southern edge
- **PIREP**: Regular ~0.5° spacing, 16 points along southern edge

## Similarity Analysis

### What's the Same
1. Both use WGS84 geographic coordinate system
2. Both define the Albuquerque sectional chart boundary
3. Both are single-polygon features
4. Both follow the same general boundary pattern
5. Similar coverage area (central New Mexico)

### What's Different
1. **Vertex density**: PIREP has 28% more vertices (18 vs 14)
2. **Bounding box**: PIREP extends further in all directions
3. **Southern edge detail**: PIREP has more regular, denser point spacing
4. **File completeness**: CHARTMAKER has DBF attributes, PIREP doesn't
5. **Additional files**: CHARTMAKER has .cpg and .qpj files

## Conclusion

These shapefiles represent **similar but not identical** boundaries for the Albuquerque sectional chart:

1. **PIREP version is larger and more detailed**
   - Extends coverage by ~1-2km in most directions
   - Uses more vertices for a smoother, more accurate boundary
   - Regular vertex spacing suggests algorithmic generation

2. **CHARTMAKER version is simpler**
   - Slightly smaller coverage area
   - Fewer vertices with irregular spacing
   - Includes attribute table structure (though empty)
   - More complete file set (.cpg, .qpj)

3. **Use Case Recommendations**
   - **For precise boundary operations**: Use PIREP (more vertices, larger coverage)
   - **For simplified visualization**: Use CHARTMAKER (fewer vertices, smaller file)
   - **For attribute-based workflows**: Use CHARTMAKER (has DBF structure)

Both are valid representations of the Albuquerque sectional chart boundary, but they were likely created using different methods or sources, resulting in slightly different interpretations of the same geographic area.
