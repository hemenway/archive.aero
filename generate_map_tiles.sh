#!/bin/bash
set -e

CHARTS=("Seattle" "Klamath Falls" "Great Falls" "Salt Lake City")

rm all_charts.vrt || true
rm -rf webviewer/tiles || true

for CHART in "${CHARTS[@]}"; do
  unzip -o "${CHART// /_}.zip"

  gdalwarp \
    -t_srs EPSG:3857 \
    -co TILED=YES \
    -dstalpha \
    -of GTiff \
    -cutline "shapefiles/$CHART.shp" \
    -crop_to_cutline \
    -wo NUM_THREADS=`grep -c ^processor /proc/cpuinfo` \
    -multi \
    -overwrite \
    "$CHART SEC.tif" \
    "$CHART cropped.tif"

  gdal_translate -of vrt -expand rgba "$CHART cropped.tif" "$CHART.vrt"
done

gdalbuildvrt all_charts.vrt *.vrt

gdal2tiles.py \
  --zoom "0-11" \
  --processes=`grep -c ^processor /proc/cpuinfo` \
  --webviewer=none \
  --exclude \
  --tiledriver=WEBP \
  --webp-quality=50 \
  all_charts.vrt \
  webviewer/tiles
