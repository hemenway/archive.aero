# GeoTIFF Aligner Tool

A sophisticated desktop application for interactively aligning GeoTIFF images with Lambert Conformal Conic (LCC) projection coordinates.

## Features

- **Interactive Alignment**: Visually align aerial maps with geographic coordinates
- **Dual Control System**: 
  - Pan the viewport independently
  - Move and scale the map image separately
- **Geographic Grid Overlay**: See latitude/longitude lines overlaid on your images
- **Undo/Redo Support**: Full undo/redo capability for all alignment operations
- **Session Persistence**: Automatically save and resume work sessions
- **Batch Processing**: Process multiple TIFF files in sequence
- **GDAL Script Generation**: Automatically generates shell script for batch processing

## Requirements

- Python 3.7+
- Required packages:
  - tkinter (usually included with Python)
  - Pillow (PIL)
  
Install dependencies:
```bash
pip install Pillow
```

## Usage

### Starting the Application

```bash
python transformer.py
```

### Workflow

1. **Load Folder**: Click "1. Load Folder" and select a directory containing TIFF files
2. **Align Image**: 
   - Use mouse and keyboard to align the image with the grid
   - Pan viewport: Left-click and drag
   - Move map: Right-click and drag (or Shift+Left-drag on Mac)
   - Scale map: Use [ and ] keys
3. **Save & Next**: Press Enter or click "2. Save & Next" to save and move to the next image
4. **Batch Process**: After aligning all images, run the generated `run_realignment.sh` script

### Keyboard Shortcuts

#### View Navigation
- `+` / `=` : Zoom in
- `-` / `_` : Zoom out
- `Mouse Wheel` : Zoom at cursor position
- `Left Click + Drag` : Pan viewport

#### Map Alignment
- `Right Click + Drag` : Move map image
- `Shift + Left Drag` : Move map (Mac compatibility)
- `Arrow Keys` : Nudge map position (1 pixel)
- `[` : Scale down (fine adjustment, 0.5%)
- `]` : Scale up (fine adjustment, 0.5%)
- `Shift + [` : Scale down (coarse, 5%)
- `Shift + ]` : Scale up (coarse, 5%)

#### File Operations
- `Enter` : Save alignment and next image

#### Editing
- `Ctrl+Z` : Undo last change
- `Ctrl+Y` / `Ctrl+Shift+Z` : Redo

#### Help
- `Esc` / `F1` : Show keyboard shortcuts help dialog

## Configuration

The application uses the following projection parameters (configurable in the `Config` class):

### Lambert Conformal Conic Projection
- **First Standard Parallel**: 38.67°
- **Second Standard Parallel**: 33.33°
- **Latitude of Origin**: 34.0°
- **Central Meridian**: -98.5°
- **Ellipsoid**: GRS 1980 / NAD83

### Default Georeferencing
- **Base Pixel Width**: 42.33 meters
- **Base Pixel Height**: -42.33 meters (negative = north-up)
- **Default Upper Left X**: -424886.449 meters
- **Default Upper Left Y**: 252841.780 meters

## Output

The tool generates a bash script (`run_realignment.sh`) containing GDAL commands to apply the alignments:

```bash
#!/bin/bash
mkdir -p aligned

echo 'Aligning image1.tif...'
gdal_translate -a_ullr X1 Y1 X2 Y2 -a_srs 'PROJ_STRING' "image1.tif" "aligned/image1.tif"

# ... more commands for each image
```

Run this script to batch process all aligned images:
```bash
chmod +x run_realignment.sh
./run_realignment.sh
```

## Session Management

The tool automatically saves your progress in `.alignment_session.json`. If you close and reopen the tool in the same folder, it will ask if you want to resume where you left off.

## Technical Details

### Architecture

The application consists of three main components:

1. **LCCProjection**: Handles Lambert Conformal Conic projection calculations
2. **SmartAlignApp**: Main application with UI and interaction logic
3. **Config**: Centralized configuration and constants

### Coordinate Systems

- **Geographic Coordinates**: Latitude/Longitude in degrees
- **Projected Coordinates**: X/Y in meters (LCC projection)
- **Screen Coordinates**: Pixels on the canvas

The tool manages three separate coordinate transformations:
1. Geographic → Projected (LCC projection)
2. Projected → Base Image (pixel coordinates at base scale)
3. Base Image → Screen (accounting for viewport pan and zoom)

### State Management

The application maintains separate state for:
- **View State**: Viewport pan and zoom (does not affect output)
- **Alignment State**: Map translation and scaling (affects output)
- **Undo/Redo Stacks**: For reverting changes

## Logging

The application logs to console with the following levels:
- **INFO**: Normal operation events
- **DEBUG**: Detailed state changes
- **WARNING**: Potential issues
- **ERROR**: Errors with stack traces

Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Troubleshooting

### Image Won't Load
- Check that the file is a valid TIFF/GeoTIFF
- Verify sufficient memory for large images
- Check console for error messages

### Grid Not Visible
- Try zooming out with `-` key
- Pan the viewport with left-click drag
- Check that the image is roughly aligned with the projection area

### Script Fails to Run
- Ensure GDAL tools are installed (`gdal_translate`)
- Check script has execute permissions (`chmod +x run_realignment.sh`)
- Verify output directory `aligned/` exists

## Contributing

Suggestions and improvements are welcome! The code is structured to be maintainable and extensible.

## License

Part of the AeroMap project. See main repository for license information.
