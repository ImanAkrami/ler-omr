# OMR Corner Detection & Deskewing Requirements

## Goal
Create a script that can reliably detect, crop, and deskew the wanted area from exam sheets using 4 L-shaped markers.

## Input Conditions
- **Constant**: 4 L-shaped markers on white background (one in each corner)
- **Variable conditions**:
  - Scanned or photographed images
  - Imperfect rotation (slight skew)
  - Perspective distortion (especially mobile photos)
  - Color variations (white may appear off-white, yellowish, etc.)
  - Lighting variations
  - Possible wrinkles, shadows, or other artifacts

## Core Functionality
The script must:
1. **Detect** the 4 L-shaped corner markers using multiple methods
2. **Crop** the image to the area bounded by the markers
3. **Deskew** the cropped area to correct perspective and rotation

## Output Requirements
For each input image, generate:
- `debug.json` - Detection metadata (corners, confidence, method used, etc.)
- `debug.png` - Visual debug image showing detected corners
- `output.png` - Final cropped and deskewed result

## Definition of Done
✅ Script processes all images in `inputs/` folder  
✅ Each image produces:
   - `debug.json` with detection results
   - `debug.png` with corner visualization
   - `output.png` with cropped and deskewed result
✅ Detected corners accurately identify the L-markers (not QR codes or other features)
✅ Cropped output shows the exam sheet area clearly and correctly oriented

## Technical Approach
- Use multiple detection methods (template matching, geometric L-shape detection, Hough lines, etc.)
- Validate detected corners (geometry, position, size)
- Apply perspective transformation for deskewing
- Handle edge cases (poor lighting, shadows, perspective distortion)




