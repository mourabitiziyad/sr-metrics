# SR Metrics

A comprehensive suite of tools for analyzing image quality metrics, particularly focused on evaluating super-resolution algorithms. The toolkit provides both traditional reference-based metrics (SSIM, PSNR, RMSE) and advanced no-reference quality assessment using state-of-the-art literature-based methods.

## Overview

This toolkit offers multiple approaches to image quality evaluation:

### Reference-Based Metrics
- **SSIM**: Structural Similarity Index (closer to 1.0 is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)  
- **RMSE**: Root Mean Square Error (lower is better)

### No-Reference Quality Metrics
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator (lower is better)
- **PIQUE**: Perception based Image Quality Evaluator (lower is better)
- **Laplacian Variance**: Edge sharpness measure (higher is better)
- **Brenner Sharpness**: Focus quality metric (higher is better)
- **RMS Contrast**: Standard contrast measure
- **Shannon Entropy**: Information content measure

## Features

- **Multi-format Support**: Handles TIFF, PNG, and other common image formats
- **Geo-referenced Images**: Automatic CRS detection and handling for satellite imagery
- **Multi-band Processing**: Supports satellite and hyperspectral imagery
- **Intelligent Resizing**: Automatic dimension matching with aspect ratio preservation
- **Batch Processing**: Analyze multiple images and algorithms simultaneously
- **Literature-based Metrics**: State-of-the-art no-reference quality assessment
- **Comprehensive Analysis**: Statistical summaries and quality rankings
- **Export Capabilities**: CSV output for further analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure BRISQUE model files are present:
   - `brisque_model_live.yml`
   - `brisque_range_live.yml`

## Usage

### 1. Basic Super-Resolution Comparison (`main.py`)

Compare two images with traditional metrics:

```bash
# Using positional arguments
python main.py original.tif sr_output.tif

# Using flags with default paths
python main.py --original path/to/original.tif --sr path/to/sr_image.tif
```

**Features:**
- Automatic downsampling of SR images by factor 4 to match original dimensions
- Per-band metric calculation for multi-band images
- Average metrics across all bands

**Example Output:**
```
Band 1: PSNR=62.52, SSIM=0.9959, RMSE=0.07
Band 2: PSNR=65.25, SSIM=0.9978, RMSE=0.05
Band 3: PSNR=67.45, SSIM=0.9987, RMSE=0.04

Average PSNR: 64.61
Average SSIM: 0.9975
Average RMSE: 0.06
```

### 2. Batch Algorithm Comparison (`batch_metrics.py`)

Compare multiple super-resolution algorithms across different image sets:

```bash
python batch_metrics.py
```

**Features:**
- Processes multiple image sets automatically
- Compares ESRGAN vs S2DR3 algorithms
- Statistical analysis and winner determination
- Configurable base path for organized datasets

**Expected Directory Structure:**
```
/path/to/datasets/
├── set 1/
│   ├── original.tif
│   ├── esrgan.tif
│   └── s2dr3.tif
├── set 2/
│   └── ...
```

### 3. Comprehensive PNG Analysis (`analyze_all_pngs.py`)

Analyze all PNG files in the directory tree with advanced metrics:

```bash
python analyze_all_pngs.py
```

**Features:**
- Recursive PNG file discovery
- Literature-based quality metrics (BRISQUE, PIQUE)
- Sharpness analysis (Laplacian, Brenner)
- Quality rankings and statistical summaries
- Optional CSV export with pandas

**Analysis Output:**
- Comprehensive quality tables
- Top performers by different metrics
- Quality distribution analysis
- Statistical summaries (min, max, average, median)

### 4. Literature-Based Metrics (`literature_metrics.py`)

Advanced standalone analysis using state-of-the-art metrics:

```python
from literature_metrics import analyze_image_quality_literature, calculate_brisque_opencv

# Analyze single image
results = analyze_image_quality_literature("image.png")

# Calculate BRISQUE score
import cv2
image = cv2.imread("image.png")
brisque_score = calculate_brisque_opencv(image)
```

## Supported Metrics

### Traditional Reference Metrics
- **SSIM** (Wang et al., 2004): Structural similarity assessment
- **PSNR**: Signal-to-noise ratio in decibels
- **RMSE**: Root mean square error

### Literature-Based No-Reference Metrics
- **BRISQUE** (Mittal et al., 2012): Spatial domain quality assessment
- **PIQUE** (Venkatanath et al., 2015): Perception-based quality evaluation
- **Laplacian Variance** (Tenenbaum et al., 1984): Edge-based sharpness
- **Brenner Sharpness** (Brenner et al., 1976): Focus quality measure
- **Gradient Magnitude**: Edge strength analysis
- **Energy of Gradient**: Alternative sharpness metric

## Image Format Support

- **TIFF/TIF**: Multi-band support via rasterio, geo-referencing preservation
- **PNG**: RGB and grayscale images
- **Other formats**: All formats supported by PIL/Pillow and OpenCV

## Quality Interpretation

### BRISQUE Scores
- **0-10**: Exceptional quality
- **10-20**: Very good quality  
- **20-30**: Good quality
- **30-50**: Fair quality
- **50-70**: Poor quality
- **70+**: Very poor quality

### General Guidelines
- **PSNR**: Higher values indicate better quality (typically 20-50 dB)
- **SSIM**: Values closer to 1.0 indicate better structural similarity
- **RMSE**: Lower values indicate less error
- **PIQUE**: Lower values indicate better perceptual quality

## Advanced Features

### Geo-referenced Image Support
- Automatic CRS detection and preservation
- Compatible with satellite imagery and drone mapping outputs
- UTM and other projection system support

### Batch Processing Capabilities
- Recursive directory processing
- Multiple algorithm comparison
- Statistical analysis and ranking
- Export functionality for further analysis

### Aspect Ratio Handling
- Intelligent resizing with aspect ratio preservation
- Center cropping/padding to match dimensions
- Prevents distortion in metric calculations

## Dependencies

- **numpy**: Numerical computing
- **scikit-image**: Image processing and metrics
- **rasterio**: Geospatial raster data handling
- **OpenCV**: Computer vision and BRISQUE calculation
- **matplotlib**: Visualization capabilities
- **tabulate**: Table formatting
- **PIL/Pillow**: Image I/O operations
- **PyTorch**: Deep learning framework support

## Example Workflows

### Super-Resolution Evaluation
```bash
# Compare single algorithm
python main.py original.tif esrgan.tif

# Batch compare multiple algorithms
python batch_metrics.py

# Comprehensive quality analysis
python analyze_all_pngs.py
```

### Research Applications
- Super-resolution algorithm benchmarking
- Image restoration quality assessment
- Satellite imagery enhancement evaluation
- General image quality research

## Output Examples

### Individual Comparison
```
Band 1: PSNR=62.52, SSIM=0.9959, RMSE=0.07
Band 2: PSNR=65.25, SSIM=0.9978, RMSE=0.05
Band 3: PSNR=67.45, SSIM=0.9987, RMSE=0.04
```

### Batch Comparison
```
SUPER-RESOLUTION METRICS COMPARISON
Set    Model      PSNR     SSIM     RMSE    
1      esrgan.tif  64.61    0.9975   0.06
1      s2dr3.tif   61.97    0.9969   0.05
2      esrgan.tif  63.28    0.9984   0.07
2      s2dr3.tif   60.15    0.9962   0.06

🏆 ESRGAN wins on PSNR: 63.95 vs 61.06
🏆 ESRGAN wins on SSIM: 0.9980 vs 0.9966
```

### Comprehensive Analysis
```
🎯 Top 5 by BRISQUE (Best Quality):
  1. high_quality_image.png        Score:  12.45 (Very good quality)
  2. moderate_image.png            Score:  18.73 (Very good quality)
  3. standard_image.png            Score:  25.12 (Good quality)
```

## Contributing

This toolkit is designed for research and evaluation of image quality metrics. Feel free to extend with additional metrics or improve existing implementations.

## References

- Mittal, A., et al. (2012). "No-reference image quality assessment in the spatial domain"
- Venkatanath, N., et al. (2015). "Blind image quality evaluation using perception based features"
- Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity"
- Tenenbaum, J. M., et al. (1984). "Accommodation-based computer vision" 