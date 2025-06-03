import numpy as np
import rasterio
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error
from skimage.transform import resize

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate super-resolution metrics between original and SR images')
parser.add_argument('--original', type=str, default="set 6/original.tif", 
                    help='Path to the original image')
parser.add_argument('--sr', type=str, default="set 6/s2dr3.tif",
                    help='Path to the super-resolved image')
parser.add_argument('images', nargs='*', 
                    help='Image paths: first is original, second is super-resolved (overrides --original and --sr)')

args = parser.parse_args()

# Determine image paths
if args.images:
    if len(args.images) < 2:
        parser.error("When using positional arguments, you must provide both original and super-resolved image paths")
    original_path = args.images[0]
    sr_path = args.images[1]
    print(f"Using positional arguments: original='{original_path}', sr='{sr_path}'")
else:
    original_path = args.original
    sr_path = args.sr
    print(f"Using default/flag arguments: original='{original_path}', sr='{sr_path}'")

# Load the original image
with rasterio.open(original_path) as src:
    original = src.read().astype(np.float32)

# Load the super-resolved image (10x)
with rasterio.open(sr_path) as src:
    sr = src.read().astype(np.float32)

# Downsample SR image by factor 4 to match original
factor = 4
downsampled_sr = np.array([
    resize(band, (original.shape[1], original.shape[2]), mode="reflect", anti_aliasing=True, preserve_range=True)
    for band in sr
])

# Calculate metrics
psnr_vals, ssim_vals, rmse_vals = [], [], []
for i in range(original.shape[0]):
    o = original[i]
    s = downsampled_sr[i]
    psnr_vals.append(psnr(o, s, data_range=o.max() - o.min()))
    ssim_vals.append(ssim(o, s, data_range=o.max() - o.min()))
    rmse_vals.append(np.sqrt(mean_squared_error(o, s)))

# Print results
for i, (p, s, r) in enumerate(zip(psnr_vals, ssim_vals, rmse_vals)):
    print(f"Band {i+1}: PSNR={p:.2f}, SSIM={s:.4f}, RMSE={r:.2f}")

print(f"\nAverage PSNR: {np.mean(psnr_vals):.2f}")
print(f"Average SSIM: {np.mean(ssim_vals):.4f}")
print(f"Average RMSE: {np.mean(rmse_vals):.2f}")
