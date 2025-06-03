import numpy as np
import rasterio
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error
from skimage.transform import resize

def calculate_metrics(original_path, sr_path):
    """Calculate metrics between original and SR image"""
    try:
        # Load images
        with rasterio.open(original_path) as src:
            original = src.read().astype(np.float32)
        
        with rasterio.open(sr_path) as src:
            sr = src.read().astype(np.float32)
        
        downsampled_sr = np.array([
            resize(band, (original.shape[1], original.shape[2]), mode="reflect", anti_aliasing=True, preserve_range=True)
            for band in sr
        ])
        
        # Calculate metrics for each band
        psnr_vals, ssim_vals, rmse_vals = [], [], []
        for i in range(min(original.shape[0], downsampled_sr.shape[0])):
            o = original[i]
            s = downsampled_sr[i]
            psnr_vals.append(psnr(o, s, data_range=o.max() - o.min()))
            ssim_vals.append(ssim(o, s, data_range=o.max() - o.min()))
            rmse_vals.append(np.sqrt(mean_squared_error(o, s)))
        
        # Return average metrics
        return {
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals),
            'rmse': np.mean(rmse_vals)
        }
    
    except Exception as e:
        print(f"Error processing {original_path} vs {sr_path}: {e}")
        return None

def batch_process():
    """Process all image sets"""
    base_path = "/Users/I752629/Desktop/Reference Thesis Images/Final Set"
    
    # Models to compare
    models = ['esrgan.tif', 's2dr3.tif']
    
    # Store results
    results = {model: [] for model in models}
    
    print("="*80)
    print("SUPER-RESOLUTION METRICS COMPARISON")
    print("="*80)
    print(f"{'Set':<6} {'Model':<10} {'PSNR':<8} {'SSIM':<8} {'RMSE':<8}")
    print("-"*80)
    
    # Process each set
    for set_num in range(1, 7):
        set_path = os.path.join(base_path, f"set {set_num}")
        original_path = os.path.join(set_path, "original.tif")
        
        if not os.path.exists(original_path):
            print(f"Warning: {original_path} not found")
            continue
        
        # Test each model
        for model in models:
            sr_path = os.path.join(set_path, model)
            
            if not os.path.exists(sr_path):
                print(f"Warning: {sr_path} not found")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(original_path, sr_path)
            
            if metrics:
                results[model].append(metrics)
                print(f"{set_num:<6} {model:<10} {metrics['psnr']:<8.2f} {metrics['ssim']:<8.4f} {metrics['rmse']:<8.2f}")
            else:
                print(f"{set_num:<6} {model:<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}")
    
    print("-"*80)
    
    # Calculate and display averages
    print("\nAVERAGE RESULTS:")
    print("="*50)
    print(f"{'Model':<10} {'Avg PSNR':<10} {'Avg SSIM':<10} {'Avg RMSE':<10}")
    print("-"*50)
    
    for model in models:
        if results[model]:
            avg_psnr = np.mean([r['psnr'] for r in results[model]])
            avg_ssim = np.mean([r['ssim'] for r in results[model]])
            avg_rmse = np.mean([r['rmse'] for r in results[model]])
            
            print(f"{model:<10} {avg_psnr:<10.2f} {avg_ssim:<10.4f} {avg_rmse:<10.2f}")
        else:
            print(f"{model:<10} {'NO DATA':<10} {'NO DATA':<10} {'NO DATA':<10}")
    
    print("="*50)
    
    # Show which model performed better
    if results['esrgan.tif'] and results['s2dr3.tif']:
        esrgan_ssim = np.mean([r['ssim'] for r in results['esrgan.tif']])
        s2dr3_ssim = np.mean([r['ssim'] for r in results['s2dr3.tif']])
        
        esrgan_psnr = np.mean([r['psnr'] for r in results['esrgan.tif']])
        s2dr3_psnr = np.mean([r['psnr'] for r in results['s2dr3.tif']])
        
        print("\nCOMPARISON:")
        if esrgan_ssim > s2dr3_ssim:
            print(f"🏆 ESRGAN wins on SSIM: {esrgan_ssim:.4f} vs {s2dr3_ssim:.4f}")
        else:
            print(f"🏆 S2DR3 wins on SSIM: {s2dr3_ssim:.4f} vs {esrgan_ssim:.4f}")
            
        if esrgan_psnr > s2dr3_psnr:
            print(f"🏆 ESRGAN wins on PSNR: {esrgan_psnr:.2f} vs {s2dr3_psnr:.2f}")
        else:
            print(f"🏆 S2DR3 wins on PSNR: {s2dr3_psnr:.2f} vs {esrgan_psnr:.2f}")

if __name__ == "__main__":
    batch_process() 