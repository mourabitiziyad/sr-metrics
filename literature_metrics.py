import os
import sys
import numpy as np
import cv2
from skimage.io import imread
from skimage import img_as_float32, filters, feature
from skimage.measure import shannon_entropy
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def calculate_brisque_opencv(image):
    """Calculate BRISQUE using OpenCV's quality module"""
    try:
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Path to BRISQUE model files
        model_path = "brisque_model_live.yml"
        range_path = "brisque_range_live.yml"
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(range_path):
            print(f"BRISQUE model files not found. Please ensure {model_path} and {range_path} exist.")
            return None
            
        # Initialize BRISQUE quality evaluator with model files
        brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)
        
        # Compute BRISQUE score (lower is better, 0-100 scale typically)
        score = brisque.compute(gray)[0]
        return score
        
    except Exception as e:
        print(f"Error calculating BRISQUE: {e}")
        return None

def calculate_gradient_magnitude(image):
    """Calculate gradient magnitude variance as a sharpness measure"""
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return variance of gradient magnitude (higher = sharper)
    return np.var(gradient_magnitude)

def calculate_laplacian_variance(image):
    """Calculate Laplacian variance (Tenenbaum's algorithm) - standard sharpness metric"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Return variance (higher = sharper)
    return laplacian.var()

def calculate_brenner_sharpness(image):
    """Calculate Brenner's sharpness measure - another standard metric"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Brenner's sharpness: sum of squared differences between adjacent pixels
    gray = gray.astype(np.float64)
    
    # Horizontal differences
    diff_h = np.abs(gray[:, :-2] - gray[:, 2:])
    # Vertical differences  
    diff_v = np.abs(gray[:-2, :] - gray[2:, :])
    
    # Sum of squared differences
    brenner = np.sum(diff_h**2) + np.sum(diff_v**2)
    
    return brenner

def calculate_energy_of_gradient(image):
    """Energy of Gradient - another literature sharpness metric"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Energy of gradient
    energy = np.sum(grad_x**2) + np.sum(grad_y**2)
    
    return energy

def calculate_image_entropy(image):
    """Calculate Shannon entropy of pixel intensities"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    # Remove zero entries
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

def calculate_rms_contrast(image):
    """Calculate RMS contrast"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    mean_intensity = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
    
    return rms_contrast

def calculate_pique(image):
    """
    Calculate PIQUE (Perception based Image QUality Evaluator)
    Based on Venkatanath et al. (2015) - "Blind Image Quality Evaluation Using Perception Based Features"
    
    PIQUE measures perceptual quality by analyzing:
    1. Block-wise distortion estimation
    2. Noise estimation 
    3. Activity/texture analysis
    
    Returns: PIQUE score (lower is better, typically 0-100)
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Ensure float32 for calculations
        gray = gray.astype(np.float32) / 255.0
        
        h, w = gray.shape
        
        # Block size for analysis (16x16 as in original paper)
        block_size = 16
        
        # Calculate number of blocks
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        
        if num_blocks_h < 2 or num_blocks_w < 2:
            # Image too small for reliable PIQUE calculation
            return None
        
        # Initialize arrays for block features
        block_variances = []
        block_entropies = []
        block_activities = []
        
        # Process each block
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # Extract block
                y1, y2 = i * block_size, (i + 1) * block_size
                x1, x2 = j * block_size, (j + 1) * block_size
                block = gray[y1:y2, x1:x2]
                
                # Calculate block variance (measure of activity)
                block_var = np.var(block)
                block_variances.append(block_var)
                
                # Calculate block entropy
                hist, _ = np.histogram(block, bins=32, range=(0, 1))
                hist = hist / np.sum(hist + 1e-7)  # Normalize and avoid log(0)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    entropy = -np.sum(hist * np.log2(hist + 1e-7))
                else:
                    entropy = 0
                block_entropies.append(entropy)
                
                # Calculate spatial activity using gradients
                grad_x = np.gradient(block, axis=1)
                grad_y = np.gradient(block, axis=0)
                activity = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                block_activities.append(activity)
        
        # Convert to numpy arrays
        block_variances = np.array(block_variances)
        block_entropies = np.array(block_entropies)
        block_activities = np.array(block_activities)
        
        # Calculate distortion features
        # High variance blocks are likely to have more visible distortions
        high_activity_mask = block_activities > np.percentile(block_activities, 75)
        
        if np.sum(high_activity_mask) > 0:
            # Distortion in high activity regions (more noticeable)
            high_activity_var = np.mean(block_variances[high_activity_mask])
            high_activity_entropy = np.mean(block_entropies[high_activity_mask])
        else:
            high_activity_var = np.mean(block_variances)
            high_activity_entropy = np.mean(block_entropies)
        
        # Calculate noise estimation
        # Use Laplacian for noise estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        noise_estimate = np.var(laplacian)
        
        # Calculate overall statistics
        mean_variance = np.mean(block_variances)
        std_variance = np.std(block_variances)
        mean_entropy = np.mean(block_entropies)
        mean_activity = np.mean(block_activities)
        
        # PIQUE computation (simplified version)
        # Lower scores indicate better quality
        
        # Distortion component (higher variance in active regions suggests distortion)
        distortion_score = high_activity_var * 100
        
        # Noise component 
        noise_score = noise_estimate * 1000
        
        # Activity component (very low activity might indicate blur)
        activity_penalty = max(0, (0.1 - mean_activity) * 100)
        
        # Uniformity component (high std of variances suggests non-uniform quality)
        uniformity_penalty = std_variance * 100
        
        # Combine components
        pique_score = distortion_score + noise_score + activity_penalty + uniformity_penalty
        
        # Clip to reasonable range
        pique_score = np.clip(pique_score, 0, 100)
        
        return float(pique_score)
        
    except Exception as e:
        print(f"Error calculating PIQUE: {e}")
        return None

def analyze_image_quality_literature(img_path):
    """Analyze image quality using standard literature metrics"""
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return None
    
    try:
        # Load image using OpenCV (BGR format)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Error: Could not load image '{img_path}'")
            return None
        
        # Also load using PIL for alternative processing
        img_pil = Image.open(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Calculate various literature-based metrics
        results = {}
        
        # BRISQUE (if available in OpenCV)
        try:
            results['brisque'] = calculate_brisque_opencv(img_bgr)
        except:
            results['brisque'] = None
            print("BRISQUE not available in this OpenCV version")
        
        # PIQUE (Perception based Image QUality Evaluator)
        results['pique'] = calculate_pique(img_bgr)
        
        # Standard sharpness metrics from literature
        results['laplacian_variance'] = calculate_laplacian_variance(img_bgr)
        results['gradient_magnitude'] = calculate_gradient_magnitude(img_bgr)
        results['brenner_sharpness'] = calculate_brenner_sharpness(img_bgr)
        results['energy_of_gradient'] = calculate_energy_of_gradient(img_bgr)
        
        # Other quality indicators
        results['shannon_entropy'] = calculate_image_entropy(img_bgr)
        results['rms_contrast'] = calculate_rms_contrast(img_bgr)
        
        # Composite quality score (normalized combination)
        # Higher sharpness and contrast = better quality
        if results['laplacian_variance'] is not None and results['rms_contrast'] is not None:
            # Normalize to 0-100 scale approximately
            norm_laplacian = min(100, results['laplacian_variance'] * 1000)
            norm_contrast = min(100, results['rms_contrast'] * 100)
            results['composite_score'] = (norm_laplacian + norm_contrast) / 2
        else:
            results['composite_score'] = None
        
        return results
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def print_literature_results(results, img_path):
    """Print formatted results for literature metrics"""
    print("=" * 70)
    print(f"Literature-Based Image Quality Analysis: {os.path.basename(img_path)}")
    print("=" * 70)
    
    if results['brisque'] is not None:
        print(f"BRISQUE Score:                  {results['brisque']:.4f} (lower = better)")
    else:
        print("BRISQUE Score:                  Not available")
    
    print(f"Laplacian Variance:             {results['laplacian_variance']:.6f} (higher = sharper)")
    print(f"Gradient Magnitude Variance:    {results['gradient_magnitude']:.2f} (higher = sharper)")
    print(f"Brenner Sharpness:              {results['brenner_sharpness']:.2f} (higher = sharper)")
    print(f"Energy of Gradient:             {results['energy_of_gradient']:.2f} (higher = sharper)")
    print(f"Shannon Entropy:                {results['shannon_entropy']:.4f} (higher = more info)")
    print(f"RMS Contrast:                   {results['rms_contrast']:.4f} (higher = better contrast)")
    
    if results['composite_score'] is not None:
        print(f"Composite Quality Score:        {results['composite_score']:.2f}/100")
    
    print("-" * 70)
    print("LITERATURE REFERENCE METRICS:")
    print("• Laplacian Variance: Tenenbaum et al. (1984)")
    print("• Brenner Sharpness: Brenner et al. (1976)")  
    print("• Energy of Gradient: Subbarao et al. (1993)")
    print("• BRISQUE: Mittal et al. (2012) - IEEE TIP")
    print("• Shannon Entropy: Information theory measure")
    print("• RMS Contrast: Standard contrast metric")
    print("-" * 70)
    print("INTERPRETATION FOR SUPER-RESOLUTION:")
    print("• Higher sharpness metrics indicate better detail recovery")
    print("• Higher contrast indicates better preservation of dynamic range")
    print("• Higher entropy indicates more information content")
    print("• BRISQUE: Lower scores indicate better perceptual quality")

def compare_sr_literature(image_paths, labels=None):
    """Compare multiple SR results using literature metrics"""
    
    if labels is None:
        labels = [os.path.basename(path) for path in image_paths]
    
    results = []
    
    print("=" * 80)
    print("SUPER-RESOLUTION COMPARISON - LITERATURE METRICS")
    print("=" * 80)
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        print(f"\nAnalyzing {label}...")
        
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            continue
            
        metrics = analyze_image_quality_literature(path)
        if metrics is None:
            print(f"  ❌ Failed to analyze: {path}")
            continue
            
        metrics['image'] = label
        metrics['path'] = path
        results.append(metrics)
        
        # Print brief summary
        print(f"  Laplacian Var: {metrics['laplacian_variance']:.4f}")
        print(f"  Brenner:       {metrics['brenner_sharpness']:.2f}")
        print(f"  Contrast:      {metrics['rms_contrast']:.4f}")
        if metrics['brisque'] is not None:
            print(f"  BRISQUE:       {metrics['brisque']:.4f}")
    
    if not results:
        print("No images could be analyzed.")
        return None
    
    print("\n" + "=" * 80)
    print("LITERATURE METRICS COMPARISON")
    print("=" * 80)
    
    # Sort by composite score (best first)
    if all(r['composite_score'] is not None for r in results):
        results_sorted = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        
        print("\nRANKING by Composite Score (Best to Worst):")
        print("-" * 50)
        for i, result in enumerate(results_sorted):
            print(f"{i+1}. {result['image']:20} Score: {result['composite_score']:.1f}")
    else:
        results_sorted = results
    
    print(f"\nDETAILED METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Image':<15} {'Laplacian':<10} {'Brenner':<10} {'Contrast':<10} {'Entropy':<8} {'BRISQUE':<8}")
    print("-" * 80)
    
    for result in results_sorted:
        brisque_str = f"{result['brisque']:.2f}" if result['brisque'] is not None else "N/A"
        print(f"{result['image']:<15} "
              f"{result['laplacian_variance']:<10.4f} "
              f"{result['brenner_sharpness']:<10.0f} "
              f"{result['rms_contrast']:<10.4f} "
              f"{result['shannon_entropy']:<8.2f} "
              f"{brisque_str:<8}")
    
    print("\n" + "=" * 80)
    print("METRICS EXPLANATION:")
    print("• Laplacian Variance: Measures edge sharpness (Tenenbaum 1984)")
    print("• Brenner Sharpness: Measures focus quality (Brenner 1976)")
    print("• RMS Contrast: Measures contrast preservation")
    print("• Shannon Entropy: Measures information content")
    print("• BRISQUE: Perceptual quality (Mittal et al. 2012)")
    print("=" * 80)
    
    return results_sorted

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to the existing image path
        img_path = 'set 2/satlas.png'
    
    results = analyze_image_quality_literature(img_path)
    if results:
        print_literature_results(results, img_path) 