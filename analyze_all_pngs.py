import os
import glob
import cv2
import numpy as np
from literature_metrics import analyze_image_quality_literature, calculate_brisque_opencv, calculate_pique

def find_all_pngs():
    """Find all PNG files in the directory tree, excluding system/library files"""
    
    # Find all PNG files
    all_pngs = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.venv') and d != '__pycache__']
        
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                all_pngs.append(full_path)
    
    return sorted(all_pngs)

def get_quality_interpretation(raw_brisque):
    """Provide interpretation of raw BRISQUE score"""
    if raw_brisque is None:
        return "Unable to calculate"
    elif raw_brisque < 10:
        return "Exceptional quality"
    elif raw_brisque < 20:
        return "Very good quality"
    elif raw_brisque < 30:
        return "Good quality"
    elif raw_brisque < 50:
        return "Fair quality"
    elif raw_brisque < 70:
        return "Poor quality"
    else:
        return "Very poor quality"

def analyze_all_png_files():
    """Analyze all PNG files with comprehensive metrics"""
    
    png_files = find_all_pngs()
    
    if not png_files:
        print("No PNG files found!")
        return None
    
    print("=" * 120)
    print("🖼️  COMPREHENSIVE ANALYSIS OF ALL PNG FILES")
    print("=" * 120)
    print(f"Found {len(png_files)} PNG files to analyze...")
    
    # Create labels from file paths
    labels = []
    for png_path in png_files:
        # Create readable label from path
        label = png_path.replace('./', '').replace('\\', '/') 
        labels.append(label)
    
    results = []
    
    # Analyze each image
    for i, (img_path, label) in enumerate(zip(png_files, labels), 1):
        print(f"\n📊 [{i:2d}/{len(png_files)}] Analyzing: {label}")
        
        if not os.path.exists(img_path):
            print(f"  ❌ File not found: {img_path}")
            continue
        
        # Check if it's a valid image file
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ Could not read image: {img_path}")
                continue
                
            # Get image dimensions for context
            height, width = img.shape[:2]
            print(f"  📏 Dimensions: {width}x{height} pixels")
            
        except Exception as e:
            print(f"  ❌ Error reading image: {e}")
            continue
        
        # Get literature metrics
        lit_metrics = analyze_image_quality_literature(img_path)
        
        # Get raw BRISQUE score only
        raw_brisque = calculate_brisque_opencv(img)
        
        # Get PIQUE score  
        pique_score = calculate_pique(img)
        
        # Combine all results
        result = {
            'label': label,
            'path': img_path,
            'width': width,
            'height': height,
            'megapixels': round((width * height) / 1000000, 2),
            'raw_brisque': raw_brisque,
            'pique_score': pique_score,
            'brisque_interpretation': get_quality_interpretation(raw_brisque)
        }
        
        # Add literature metrics if available
        if lit_metrics:
            result.update(lit_metrics)
        
        results.append(result)
        
        # Print brief summary for this image
        if raw_brisque is not None:
            print(f"  🎯 BRISQUE: {raw_brisque:.2f} ({get_quality_interpretation(raw_brisque)})")
        if pique_score is not None:
            print(f"  🔍 PIQUE: {pique_score:.2f} (lower = better)")
        if lit_metrics and lit_metrics.get('laplacian_variance'):
            print(f"  🔪 Sharpness: {lit_metrics['laplacian_variance']:.4f}")
        if lit_metrics and lit_metrics.get('rms_contrast'):
            print(f"  🌈 Contrast: {lit_metrics['rms_contrast']:.4f}")
    
    if not results:
        print("No images could be analyzed successfully")
        return None
    
    print(f"\n✅ Successfully analyzed {len(results)} images")
    
    # Display comprehensive results
    display_comprehensive_results(results)
    
    return results

def display_comprehensive_results(results):
    """Display comprehensive analysis results"""
    
    print("\n" + "=" * 120)
    print("📋 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 120)
    
    # Basic info table
    print(f"\n{'#':<3} {'Image':<45} {'Dimensions':<12} {'MP':<5} {'BRISQUE':<10} {'PIQUE':<8} {'Quality':<20}")
    print("-" * 120)
    
    for i, result in enumerate(results, 1):
        dims = f"{result['width']}x{result['height']}"
        mp = f"{result['megapixels']:.1f}"
        brisque = f"{result['raw_brisque']:.2f}" if result['raw_brisque'] else "N/A"
        pique = f"{result['pique_score']:.2f}" if result['pique_score'] else "N/A"
        quality = result['brisque_interpretation'] if result['brisque_interpretation'] else "N/A"
        
        print(f"{i:<3} {result['label'][:44]:<45} {dims:<12} {mp:<5} {brisque:<10} {pique:<8} {quality:<20}")
    
    # Detailed metrics table
    print("\n" + "=" * 120)
    print("📊 DETAILED QUALITY METRICS")
    print("=" * 120)
    
    print(f"{'Image':<40} {'BRISQUE':<10} {'PIQUE':<8} {'Laplacian':<12} {'Brenner':<12} {'Contrast':<10} {'Entropy':<8}")
    print("-" * 120)
    
    for result in results:
        img_name = result['label'].split('/')[-1][:39]  # Just filename, truncated
        brisque = f"{result['raw_brisque']:.2f}" if result['raw_brisque'] else "N/A"
        pique = f"{result['pique_score']:.2f}" if result['pique_score'] else "N/A"
        lap = f"{result.get('laplacian_variance', 0):.4f}" if result.get('laplacian_variance') else "N/A"
        bre = f"{result.get('brenner_sharpness', 0):.0f}" if result.get('brenner_sharpness') else "N/A"
        con = f"{result.get('rms_contrast', 0):.4f}" if result.get('rms_contrast') else "N/A"
        ent = f"{result.get('shannon_entropy', 0):.2f}" if result.get('shannon_entropy') else "N/A"
        
        print(f"{img_name:<40} {brisque:<10} {pique:<8} {lap:<12} {bre:<12} {con:<10} {ent:<8}")
    
    # Rankings
    print("\n" + "=" * 120)
    print("🏆 QUALITY RANKINGS")
    print("=" * 120)
    
    # Filter for valid results
    valid_brisque = [r for r in results if r.get('raw_brisque') is not None]
    valid_pique = [r for r in results if r.get('pique_score') is not None]
    valid_sharpness = [r for r in results if r.get('laplacian_variance') is not None]
    
    if valid_brisque:
        print("\n🎯 Top 5 by BRISQUE (Best Quality - Lower Scores):")
        brisque_ranked = sorted(valid_brisque, key=lambda x: x['raw_brisque'])[:5]
        for i, result in enumerate(brisque_ranked, 1):
            score = result['raw_brisque']
            quality = result['brisque_interpretation']
            name = result['label'].split('/')[-1]
            print(f"  {i}. {name:<40} Score: {score:6.2f} ({quality})")
    
    if valid_pique:
        print("\n🔍 Top 5 by PIQUE (Best Quality - Lower Scores):")
        pique_ranked = sorted(valid_pique, key=lambda x: x['pique_score'])[:5]
        for i, result in enumerate(pique_ranked, 1):
            score = result['pique_score']
            name = result['label'].split('/')[-1]
            print(f"  {i}. {name:<40} Score: {score:6.2f} (lower = better)")
    
    if valid_sharpness:
        print("\n🔪 Top 5 by Sharpness (Laplacian Variance):")
        sharp_ranked = sorted(valid_sharpness, key=lambda x: x['laplacian_variance'], reverse=True)[:5]
        for i, result in enumerate(sharp_ranked, 1):
            score = result['laplacian_variance']
            name = result['label'].split('/')[-1]
            print(f"  {i}. {name:<40} Score: {score:.4f}")
    
    # Quality distribution
    if valid_brisque:
        print("\n📈 Quality Distribution:")
        quality_counts = {}
        for result in valid_brisque:
            quality = result['brisque_interpretation']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        for quality, count in sorted(quality_counts.items()):
            print(f"  {quality}: {count} images")
    
    # Summary statistics
    if valid_brisque:
        brisque_scores = [r['raw_brisque'] for r in valid_brisque]
        print(f"\n📊 BRISQUE Statistics:")
        print(f"  Best (lowest): {min(brisque_scores):.2f}")
        print(f"  Worst (highest): {max(brisque_scores):.2f}")
        print(f"  Average: {np.mean(brisque_scores):.2f}")
        print(f"  Median: {np.median(brisque_scores):.2f}")
    
    if valid_pique:
        pique_scores = [r['pique_score'] for r in valid_pique]
        print(f"\n📊 PIQUE Statistics:")
        print(f"  Best (lowest): {min(pique_scores):.2f}")
        print(f"  Worst (highest): {max(pique_scores):.2f}")
        print(f"  Average: {np.mean(pique_scores):.2f}")
        print(f"  Median: {np.median(pique_scores):.2f}")
    
    if valid_sharpness:
        sharp_scores = [r['laplacian_variance'] for r in valid_sharpness]
        print(f"\n📊 Sharpness Statistics:")
        print(f"  Sharpest (highest): {max(sharp_scores):.4f}")
        print(f"  Least sharp (lowest): {min(sharp_scores):.4f}")
        print(f"  Average: {np.mean(sharp_scores):.4f}")
    
    print("\n" + "=" * 120)
    print("📚 LITERATURE METRICS USED:")
    print("• BRISQUE: Mittal et al. (2012) - No-reference perceptual quality")
    print("• PIQUE: Venkatanath et al. (2015) - Perception based quality evaluation")
    print("• Laplacian Variance: Tenenbaum et al. (1984) - Edge sharpness") 
    print("• Brenner Sharpness: Brenner et al. (1976) - Focus quality")
    print("• RMS Contrast: Standard contrast measure")
    print("• Shannon Entropy: Information theory content measure")
    print("=" * 120)
    print("💡 ANALYSIS COMPLETE")
    print("=" * 120)

def save_results_to_csv(results):
    """Save results to CSV file for further analysis"""
    try:
        import pandas as pd
        
        # Flatten results for CSV
        csv_data = []
        for result in results:
            csv_row = {
                'filename': result['label'].split('/')[-1],
                'full_path': result['label'],
                'width': result['width'],
                'height': result['height'],
                'megapixels': result['megapixels'],
                'raw_brisque': result.get('raw_brisque'),
                'pique_score': result.get('pique_score'),
                'quality_interpretation': result.get('brisque_interpretation'),
                'laplacian_variance': result.get('laplacian_variance'),
                'brenner_sharpness': result.get('brenner_sharpness'),
                'rms_contrast': result.get('rms_contrast'),
                'shannon_entropy': result.get('shannon_entropy'),
                'gradient_magnitude': result.get('gradient_magnitude'),
                'energy_of_gradient': result.get('energy_of_gradient')
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv('png_analysis_results.csv', index=False)
        print(f"\n💾 Results saved to 'png_analysis_results.csv'")
        
    except ImportError:
        print("\n💾 CSV export requires pandas. Install with: pip install pandas")

if __name__ == "__main__":
    # Find and display PNG files first
    png_files = find_all_pngs()
    print(f"Found {len(png_files)} PNG files:")
    for i, png in enumerate(png_files, 1):
        print(f"  {i:2d}. {png}")
    
    # Ask for confirmation if many files
    if len(png_files) > 10:
        print(f"\n⚠️  This will analyze {len(png_files)} images. This may take a while.")
        response = input("Continue? (y/n): ").lower().strip()
        if response != 'y':
            print("Analysis cancelled.")
            exit()
    
    # Run the analysis
    results = analyze_all_png_files()
    
    if results:
        # Optionally save to CSV
        try:
            save_results_to_csv(results)
        except:
            pass  # CSV export is optional 