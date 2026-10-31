#!/usr/bin/env python
# coding: utf-8

"""
Image Processing Utility Module
Features: Unify image formats, check images, batch processing, etc.
"""

import os
import cv2
import glob
import numpy as np
from PIL import Image

# Optional: try to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def convert_to_jpg(input_dir, output_dir=None, quality=95, remove_original=False):
    """
    Convert all images in a folder to JPG format.
    
    Args:
        input_dir: Input folder path
        output_dir: Output folder path (if None, convert in-place)
        quality: JPEG compression quality (1-100)
        remove_original: Whether to delete original files
    
    Returns:
        converted_count: Number of converted images
        failed_count: Number of failed images
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats (excluding jpg)
    image_extensions = {
        '.png': 'PNG',
        '.bmp': 'BMP',
        '.tif': 'TIFF', 
        '.tiff': 'TIFF',
        '.jpeg': 'JPEG',
        '.JPEG': 'JPEG',
        '.PNG': 'PNG',
        '.BMP': 'BMP'
    }
    
    converted_count = 0
    failed_count = 0
    
    print(f"\nStarting image conversion...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Iterate over all supported formats
    for ext, format_name in image_extensions.items():
        for img_path in glob.glob(os.path.join(input_dir, f"*{ext}")):
            try:
                img = None
                
                # Method 1: Read with cv2
                img = cv2.imread(img_path)
                
                # Method 2: If cv2 fails, use PIL
                if img is None:
                    pil_img = Image.open(img_path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if img is None:
                    print(f"❌ Cannot read: {os.path.basename(img_path)}")
                    failed_count += 1
                    continue
                
                # Check and handle channels
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Generate output path
                filename = os.path.basename(img_path)
                name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{name}.jpg")
                
                # Save as JPG
                success = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                if success:
                    converted_count += 1
                    if remove_original and output_dir == input_dir:
                        os.remove(img_path)
                        print(f"✅ Converted and removed original: {filename} -> {name}.jpg")
                    else:
                        print(f"✅ Converted: {filename} -> {name}.jpg")
                else:
                    print(f"❌ Save failed: {filename}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Conversion failed {os.path.basename(img_path)}: {e}")
                failed_count += 1
    
    # Statistics
    print(f"\nConversion complete!")
    print(f"✅ Success: {converted_count}")
    print(f"❌ Failed: {failed_count}")
    
    # Check converted jpg files
    jpg_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    print(f"📊 Total JPG images in directory: {len(jpg_files)}")
    
    return converted_count, failed_count


def check_image_formats(image_dir, show_details=True):
    """
    Check the distribution of image formats in a folder.
    """
    all_extensions = []
    
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            all_extensions.append(ext)
    
    unique_exts = set(all_extensions)
    format_stats = {}
    
    print(f"\n📁 Folder: {image_dir}")
    print(f"Total images: {len(all_extensions)}")
    print("\nFormat distribution:")
    
    for ext in sorted(unique_exts):
        count = all_extensions.count(ext)
        percentage = count / len(all_extensions) * 100
        format_stats[ext] = count
        print(f"  {ext}: {count} ({percentage:.1f}%)")
    
    if show_details and len(all_extensions) > 0:
        print("\nSample files:")
        for i, filename in enumerate(sorted(os.listdir(image_dir))[:10]):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                print(f"  {filename}")
    
    return format_stats


def batch_resize(image_dir, target_size=(512, 512), output_dir=None):
    """Batch resize images."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = image_dir
    
    resized_count = 0
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in glob.glob(os.path.join(image_dir, ext)):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img_resized)
                resized_count += 1
                
            except Exception as e:
                print(f"Resize failed {img_path}: {e}")
    
    print(f"Resize complete: {resized_count} images -> {target_size}")
    return resized_count


def verify_images(image_dir, required_size=(512, 512)):
    """Verify if images meet requirements (size, channels, etc.)."""
    valid = 0
    invalid = 0
    issues = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in glob.glob(os.path.join(image_dir, ext)):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    invalid += 1
                    issues.append(f"{os.path.basename(img_path)}: Cannot read")
                    continue
                
                h, w = img.shape[:2]
                if (h, w) != required_size:
                    issues.append(f"{os.path.basename(img_path)}: Size should be {required_size}, actual is ({h}, {w})")
                    invalid += 1
                    continue
                
                if len(img.shape) != 3 or img.shape[2] != 3:
                    issues.append(f"{os.path.basename(img_path)}: Channels should be 3, actual is {img.shape[2] if len(img.shape)==3 else 1}")
                    invalid += 1
                    continue
                
                valid += 1
                
            except Exception as e:
                invalid += 1
                issues.append(f"{os.path.basename(img_path)}: {str(e)}")
    
    print(f"\nImage verification results:")
    print(f"✅ Valid: {valid}")
    print(f"❌ Invalid: {invalid}")
    if issues and len(issues) <= 10:
        print("Issues list:")
        for issue in issues:
            print(f"  {issue}")
    elif issues:
        print(f"There are {len(issues)} issues in total.")
    
    return valid, invalid, issues


if __name__ == '__main__':
    # ========== Set paths (Updated to sample_data for GitHub) ==========
    # You can change these to your actual folders when using locally
    input_dir = './sample_data/contrastive_data/images'
    output_dir = './sample_data/contrastive_data/processed_jpg'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Check image formats ==========
    print("1. Checking image formats...")
    check_image_formats(input_dir, show_details=True)
    
    # ========== 2. Convert image formats ==========
    print("\n2. Converting image formats to JPG...")
    convert_to_jpg(input_dir, output_dir, quality=95, remove_original=False)
    
    print("\n✅ All operations completed!")