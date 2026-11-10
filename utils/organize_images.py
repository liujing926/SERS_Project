#!/usr/bin/env python
# coding: utf-8

# train.py header
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.getcwd()) 
if project_root not in sys.path:
    sys.path.append(project_root)

print(sys.path)



from utils import image_utils  

# Set a switch
RUN_PREPROCESS = False  
if RUN_PREPROCESS:
    print("Starting image format conversion...")
    
    # Call your function
    converted, failed = image_utils.convert_to_jpg(
        input_dir='./sample_data/contrastive_data/images',        
        output_dir='./sample_data/contrastive_data/processed_jpg', 
        quality=95,
        remove_original=False
    )
    
    print(f"Conversion complete, Success: {converted}, Failed: {failed}")
    
    # Optional: Resize after conversion
    resized = image_utils.batch_resize(
        image_dir='./sample_data/contrastive_data/processed_jpg',
        target_size=(512, 512)
    )
else:
    print("Skipping preprocessing, using already processed images.")