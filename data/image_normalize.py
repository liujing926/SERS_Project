#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from PIL import Image

# =======================================================
# 1. Dataset paths (Using relative paths for GitHub)
# =======================================================

excel_path = './sample_data/labels/data_EF.xlsx'
image_dir = './sample_data/ranking_data/images'
output_dir = './sample_data/normalized_images'


pixel_size_column = 'nm/pixel'  
filename_column = 'Filename'        

# Target scale (nm/pixel)
target_nm_per_px = 4.1666667  
# =======================================================


os.makedirs(output_dir, exist_ok=True)

# Read Excel file
df = pd.read_excel(excel_path)
print(f"✅ Successfully read Excel file, total {len(df)} records.")

# Iterate through each image
for idx, row in df.iterrows():
    img_name = row[filename_column]
    current_nm_per_px = row[pixel_size_column]
    
    # Calculate scale factor
    if current_nm_per_px <= 0:
        print(f"⚠️ Skipped {img_name}: Invalid nm/pixel ({current_nm_per_px})")
        continue
        
    scale_factor = current_nm_per_px / target_nm_per_px
    
    # 🔥 Smart extension check: support jpg, jpeg, png, tif
    base_path = os.path.join(image_dir, str(img_name))
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        if os.path.exists(base_path + ext):
            img_path = base_path + ext
            break
            
    if not img_path:
        print(f"❌ Error: Image {img_name} not found with any standard extension in {image_dir}")
        continue
    
    # Open and resize image
    img = Image.open(img_path)
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    # Use high-quality LANCZOS resampling
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    

    if img_resized.mode in ("RGBA", "P"):
        img_resized = img_resized.convert("RGB")
        
    new_img_name = f"{img_name}.jpg"
    new_img_path = os.path.join(output_dir, new_img_name)
    img_resized.save(new_img_path, format='JPEG', quality=100) # Use high quality
    
    print(f"✅ Processed: {img_name} ({current_nm_per_px:.3f} nm/px) -> {new_img_name} ({target_nm_per_px} nm/px), Scale factor: {scale_factor:.2f}")

print("\n🎉 All images processed successfully!")