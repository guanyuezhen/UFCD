import os
import glob
import re

train_dir = "./train"
masks_dir = os.path.join(train_dir, "images")
output_file = os.path.join(train_dir, "image_list.txt")

mask_files = glob.glob(os.path.join(masks_dir, '*post_disaster.png*'))

with open(output_file, 'w') as f:
    for mask_file in mask_files:
        f.write(f"{os.path.basename(mask_file)}\n")

print(f"completely save in {output_file}")


train_dir = "./test"
masks_dir = os.path.join(train_dir, "images")
output_file = os.path.join(train_dir, "image_list.txt")

mask_files = glob.glob(os.path.join(masks_dir, '*post_disaster.png*'))

with open(output_file, 'w') as f:
    for mask_file in mask_files:
        f.write(f"{os.path.basename(mask_file)}\n")

print(f"completely save in {output_file}")

