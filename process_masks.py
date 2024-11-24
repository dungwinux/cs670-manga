'''
Manga109_Masks format:
    Parent directory: Manga109_Masks
        Subdirectories: <MangaName>
            Files: <PageNumber>.png in the format 001.png, 002.png, ... 009.png
'''

#Each Mask is an image where white pixels represent non-masked regions, and non-white pixels represent masked regions.
#We want to convert the masks to images where white pixels represent masked regions, and black pixels represent non-masked regions.

import os
import cv2
import numpy as np

# Path to the Manga109_Masks directory
mask_dir = 'Manga109_Masks'

# Path to the Manga109_Recropped directory
new_dir_name = 'Manga109_Recropped'

# Create the Manga109_Recropped directory if it does not exist
if not os.path.exists(new_dir_name):
    os.makedirs(new_dir_name)

# Iterate through each manga directory in the Manga109_Masks directory

for manga_name in os.listdir(mask_dir):
    print("Full path to manga directory:", os.path.join(mask_dir, manga_name))


for manga_name in os.listdir(mask_dir):
    # Create a new directory for the manga in the Manga109_Recropped directory
    new_manga_dir = os.path.join(new_dir_name, manga_name)
    if not os.path.exists(new_manga_dir):
        os.makedirs(new_manga_dir)
    
    #check if the file is a directory
    if not os.path.isdir(os.path.join(mask_dir, manga_name)):
        print("Skipping", manga_name, "as it is not a directory.")
        continue

    print("Processing masks for", manga_name)

    # Iterate through each mask in the manga directory
    for mask_file in os.listdir(os.path.join(mask_dir, manga_name)):
        print("Processing mask", mask_file)
        # Read the mask image
        mask_path = os.path.join(mask_dir, manga_name, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #If a pixel is not white, set it to black
        mask[mask != 255] = 0

        # Invert the mask
        mask = cv2.bitwise_not(mask)

        # Save the inverted mask image
        new_mask_path = os.path.join(new_manga_dir, mask_file)
        cv2.imwrite(new_mask_path, mask)

print("Masks processed successfully.")
