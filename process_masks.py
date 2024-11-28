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

#Edit Manga109 Dataset to match the masks dataset

manga_109_dir = 'Manga109_Dataset/images'

new_dir_name = 'Manga109_Truncated'

#create a new directory for the truncated dataset that is a copy of the original dataset
if not os.path.exists(new_dir_name):
    os.makedirs(new_dir_name)

names = ['000.jpg', '001.jpg', '002.jpg', '003.jpg', '004.jpg', '005.jpg', '006.jpg', '007.jpg', '008.jpg', '009.jpg']

#iterate through each manga directory in the Manga109_Dataset directory
for manga_name in os.listdir(manga_109_dir):
    #create a new directory for the manga in the Manga109_Truncated directory
    new_manga_dir = os.path.join(new_dir_name, manga_name)
    if not os.path.exists(new_manga_dir):
        os.makedirs(new_manga_dir)
    
    #check if the file is a directory
    if not os.path.isdir(os.path.join(manga_109_dir, manga_name)):
        print("Skipping", manga_name, "as it is not a directory.")
        continue

    print("Processing images for", manga_name)

    #iterate through each image in the manga directory
    for image_file in os.listdir(os.path.join(manga_109_dir, manga_name)):
        print("Processing image", image_file)
        # if the image is not in the list of names, skip it
        if image_file not in names:
            print("Skipping", image_file, "as it is not in the list of names.")
            continue

        #copy the image to the new directory
        image_path = os.path.join(manga_109_dir, manga_name, image_file)
        new_image_path = os.path.join(new_manga_dir, image_file)
        img = cv2.imread(image_path)
        cv2.imwrite(new_image_path, img)


exit()

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
