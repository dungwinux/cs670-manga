import cv2
import os
from manga_ocr import MangaOcr
import json
import argparse
from tqdm import tqdm

def detect(path, coordinates, mocr, save=True): 
    '''
    Given an image stored in a path, crop the image based on 
    the given coordinates and optionally save cropped images.
    ''' 
    # extend the path to the image
    path = "../" + path
    image = cv2.imread(path)
    
    texts = []
    for i, coordinate in enumerate(coordinates): 
        x, y, w, h = coordinate[0], coordinate[1], coordinate[2], coordinate[3] 
        cropped_image = image[y:y+h, x:x+w]
        if save: 
            # Construct the output path
            base_dir, ext = os.path.splitext(path)
            output_dir = base_dir.replace("input", "output")
            output_path = f"{output_dir}_{i}{ext}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cropped_image)
        else:
            output_path = None
        
        # Perform OCR
        texts.append(mocr(output_path) if output_path else mocr(cropped_image))

    return texts

def main(): 
    parser = argparse.ArgumentParser(description='Preprocess the manga images')
    parser.add_argument('--location_file', type=str, help='File that contains the coordinates of the bounding boxes')
    args = parser.parse_args()

    # Initialize MangaOCR
    mocr = MangaOcr()

    # Load coordinates file
    with open(args.location_file) as f:
        d = json.load(f)

    # Process each file
    for f in tqdm(d): 
        path = f['path']
        coordinates = f['coordinates']
        f['text'] = detect(path, coordinates, mocr)

    # Save the updated JSON file
    output_dir = os.path.dirname(args.location_file).replace("input", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.location_file))

    with open(output_path, 'w') as f:
        json.dump(d, f, indent=4)

if __name__ == "__main__": 
    main()
