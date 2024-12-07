
import os
import cv2

input_images = "Manga109_Truncated"
masks = "Manga109_Recropped"
output = "Manga109_Overlayed2"
lines = "Lines2"

# Create the Manga109_Overlayed directory if it does not exist
if not os.path.exists(output):
    os.makedirs(output)

images = []
for root, dirs, files in os.walk(input_images):
    # Check if the corresponding directory exists in the output, create it if not
    for dir in dirs:
        output_dir = os.path.join(output, os.path.relpath(os.path.join(root, dir), input_images))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(root, file))

for image in images:
    print("Processing image", image)
    img = cv2.imread(image)

    # Try to find the mask with different possible extensions
    mask_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        possible_mask_path = image.replace(input_images, masks).rsplit('.', 1)[0] + ext
        if os.path.exists(possible_mask_path):
            mask_path = possible_mask_path
            break
    if mask_path is None:
        print(f"No mask found for image {image}")
        continue
    #print("Mask path:", mask_path)

    # Get the mask, resized to the same size as the image
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    cv2.imwrite(image.replace(input_images, output), overlay)
    # Create a whiteout version of the image by iterating through the pixels of mask. If the pixel is white, set the corresponding pixel in the image to white
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i][j][0] == 255:
    #             img[i][j] = [255, 255, 255]
    # cv2.imwrite(image.replace(input_images, output).replace('.png', '_whiteout.png').replace('.jpg', '_whiteout.jpg'), img)
    
    lines_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        possible_lines_path = image.replace(input_images, lines).rsplit('.', 1)[0] + ext
        if os.path.exists(possible_lines_path):
            lines_path = possible_lines_path
            break
    if lines_path is None:
        print(f"No lines found for image {image}")
        continue
    #print("Lines path:", lines_path)
    lines_img = cv2.imread(lines_path)
    lines_img = cv2.resize(lines_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_lines = cv2.addWeighted(lines_img, 0.5, mask, 0.5, 0)
    cv2.imwrite(image.replace(input_images, output).replace('.png', '_lines_overlay.png').replace('.jpg', '_lines_overlay.jpg'), overlay_lines)
    

    
    #print("Overlayed image saved to", image.replace(input_images, output))
    #exit()

print("Done!")