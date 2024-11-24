import math
from PIL import ImageFont
from PIL import Image, ImageDraw

#Input, width and height of a speech bubble (ellipse/recangle)
#returns the width, height and area of the largest rectangle that can be inscribed inside the ellipse
def calculate_max_space_available(width, height):
    ellipse_width_axis = width / 2
    ellipse_height_axis = height / 2

    #calculate the largest rectangle that can be inscribed inside the ellipse
    available_width = ellipse_width_axis * (2 - math.sqrt(2))
    available_height = ellipse_height_axis * (2 - math.sqrt(2))
    available_area = available_width * available_height

    return (available_width, available_height, available_area)

#Input, width and height of a rectangle
def calculate_optimal_font_size(width, height):
    #calculate the largest rectangle that can be inscribed inside the ellipse
    available_width = width * (2 - math.sqrt(2))
    available_height = height * (2 - math.sqrt(2))
    available_area = available_width * available_height

    #calculate the optimal font size
    font_size = math.sqrt(available_area / 20)
    return font_size

def calculate_font_size(h, w, font=None, text=""):
    """
    Determine the optimal font size for the given text and rectangle dimensions.

    Parameters:
    h (int): Height of the rectangle.
    w (int): Width of the rectangle.
    font (str): Path to the .ttf font file (optional). Uses default font if not provided.
    text (str): The text to fit inside the rectangle.

    Returns:
    int: Optimal font size.
    """
    if not text:
        return 0  # No text to fit, font size of 0
    
    # Load font, default to a basic one if not specified
    font_path = font if font else "arial.ttf"
    
    # Start with a small font size and increase until it exceeds the rectangle
    font_size = 1
    text_font = ImageFont.truetype(font_path, font_size)
    text_bbox = text_font.getbbox(text)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    # Increase font size until text dimensions exceed rectangle dimensions
    while text_width <= w and text_height <= h:
        font_size += 1
        text_font = ImageFont.truetype(font_path, font_size)
        text_bbox = text_font.getbbox(text)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    # Return the font size that fits (last size was too large, so we go back by 1)
    return font_size - 1

#input, x and y coordinates of the center of the rectangle + the dimensions of the rectangle + the text to be rendered
def typeset(x, y, w, h, text, font=None):
    """
    Render text inside a rectangle with the given dimensions.

    Parameters:
    x (int): X-coordinate of the rectangle center.
    y (int): Y-coordinate of the rectangle center.
    w (int): Width of the rectangle.
    h (int): Height of the rectangle.
    text (str): The text to fit inside the rectangle.
    font (str): Path to the .ttf font file (optional). Uses default font if not provided.

    Returns:
    str: Rendered text.
    """
    # Calculate the optimal font size for the given text and rectangle dimensions
    font_size = calculate_font_size(h, w, font, text)
    
    # Load the font and render the text
    text_font = ImageFont.truetype(font, font_size)
    text_bbox = text_font.getbbox(text)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    # Calculate the position to center the text in the rectangle
    text_x = x - text_width / 2
    text_y = y - text_height / 2
    
    # Return the rendered text
    return text_font, text_x, text_y, text_width, text_height

# {
#     "pages": [
#         {
#             "image_path": "to_typeset/test_image.png",
#             "text_to_render": [
#                 {
#                     "text": "This is a test image",
#                     "color": "black",
#                     "position": {
#                         "x": 220,
#                         "y": 350,
#                         "w": 400,
#                         "h": 200
#                     }
#                 },
#                 {
#                     "text": "Second Test Text",
#                     "color": "black",
#                     "position": {
#                         "x": 710,
#                         "y": 620,
#                         "w": 300,
#                         "h": 400
#                     }
#                 }
#             ]
#         }
#     ]
# }

def render_text_on_image(image_path, text_to_render, font=None):
    """
    Render text on an image at the specified position.

    Parameters:
    image_path (str): Path to the image file.
    text_to_render (List[Dictionary]): List of text to render with position and dimensions.
    font (str): Path to the .ttf font file (optional). Uses default font if not provided.

    Returns:
    PIL.Image: Image with rendered text.
    """
    # Open the image and create a drawing object
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Render each text element on the image
    for text_data in text_to_render:
        text = text_data["text"]
        x, y, w, h = text_data["position"].values()
        
        # Use the typeset function to calculate the font size and position
        text_font, text_x, text_y, text_width, text_height = typeset(x, y, w, h, text, font)

        # Render the text on the image using the calculated position and font and black if no color is provided
        color = text_data.get("color", "black")
        draw.text((text_x, text_y), text, font=text_font, fill=color)
    
    return image

def typeset_all_images(pages, font=None):
    """
    Render text on all images in the given list of pages.

    Parameters:
    pages (List[Dictionary]): List of pages with images and text to render.

    Returns:
    List[PIL.Image]: List of images with rendered text.
    """
    rendered_images = []
    
    for page in pages:
        image = Image.open(page["image_path"])
        image_name = page["image_path"].split("/")[-1]
        draw = ImageDraw.Draw(image)

        #use render_text_on_image function to render text on the image
        rendered_image = render_text_on_image(page["image_path"], page["text_to_render"], font)
        rendered_images.append((rendered_image, image_name))
    
    return rendered_images

#Open to_typeset.json and read the data
import json
with open("to_typeset.json") as f:
    data = json.load(f)

#use typeset_all_images function to render text on all images
#https://www.dafont.com/manga-temple.font
rendered_images = typeset_all_images(data["pages"], "fonts/manga_temple/mangat.ttf")

#save the rendered images to the output folder with the same name as the input image
import os

# Save the rendered images to the output folder with the same name as the input image
for i, image in enumerate(rendered_images):
    output_path = os.path.join("outputs", image[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image[0].save(output_path)
    print(f"Image {i+1} rendered and saved as {output_path}")

