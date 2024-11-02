import math
from PIL import ImageFont

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

def typeset(h, w, font=None, text=""):
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
    text_width, text_height = text_font.getsize(text)
    
    # Increase font size until text dimensions exceed rectangle dimensions
    while text_width <= w and text_height <= h:
        font_size += 1
        text_font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = text_font.getsize(text)
    
    # Return the font size that fits (last size was too large, so we go back by 1)
    return font_size - 1


