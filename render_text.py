import math
from PIL import ImageFont
from PIL import Image, ImageDraw
import textwrap

def bubble_wrap_text(text, max_chars_per_line):
    """
    Formats the text into a bubble shape with line wrapping.
    
    Parameters:
    - text (str): The input text to wrap.
    - max_chars_per_line (int): The maximum characters allowed per line.
    
    Returns:
    - str: The formatted bubble-shaped text.
    """
    words = text.split()  # Split text into words
    lines = []  # Stores the resulting lines
    current_line = []  # Stores the words of the current line
    line_max = max_chars_per_line

    # Function to calculate max chars for the current line based on bubble shape
    def get_line_limit(line_index, total_lines):
        if total_lines <= 1:
            return max_chars_per_line  # Only one line, no bubble shape
        mid = total_lines // 2
        if line_index <= mid:  # Lines before and including the middle
            return max_chars_per_line - (mid - line_index) * 2
        else:  # Lines after the middle
            return max_chars_per_line - (line_index - mid) * 2

    # First pass: Create lines without the bubble effect
    for word in words:
        if current_line and len(" ".join(current_line + [word])) > line_max:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))

    # Second pass: Apply the bubble shape
    total_lines = len(lines)
    bubble_lines = []
    for i, line in enumerate(lines):
        line_limit = get_line_limit(i, total_lines)
        # Break long lines if needed
        line_words = line.split()
        current_bubble_line = []
        while line_words:
            word = line_words.pop(0)
            if current_bubble_line and len(" ".join(current_bubble_line + [word])) > line_limit:
                bubble_lines.append(" ".join(current_bubble_line))
                current_bubble_line = [word]
            else:
                current_bubble_line.append(word)
        if current_bubble_line:
            bubble_lines.append(" ".join(current_bubble_line))

    # Third pass. Any line that contains only one word should added to the shorter line above it or below it
    # except for the first line of the bubble
    i = 1
    while i < len(bubble_lines)-1:
        if len(bubble_lines) <= 2: #Don't do this if there are only two lines since that will result in 1 line
            break
        if len(bubble_lines[i].split()) == 1: #This line has only one word
            #The line above it is shorter than the line below it
            if i == len(bubble_lines) - 1:
                bubble_lines[i-1] += " " + bubble_lines[i]
            elif len(bubble_lines[i-1].split()) < len(bubble_lines[i+1].split()):
                bubble_lines[i-1] += " " + bubble_lines[i]
            else:
                bubble_lines[i+1] = bubble_lines[i] + " " + bubble_lines[i+1]
            #Remove the line
            bubble_lines.pop(i)
        i += 1

    return "\n".join(bubble_lines)

def calculate_font_size(h, w, font_in=None, text="", max_chars_per_line=20):
    """
    Determine the optimal font size for the given text and rectangle dimensions. Also wraps the text to fit the rectangle if needed.

    Parameters:
    h (int): Height of the rectangle.
    w (int): Width of the rectangle.
    font (str): Path to the .ttf font file (optional). Uses default font if not provided.
    text (str): The text to fit inside the rectangle.
    max_chars_per_line (int): Maximum number of chars per line (optional).

    Returns:
    int: Optimal font size. 
    str: Wrapped text.
    """
    #Process the text to the maximum chars per line. We want to prefer a bubble shape where the first and last lines are shorter than the middle lines
    new_text = bubble_wrap_text(text, max_chars_per_line)

    # Initialize the font size and text size
    best_size = 1
    best_text = new_text

    #Slowly increase the font size until the text no longer fits within the rectangle
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while True:
        font = ImageFont.truetype(font_in, best_size)
        bbox = draw.textbbox((0, 0), new_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_width > w or text_height > h:
            break
        best_size += 1

    # We can afford to increase the size of long text to make it fit better since they usually appear in whitespaces
    # if new_text.count('\n') >= 1 or new_text.count(' ') >= 2:
    #     best_size = int(best_size * 1.1)
    
    return best_size, best_text

#input, x and y coordinates of the center of the rectangle + the dimensions of the rectangle + the text to be rendered
def typeset(x, y, w, h, text, font=None, max_chars_per_line=20):
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
    font_size, new_text = calculate_font_size(h, w, font, text, max_chars_per_line)
    
    # Load the font and render the text
    text_font = ImageFont.truetype(font, font_size)
    
    
    # Return the rendered text
    return text_font, x, y, new_text

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

def render_text_on_image(image_path, text_to_render, font=None, max_chars_per_line=20):
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

        #If the height is significantly larger than the width, adjust width to make it more square
        ratio = 2
        new_width = w
        if h > w * ratio:
            #print("original width:", w)
            new_width = int(h // ratio)
            x = x - (new_width - w) // 2 #push x back to account for the new width
            if x < 0:
                x = 0
        #print("new width:", w)
        
        # Use the typeset function to calculate the font size and position
        text_font, text_x, text_y, new_text = typeset(x, y, new_width, h, text, font, max_chars_per_line)

        # Render the text on the image using the calculated position and font and black if no color is provided
        color = text_data.get("color", "black")
        bbox = draw.textbbox((text_x, text_y), new_text, font=text_font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center_x = x + (new_width - text_width) / 2
        center_y = y + (h - text_height) / 2
        draw.multiline_text((center_x,center_y), new_text, font=text_font, fill=color, align="center")
        # Draw the bounding box (optional, for visualization)
        # draw.rectangle([x, y, x + w, y + h], outline="blue", width=2)
        # draw.rectangle([x, y, x + new_width, y + h], outline="red", width=2)
    
    return image

def typeset_all_images(pages, font=None, max_chars_per_line=20):
    """
    Render text on all images in the given list of pages.

    Parameters:
    pages (List[Dictionary]): List of pages with images and text to render.

    Returns:
    List[PIL.Image]: List of images with rendered text.
    """
    rendered_images = []
    
    for page in pages:
        print("Typesetting image:", page["image_path"])
        #use render_text_on_image function to render text on the image
        rendered_image = render_text_on_image(page["image_path"], page["text_to_render"], font, max_chars_per_line)
        rendered_images.append((rendered_image, page["image_path"]))
    
    return rendered_images

#Open to_typeset.json and read the data
import json
with open("data/output/openmantra_merged.json") as f:
    data = json.load(f)

pages = data["pages"]

#use typeset_all_images function to render text on all images
#https://www.dafont.com/manga-temple.font
rendered_images = typeset_all_images(pages, "fonts/manga_temple/mangat.ttf", 15)

#save the rendered images to the output folder with the same name as the input image
import os

out_folder = "rendered_images"

# Save the rendered images to the output folder with the same name as the input image
for i, image in enumerate(rendered_images):
    output_path = os.path.join(out_folder, image[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image[0].save(output_path)
    print(f"Image {i+1} rendered and saved as {output_path}")

