import gradio as gr
from PIL import Image
from PIL import ImageOps
import numpy as np

def create_cereal_box(input_image):
    # Convert the input numpy array to PIL Image
    cover_img = Image.fromarray((input_image.astype(np.uint8)))

    # Load the template image
    template_img = Image.open('CerealBoxMaker/template.jpeg')

    # Define scaling factor for diagonal resizing
    scaling_factor = 1.5

    # Resize cover image
    rect_height = int(template_img.height * 0.32)
    new_width = int(rect_height * 0.70)
    cover_resized = cover_img.resize((new_width, rect_height), Image.LANCZOS)

    # Apply diagonal scaling
    new_width_scaled = int(new_width * scaling_factor)
    new_height_scaled = int(rect_height * scaling_factor)
    cover_resized_scaled = cover_resized.resize((new_width_scaled, new_height_scaled), Image.LANCZOS)

    # Positioning the resized cover image on the template
    left_x = int(template_img.width * 0.085)
    left_y = int((template_img.height - new_height_scaled) // 2 + template_img.height * 0.012)
    left_position = (left_x, left_y)
    
    right_x = int(template_img.width * 0.82) - new_width_scaled
    right_y = left_y
    right_position = (right_x, right_y)

    # Create a copy of the template to paste on
    template_copy = template_img.copy()

    # Paste the resized and scaled cover image
    template_copy.paste(cover_resized_scaled, left_position)
    template_copy.paste(cover_resized_scaled, right_position)

    # Convert the PIL Image back to a numpy array
    template_copy_array = np.array(template_copy)
    
    return template_copy_array

# Your existing Gr.Interface for the model that takes text and returns an image
iface = gr.Interface.load("models/ostris/super-cereal-sdxl-lora")

# Chain the existing interface with your new cereal box creation function
chained_iface = gr.Interface(create_cereal_box, inputs=iface.outputs, outputs="image")

# Launch the chained interface
chained_iface.launch()
