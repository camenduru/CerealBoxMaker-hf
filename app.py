import gradio as gr
import torch
import numpy as np
from PIL import Image
import random
from diffusers import DiffusionPipeline

# Initialize DiffusionPipeline with LoRA weights
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora")

def text_to_image(prompt):
    generated_img = pipeline(prompt)
    return generated_img

def create_cereal_box(input_image):
    cover_img = input_image  # This should already be a PIL Image
    template_img = Image.open('CerealBoxMaker/template.jpeg')
    
    # Cereal box creation logic
    scaling_factor = 1.5
    rect_height = int(template_img.height * 0.32)
    new_width = int(rect_height * 0.70)
    cover_resized = cover_img.resize((new_width, rect_height), Image.LANCZOS)
    new_width_scaled = int(new_width * scaling_factor)
    new_height_scaled = int(rect_height * scaling_factor)
    cover_resized_scaled = cover_resized.resize((new_width_scaled, new_height_scaled), Image.LANCZOS)
    left_x = int(template_img.width * 0.085)
    left_y = int((template_img.height - new_height_scaled) // 2 + template_img.height * 0.012)
    left_position = (left_x, left_y)
    right_x = int(template_img.width * 0.82) - new_width_scaled
    right_y = left_y
    right_position = (right_x, right_y)
    template_copy = template_img.copy()
    template_copy.paste(cover_resized_scaled, left_position)
    template_copy.paste(cover_resized_scaled, right_position)
    
    # Convert to a numpy array for Gradio output
    template_copy_array = np.array(template_copy)
    return template_copy_array

def combined_function(prompt):
    generated_img = text_to_image(prompt)
    final_img = create_cereal_box(generated_img)
    return final_img

# Create Gradio Interface
gr.Interface(fn=combined_function, inputs="text", outputs="image").launch()
