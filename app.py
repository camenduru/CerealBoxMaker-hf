import gradio as gr
import torch
import numpy as np
from PIL import Image
import random
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora")
pipeline.to("cuda:0")

MAX_SEED = np.iinfo(np.int32).max

def text_to_image(prompt):
    seed = random.randint(0, MAX_SEED)
    negative_prompt = "ugly, blurry, nsfw, gore, blood"
    output = pipeline(prompt=prompt, negative_prompt=negative_prompt, width=1024, height=1024, guidance_scale=7.0, num_inference_steps=25, generator=torch.Generator().manual_seed(seed))
    generated_img = output.images[0]
    generated_img_array = np.array(generated_img)
    return generated_img_array

def create_cereal_box(input_image):
    cover_img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    template_img = Image.open('/content/866b9b8f50b50879120be0b87dfd6050.jpg')
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
    template_copy_array = np.array(template_copy)
    return template_copy_array

def combined_function(prompt):
    generated_img_array = text_to_image(prompt)
    final_img = create_cereal_box(generated_img_array)
    return final_img

with gr.Blocks() as app:
    gr.HTML("<div style='text-align: center;'><h1>Cereal Box Maker 🥣</h1></div>")
    gr.HTML("<div style='text-align: center;'><p>This application uses StableDiffusion XL to create any cereal box you could ever imagine!</p></div>")
    gr.HTML("<div style='text-align: center;'><h3>Instructions:</h3><ol><li>Describe the cereal box you want to create and hit generate!</li><li>Print it out, cut the outside, fold the lines, and then tape!</li></ol></div>")
    gr.HTML("<div style='text-align: center;'><p>A space by AP 🐧, follow me on <a href='https://twitter.com/angrypenguinPNG'>Twitter</a>! H/T to OstrisAI <a href='https://twitter.com/ostrisai'>Twitter</a> for their Cereal Box LoRA!</p></div>")
    
    with gr.Row():
        textbox = gr.Textbox(label="Describe your cereal box: Ex: 'Avengers Cereal'")
        btn_generate = gr.Button("Generate", label="Generate")
    
    with gr.Row():
        output_img = gr.Image(label="Your Custom Cereal Box")

    btn_generate.click(
        combined_function,
        inputs=[textbox],
        outputs=[output_img]
    )

app.queue(concurrency_count=4, max_size=20, api_open=False)
app.launch(debug=True)