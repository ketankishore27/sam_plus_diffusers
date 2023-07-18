from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import gradio as gr
from PIL import Image
import sys


sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam.to(device)
predictor = SamPredictor(sam)
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype = torch.float16)
pipe = pipe.to(device)
selected_pixel = []


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="prompt")

    with gr.Row():
        submit = gr.Button(label="Submit")

    def generate_mask(image, evt: gr.SelectData):
        selected_pixel.append(evt.index)
        predictor.set_image(image)
        input_points = np.array(selected_pixel)
        input_label = np.ones(input_points.shape[0])
        mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_label, multimask_output=False)
        #mask = np.logical_not(mask)
        mask = Image.fromarray(mask[0, :, :])
        return mask


    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        image = image.resize((512, 512))
        mask = Image.fromarray(mask)
        mask = mask.resize((512, 512))

        output = pipe(image=image, prompt=prompt, mask_image=mask).images[0]
        return output
    
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(inpaint, inputs=[input_img, mask_img, prompt_text], outputs=[output_img])
    

if __name__ == "__main__":
    demo.launch()