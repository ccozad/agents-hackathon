import numpy as np
import gradio as gr
from yolo.pose import process_image, process_pose_data


def annotate_image(image):
    return process_image(image)

def generate_pose_data(image):
    return process_pose_data(image)

with gr.Blocks() as demo:
    gr.Markdown("# Yolo v11 Pose Agent")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
    with gr.Row(equal_height=True):
        image_output = gr.Image(type="pil", label="Annotated Image")
        pose_data = gr.Json(label="Pose Data")
    with gr.Row():
        annotate_image_button = gr.Button("Annotate Image")
        generate_pose_data_button = gr.Button("Generate Pose Data")

    annotate_image_button.click(annotate_image, inputs=image_input, outputs=image_output)
    generate_pose_data_button.click(generate_pose_data, inputs=image_input, outputs=pose_data)

demo.launch(mcp_server=True)