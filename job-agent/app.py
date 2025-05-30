import numpy as np
import gradio as gr

def matching_skills(resume, job_description):
    # Placeholder function to simulate matching skills
    return "Skill1, Skill2, Skill3"

def missing_skills(resume, job_description):
    # Placeholder function to simulate missing skills
    return "Skill4, Skill5"

def cover_letter(resume, job_description):
    # Placeholder function to simulate cover letter generation
    return "Dear Hiring Manager, I am excited to apply for the position. My skills include..."

with gr.Blocks() as demo:
    gr.Markdown("Get help with your job search")
    with gr.Tab("Resume Tools"):
        with gr.Row():
            resume_input = gr.Textbox(label="Resume", placeholder="Paste your resume here...", lines=10)
            job_description_input = gr.Textbox(label="Job Description", placeholder="Paste the job description here...", lines=10)
        text_output = gr.Textbox(label="Output", lines=10, placeholder="Results will appear here...")
        with gr.Row():
            matches_button = gr.Button("Matching Skills")
            missing_button = gr.Button("Missing Skills")
            cover_letter_button = gr.Button("Cover Letter")

    matches_button.click(matching_skills, inputs=[resume_input,job_description_input], outputs=text_output)
    missing_button.click(missing_skills, inputs=[resume_input,job_description_input], outputs=text_output)
    cover_letter_button.click(cover_letter, inputs=[resume_input,job_description_input], outputs=text_output)

demo.launch(mcp_server=True)