# Introduction
This is a a gathering place for a Gradio Agents hackathon team.

More info about the event: https://huggingface.co/Agents-MCP-Hackathon

# Team members

- [Charles Cozad](https://github.com/ccozad) 
- [Bruno Silva](https://github.com/brunosilvadev)

# Brainstorming

## Job agent
Given a resume and job description: Identify matches, identify gaps, generate cover letter

 - [Code](/job-agent/)

## Human Pose Widget
Given pose information, show a 3D avatar with the pose. New Gradio widget

<img src="/images/annotated_image0.jpg" width="400">

Resources
 - [Sapiens Foundation Model](https://www.meta.com/emerging-tech/codec-avatars/sapiens/)
 - [Ultralytics Pose Task](https://docs.ultralytics.com/tasks/pose/)
   - [Yolo v11 Samples](/yolo/)
     - [Detect GPU](/yolo/check_env.py)
     - [Pose Data](/yolo/pose_data.py)
     - [Annotate Pose](/yolo/annotate_pose.py)

## Next Move MCP
Take in state for a simple game (Black Jack, Uno, Poker, a maze, etc.) and suggest next move. Have agent play a few rounds of the game. Maybe have agents play each other.

## Tough conversations coach
A voice or chat-based agent that helps users rehearse difficult conversations by roleplaying as the other person. Examples: asking for a raise, practicing an apology

## Game rule explainer
An AI-powered sports explainer that helps you make sense of confusing moments during live games. Watch Judo, King's league soccer or rugby union and the agent will explain the rules and dynamics as they develop

# Inspiration

 - The Stable Diffusion Web UI, built entirely with Gradio https://github.com/AUTOMATIC1111/stable-diffusion-webui
