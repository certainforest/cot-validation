import pandas as pd
import torch
import time
from typing import List, Dict, Any
import yaml 
import os

config_path = os.path.join(os.path.dirname(__file__), '..', "config", "config.yaml")
config_path = os.path.abspath(config_path)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def apply_instruct_format(inputs: list, model: str = None, is_thinking = True, is_answer_format = False) -> List:
    """
    Modify prompts to incorporate DeepSeek's usage recommendations: 
        1. Ask model to initiate response with "<think>\n"
        2. If math, include "please reason step by step, and put your final answer within \\boxed{}"
        3. Anything you'd put in the system prompt should, instead, be put in the user instructions.
        see: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """
    formatted_inputs = []
                
    if is_answer_format:
            answer_format = "“Answer the below question. Answer with only the final response without any additional words or phrases. Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    else: 
            answer_format = ""

    if is_thinking:
            think_tag = "<think>\n"
    else: 
            think_tag = ""
    model_config = config['models'][model]
    template = model_config['format']['template']

    for input in inputs: 
        out = template.format(content = input, 
                              answer_format = answer_format, think_tag = think_tag) 
        formatted_inputs.append(out)

    return formatted_inputs

# def extract_answer(response: List[Dict], ): 
    