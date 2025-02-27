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

def apply_instruct_format(inputs: list, model: str = None, is_thinking = True, is_math = False) -> List:
    """
    Modify prompts to incorporate DeepSeek's usage recommendations: 
        1. Ask model to initiate response with "<think>\n"
        2. If math, include "please reason step by step, and put your final answer within \\boxed{}"
        see: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """
    formatted_inputs = []
                
    if is_math:
            answer_format = "please reason step by step, and put your final answer within \\boxed{}."
    else: 
            answer_format = ""

    if is_thinking:
            think_tag = "<think>\n"
    else: 
            think_tag = ""
    model_config = config['models'][model]
    system_prompt = model_config.get('format', {}).get('system', 'You are an AI assistant.')
    template = model_config['format']['template']

    for input in inputs: 
        out = template.format(system = system_prompt, content = input, 
                              answer_format = answer_format, think_tag = think_tag) 
        formatted_inputs.append(out)

    return formatted_inputs

# check + resubmit