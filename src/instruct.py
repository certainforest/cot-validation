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

def apply_instruct_format(inputs: list, model = None, append_response_start = True, is_math = False) -> List:
    """
    Modify prompts to incorporate DeepSeek's usage recommendations: 
        1. Ask model to initiate response with "<think>\n"
        2. If math, include "please reason step by step, and put your final answer within \boxed{}"
        see: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """
    formatted_inputs = []

    

        if is_math:
            formatted += "please reason step by step, and put your final answer within \boxed{}"
    
        if append_response_start: 
            formatted += "\n<|assistant|>"

    return formatted