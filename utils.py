import pandas as pd
import torch
import time
from typing import List, Dict, Any

def bloomer(n = 1): 
    """
    freedom involves attention and awareness and discipline and effort.
    """
    print('(˵◕ ɛ ◕˵✿)'+ ' ❀'*n)

# df => list of dicts
def batch_generate(df: pd.DataFrame = None, batch_size: int = 5) -> List[Dict[str, Any]]: 
    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_dict = {
            "prompts": batch_df["prompt"].tolist(),
            "labels": batch_df["label"].tolist(),
            "metadata": {col: batch_df[col].tolist() for col in df.columns if col not in ["prompt", "label"]},
        }
        batches.append(batch_dict)
    return batches

# prompts => toks 
def tokens_generate(batches: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    """
    Tokenizes the 'prompts' field in each batch while keeping context. 

    Args: 
        batches (List[Dict[str, Any]]): List of batched data where each element is a dictionary.
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer.

    Returns: 
        List[Dict[str, Any]]: List of tokenized batches where each element is a dictionary.
    """
    tokenized_batches = []
    for batch in batches: 
        tokenized_prompts = tokenizer(
            batch["prompts"],
            padding = True, 
            truncation = True, 
            return_tensors = 'pt'
        )

        tokenized_batches.append({
            "tokenized_prompts": tokenized_prompts, # dict. of tensors
            "labels": batch["labels"],
            "metadata": batch["metadata"]
        })

    return tokenized_batches

# toks => response 
def run_inference(model, tokens, time_tracking = True): 
    """
    Inference on tokenized batches + option for tracking inference time. 
    """
    outputs = []
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(tokens): 
            inputs = {k: v.to(device) for k, v in batch['tokenized_prompts'].items()}
            if time_tracking:
                start_time = time.time()

            outputs = model.generate(**inputs)

            if time_tracking:
                    inference_time = time.time() - start_time 
            else:
                inference_time = None

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)

            outputs.append({
                "batch_idx": batch_idx,
                "inputs": inputs,
                "outputs": decoded_outputs,
                "labels": batch.get("labels", None),
                "metadata": batch.get("metadata", {}),
                "inference_time": inference_time
            })

    return outputs
            

# TK - check response format for <think></think> – check usage guide from HF; it's known that output doesn't always include the <think> tokens!

# response linting – need this for evals :3 (must disambiguate cot + final answer...in order to eval. correctness)
    




# TK – need function to check mem. usage :) 
