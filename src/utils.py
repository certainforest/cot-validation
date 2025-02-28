import pandas as pd
import torch
from tqdm import tqdm
import time
from typing import List, Dict, Any

def bloomer(n = 1): 
    """
    freedom involves attention and awareness and discipline and effort.
    """
    print('(˵◕ ɛ ◕˵✿)'+ ' ❀'*n)

# df => list of dicts
def batch_generate(df: pd.DataFrame = None, input_column = "prompt", target_column = "label", batch_size: int = 5) -> List[Dict[str, Any]]: 
    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_dict = {
            "input": batch_df[input_column].tolist(),
            "target": batch_df[target_column].tolist(),
            "metadata": {col: batch_df[col].tolist() for col in df.columns if col not in [input_column, target_column]},
        }
        batches.append(batch_dict)
    return batches

# prompts => toks 
def tokens_generate(batches: List[Dict[str, Any]], tokenizer, device = 'mps') -> List[Dict[str, Any]]:
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
            batch["input"],
            padding = True, 
            truncation = True, 
            return_tensors = 'pt'
        )
        tokenized_prompts = {k: v.to(device) for k, v in tokenized_prompts.items()}

        tokenized_batches.append({
            "input": batch["input"],
            "tokenized_prompts": tokenized_prompts, # dict. of tensors
            "target": batch["target"],
            "metadata": batch["metadata"]
        })

    return tokenized_batches

# toks => response 
def run_inference(model, tokens, tokenizer, time_tracking = True) -> List[str]: 
    """
    Inference on tokenized batches + option for tracking inference time. Self-note: tokenizer arg. here is because tokenizer is needed for decoding.
    """
    outputs = []
    with torch.no_grad(): 
        for i, batch in tqdm(enumerate(tokens), desc = 'inference', total = len(tokens)): 
            if time_tracking:
                start_time = time.time()

            response = model.generate(**tokens[i]['tokenized_prompts'],
                                      max_new_tokens = 400,
                                      temperature = 0.6
                                     )

            if time_tracking:
                    inference_time = time.time() - start_time 
            else:
                inference_time = None

            decoded_response = tokenizer.batch_decode(response, skip_special_tokens = True)

            outputs.append({
                "batch_idx": i,
                "input": tokens[i]["input"],
                "tokenized_prompts": tokens[i]["tokenized_prompts"],
                "response": decoded_response,
                "target": batch.get("target", None),
                "metadata": batch.get("metadata", {}),
                "inference_time": inference_time
            })

    return outputs

## need function to check formatting 
# TK - check response format for <think></think> – check usage guide from HF; it's known that output doesn't always include the <think> tokens! - maybe consolidate this with the bottom piece? 

# TK – response linting – need this for evals :3 (must disambiguate cot + final answer...in order to eval. correctness)

