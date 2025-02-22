# gen. batches
def batch_generate(df: pd.DataFrame = None, batch_size: int = 5) -> list: 
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    return batches

# TK - check response format for <think></think> – check usage guide from HF; it's known that output doesn't always include the <think> tokens!


# TK – need function to check mem. usage :) 
