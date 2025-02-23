import torch
import time 
import gc

def check_memory(device = 'mps'):
    if device == 'mps': 
        print("Allocated: %fGB"%(torch.mps.current_allocated_memory()/1024/1024/1024))
        print("Total: %fGB"%(torch.mps.recommended_max_memory()/1024/1024/1024))
    elif device == 'cuda':
        print("Allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("Reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("Total: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))