import argparse
import torch
from functools import wraps
import time
import numpy as np
import random


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def find_subtensor_indices(tensor, sub):
    n, m = tensor.size(0), sub.size(0)
    for i in range(n - m + 1):
        if torch.equal(tensor[i:i+m], sub):
            return i, i+m  # start, end (exclusive)
    return None

def split_reasoning_and_answer(text):
    """Split the text into reasoning and answer."""
    if "<reasoning>" not in text:
        # not a reasoning model, return empty string for reasoning
        return "", text
    if "<answer>" in text:
        # contains both reasoning and answer, split accordingly
        reasoning = text.split("<answer>")[0]
        answer = "<answer>" + text.split("<answer>")[1]
        return reasoning, answer
    else:
        # output contains only reasoning
        return text, ""

def format_reasoning_and_answer(text):
    """Format the reasoning and answer."""
    reasoning, answer = split_reasoning_and_answer(text)
    text = f"[reasoning] {reasoning}\n[answer] {answer}"
    return text


def print_runtime(runtime, desc="Runtime"):
    """Return the runtime in h:m:s (total seconds)"""
    return f"{desc}: {int(runtime // 3600):02d}h:{int((runtime % 3600) // 60):02d}m:{int(runtime % 60):02d}s ({runtime:.2f}s)"

def timing_decorator(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract logger from kwargs
        logger = kwargs.get('logger')
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Log the result using print_runtime utility
        if logger:
            logger.write(print_runtime(elapsed_time, desc=f"{func.__name__} execution time"))
        
        return result
    return wrapper

def print_gpu_memory(device=None):
    """Return current GPU memory usage as string."""
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    
    return f"GPU {device} Memory:  [Allocated] {allocated:.2f} GB  [Reserved] {reserved:.2f} GB  [Max Alloc] {max_allocated:.2f} GB"


def set_seeds(seed, cuda_reproducibility=True):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)