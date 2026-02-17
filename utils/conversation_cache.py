#!/usr/bin/env python3
"""
Conversation cache utilities for multi-turn conversations.
Enables caching of text-only conversation turns to speed up repeated runs.
"""

import json
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def compute_cache_key(model_short_name: str, prompts_name: str, 
                     temperature: float, top_p: float, 
                     do_sample: bool, max_tokens: int) -> str:
    """
    Generate a unique cache key based on model and generation parameters.
    
    Args:
        model_short_name: Short name of the model
        prompts_name: Name of the prompts
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        
    Returns:
        Cache key string
    """
    raise NotImplementedError("This function is not used anymore. Use the insert mode cache instead.")
    # Create a deterministic string representation
    # key_data = {
    #     "model": model_short_name,
    #     "prompts_name": prompts_name,
    #     "temperature": temperature,
    #     "top_p": top_p,
    #     "do_sample": do_sample,
    #     "max_tokens": max_tokens
    # }
    
    # Convert to JSON string for hashing (sort keys for consistency)
    # key_string = json.dumps(key_data, sort_keys=True)
    key_string = f"{model_short_name}_{prompts_name}_temp{temperature:.3f}_p{top_p:.3f}_sample{do_sample}_max{max_tokens}"
    
    # Generate hash
    # hash_obj = hashlib.sha256(key_string.encode('utf-8'))
    # hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
    
    return key_string


def get_cache_dir(model_short_name: str) -> str:
    """Get the cache directory path for a given model."""
    cache_dir = os.path.join("./cache", model_short_name)
    return cache_dir


def load_from_cache(model_short_name: str, prompts_name: str, prompts: List[str],
                   temperature: float, top_p: float,
                   do_sample: bool, max_tokens: int) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """
    Load conversation history from cache for matching prompt prefix.
    
    Args:
        model_short_name: Short name of the model
        prompts_name: Name of the prompts
        prompts: List of text prompts
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        
    Returns:
        Tuple of (num_cached_turns, conversation_history) if found, None otherwise
    """
    raise NotImplementedError("This function is not used anymore. Use the insert mode cache instead.")



def save_to_cache(model_short_name: str, prompts_name: str, prompts: List[str],
                 temperature: float, top_p: float,
                 do_sample: bool, max_tokens: int,
                 conversation_history: List[Dict[str, Any]]):
    """
    Save conversation history to cache.
    
    Args:
        model_short_name: Short name of the model
        prompts_name: Name of the prompts
        prompts: List of text prompts
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        conversation_history: Conversation history to cache
    """
    raise NotImplementedError("This function is not used anymore. Use the insert mode cache instead.")


# ============================================================================
# Insert Mode Caching Functions
# For caching conversations where the first turn contains a perturbed image.
# Cache is per-image since the conversation depends on the specific image.
# ============================================================================

def compute_insert_cache_key(image_path: str, temperature: float, top_p: float, 
                             do_sample: bool, max_tokens: int) -> str:
    """
    Generate a unique cache key for insert mode based on image and generation parameters.
    
    Args:
        image_path: Path to the image file
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        
    Returns:
        Cache key string
    """
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    temperature_formatted = f"{temperature:.3f}" if temperature is not None else "None"
    top_p_formatted = f"{top_p:.3f}" if top_p is not None else "None"
    key_string = f"{image_basename}_temp{temperature_formatted}_p{top_p_formatted}_sample{do_sample}_max{max_tokens}"
    return key_string


def get_insert_cache_dir(model_short_name: str, prompts_name: str, target_prompts_name: str) -> str:
    """Get the cache directory path for insert mode (per prompts_name subdirectory)."""
    cache_dir = os.path.join("./cache", model_short_name, "insert", f"{prompts_name}-{target_prompts_name}")
    return cache_dir


def load_from_insert_cache(model_short_name: str, prompts_name: str, target_prompts_name: str,
                           image_path: str, num_prompts: int,
                           temperature: float, top_p: float,
                           do_sample: bool, max_tokens: int) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """
    Load conversation history from cache for insert mode (per-image caching).
    
    Args:
        model_short_name: Short name of the model
        prompts_name: Name of the prompts
        target_prompts_name: Name of the target prompts
        image_path: Path to the image file
        num_prompts: Total number of prompts in the conversation
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        
    Returns:
        Tuple of (num_cached_turns, conversation_history) if found, None otherwise
    """
    cache_key = compute_insert_cache_key(
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_tokens=max_tokens
    )

    cache_dir = get_insert_cache_dir(model_short_name, prompts_name, target_prompts_name)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    assert cache_data["image_path"] == image_path, f"Image path mismatch: {cache_data['image_path']} != {image_path}"
    
    # Get the cached conversation history, limited to the requested number of turns
    # Each turn consists of 2 messages: user and assistant
    system_message_indicator = 1 if cache_data.get('conversation_history', [])[0]["role"] == "system" else 0
    conversation_history = cache_data.get('conversation_history', [])[:num_prompts * 2 + system_message_indicator]
    cached_turns = (len(conversation_history) - system_message_indicator) // 2
    
    return (cached_turns, conversation_history)


def save_to_insert_cache(model_short_name: str, prompts_name: str, target_prompts_name: str,
                         image_path: str, prompts: List[str],
                         temperature: float, top_p: float,
                         do_sample: bool, max_tokens: int,
                         conversation_history: List[Dict[str, Any]]):
    """
    Save conversation history to cache for insert mode (per-image caching).
    
    Args:
        model_short_name: Short name of the model
        prompts_name: Name of the prompts
        target_prompts_name: Name of the target prompts
        image_path: Path to the image file
        prompts: List of all prompts
        temperature: Temperature parameter
        top_p: Top-p parameter
        do_sample: Sampling flag
        max_tokens: Maximum tokens
        conversation_history: Conversation history to cache
    """
    cache_dir = get_insert_cache_dir(model_short_name, prompts_name, target_prompts_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = compute_insert_cache_key(
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_tokens=max_tokens
    )
    
    cache_data = {
        "cache_key": cache_key,
        "model_short_name": model_short_name,
        "prompts_name": prompts_name,
        "target_prompts_name": target_prompts_name,
        "image_path": image_path,
        "prompts": prompts,
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "max_tokens": max_tokens
        },
        "conversation_history": conversation_history,
        "created_at": datetime.now().isoformat()
    }
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"Insert cache saved: {cache_file}")

