import json
import os
from datetime import datetime
from evaluation.eval_queries import PROMPTS_DICT
from multi_turn_conversation import MultiTurnConversation
from utils.general import set_seeds
from utils.naming import get_model_short_name
from utils.config import GENERATION_CONFIGS



if __name__ == "__main__":
    # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "Qwen/Qwen3-VL-8B-Instruct"
    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    generation_config = GENERATION_CONFIGS[model_path]
    enable_flash_attn = generation_config["enable_flash_attn"]
    max_tokens = generation_config["max_tokens"]
    do_sample = generation_config["do_sample"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    seed = 0

    prompts_name = "diverse-v1"
    prompts = PROMPTS_DICT[prompts_name]["prompts"]
    num_prompts = len(prompts)

    set_seeds(seed)

    model_short_name = get_model_short_name(model_path)
    cache_dir = os.path.join("./cache", model_short_name)
    cache_key = f"{model_short_name}-{prompts_name}-{enable_flash_attn}-{max_tokens}-{do_sample}-{temperature}-{top_p}-{seed}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        print(f"Cache file {cache_file} already exists. Exiting...")
        exit()
    os.makedirs(cache_dir, exist_ok=True)

    model = MultiTurnConversation(
        model_path=model_path,
        enable_flash_attn=enable_flash_attn,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed
    )

    results = {
        "cache_key": cache_key,
        "model_short_name": model_short_name,
        "model_path": model_path,
        "prompts_name": prompts_name,
        "prompts": prompts,
        "params": {
            "enable_flash_attn": enable_flash_attn,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        },
        "timestamp": datetime.now().isoformat(),
    }
    for i, prompt in enumerate(prompts):
        print(f"######## Processing prompt: {prompt} ({i+1}/{num_prompts})")
        model.add_message("user", [{"type": "text", "text": prompt}])
        response = model.get_response()
        print(f"Response: {response}\n\n")
        model.add_message("assistant", response)

    results["conversation_history"] = model.conversation_history

    # save results to json in cache/
    with open(os.path.join(cache_dir, f"{cache_key}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {cache_file}")