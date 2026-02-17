import argparse
import os
import json
import sys
from datetime import datetime
import time
from multi_turn_conversation import MultiTurnConversation
from utils.general import format_reasoning_and_answer, print_runtime
from utils.naming import get_model_short_name
from utils.logger import Logger
from utils.general import set_seeds
from evaluation.eval_queries import PROMPTS_DICT, TARGET_PROMPTS_DICT
from utils.conversation_cache import load_from_cache, save_to_cache, load_from_insert_cache, save_to_insert_cache
from utils.config import GENERATION_CONFIGS


def run_n_turn_conversation(image_path, prompts_name, prompts, model_path, enable_flash_attn,
                           max_tokens, temperature, top_p, 
                           do_sample, system_message, seed=0, target_prompts=None, target_prompts_name=None,
                           prepend_mode="prepend", save_dir=None, logger=None, use_cache=True, save_to_cache=True):
    """
    Run an n-turn conversation with the specified parameters.
    
    Args:
        image_path: Path to the input image
        prompts_name: Name of the prompts
        prompts: List of prompts to send in sequence
        model_path: Path to the model
        enable_flash_attn: Whether to use flash attention
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        system_message: System message to add to the conversation history
        seed: Random seed
        target_prompts: List of target prompts, just for logging purposes
        target_prompts_name: Name of the target prompts (for cache key in insert mode)
        prepend_mode: Mode of prepend messages ("prepend" or "insert")
        save_dir: Path to save the conversation and arguments
        use_cache: Whether to retrieve conversation history from cache (if exists)
        save_to_cache: Whether to save conversation history to cache
    """
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if target_prompts is not None:
        # make sure all target prompts are in the prompts list
        assert all(target_prompt in prompts for target_prompt in target_prompts)
    
    # Initialize conversation system
    print(f"Loading model from: {model_path}")
    conversation = MultiTurnConversation(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        enable_flash_attn=enable_flash_attn,
        system_message=system_message,
        seed=seed
    )
    
    # Prepare arguments for saving
    args_dict = {
        "image_path": image_path,
        "prompts_name": prompts_name,
        "prompts": prompts,
        "target_prompts": target_prompts,
        "model_path": model_path,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "system_message": system_message,
        "seed": seed,
        "timestamp": datetime.now().isoformat()
    }
    
    num_turns = len(prompts)
    logger.write(f"\n=== Starting {num_turns}-turn conversation ===")
    print(f"Image: {image_path}")
    print(f"Number of turns: {num_turns}")
    for i, prompt in enumerate(prompts):
        print(f"Turn {i+1} prompt: {prompt}")
    print("=" * 50)
    
    # Find which turn contains the image marker
    image_turn_idx = None
    for i, prompt in enumerate(prompts):
        if prompt.startswith("<image>"):
            if image_turn_idx is not None:
                raise ValueError(f"Multiple prompts contain <image> marker (turns {image_turn_idx+1} and {i+1}). Only one is allowed.")
            image_turn_idx = i

    if image_turn_idx is None:
        raise ValueError(f"No prompt contains <image> marker. Exactly one prompt must start with <image>. Prompts: {prompts}")
    
    # Extract text-only prompts before the image turn for caching (prepend mode)
    text_only_prompts = prompts[:image_turn_idx]
    model_short_name = get_model_short_name(model_path)
    
    # Try to load from cache
    cached_turns = 0
    cache_result = None
    if use_cache:
        if prepend_mode == "prepend" and len(text_only_prompts) > 0:
            # Prepend mode: cache text-only turns before the image
            print(f"\nChecking cache for {len(text_only_prompts)} text-only prompts before image turn...")
            cache_result = load_from_cache(
                model_short_name=model_short_name,
                prompts_name=prompts_name,
                prompts=text_only_prompts,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_tokens=max_tokens
            )
        elif prepend_mode == "insert":
            # Insert mode: cache per-image (image is in first turn)
            print(f"\nChecking insert cache for image: {os.path.basename(image_path)}...")
            assert len(target_prompts) <= 2, "One target prompt before and one after the context"
            cache_result = load_from_insert_cache(
                model_short_name=model_short_name,
                prompts_name=prompts_name,
                target_prompts_name=target_prompts_name or "default",
                image_path=image_path,
                num_prompts=num_turns-1,  # NOTE: this assumes that we have exactly one target turn at the end
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_tokens=max_tokens
            )
        else:
            print("Skipping cache check (first turn contains image in prepend mode OR use_cache=False).")
    else:
        print("Caching disabled (use_cache=False).")

    if cache_result is not None:
            cached_turns, cached_history = cache_result
            print(f"✓ Cache hit! Loaded {cached_turns} turns from cache.")
            logger.write(f"\n[Cache] Loaded {cached_turns} turns from cache")
            
            for msg in cached_history:
                if msg["role"] == "system":
                    # make sure the same system message is used
                    assert conversation.conversation_history[0]["role"] == "system"
                    assert conversation.conversation_history[0]["content"] == msg["content"]
                    continue
                conversation.conversation_history.append(msg)
                
            # Log cached turns
            for step in range(len(conversation.conversation_history)):
                role = conversation.conversation_history[step]["role"]
                content = conversation.conversation_history[step]["content"]
                if role == "user":
                    logger.write(f"\n--- Turn {(step + 1)//2} (from cache) ---")
                logger.write(f"{role}:\n{content}")
    else:
            print("✗ Cache miss. Will compute all turns.")
    
    # Unified conversation loop
    for turn in range(num_turns):
        # Skip turns that were loaded from cache
        if turn < cached_turns:
            continue
        
        # set seeds again for each turn, in particular so that random state does not depend on whether the turn was loaded from cache or not
        # use turn-specific seed for variability
        set_seeds(seed + turn)  

        logger.write(f"\n--- Turn {turn + 1} ---")
        
        if turn == image_turn_idx:
            # This is the image turn - strip <image> marker and send with image
            cleaned_prompt = prompts[turn].replace("<image>", "").strip()
            logger.write(f"User: [Image: {image_path}] {cleaned_prompt}")
            conversation.add_image_message("user", image_path, cleaned_prompt)
        else:
            # Regular text turn
            logger.write(f"User: {prompts[turn]}")
            conversation.add_text_message("user", prompts[turn])
        
        print(f"Generating response for turn {turn + 1}...")
        response = conversation.get_response()
        conversation.add_text_message("assistant", response)
        
        logger.write(f"Assistant:\n{format_reasoning_and_answer(response)}")
    
    logger.write(f"\n=== {num_turns}-turn conversation completed ===")
    
    # Save to cache if we computed any new turns
    if save_to_cache:
        if prepend_mode == "prepend" and len(text_only_prompts) > 0 and cached_turns < len(text_only_prompts):
            # Prepend mode: save text-only turns before the image
            # Extract conversation history for text-only turns (before the image turn)
            # Each turn consists of 2 messages: user and assistant
            text_only_history = conversation.conversation_history[:len(text_only_prompts) * 2]
            
            print(f"\nSaving {len(text_only_prompts)} text-only turns to cache...")
            save_to_cache(
                model_short_name=model_short_name,
                prompts_name=prompts_name,
                prompts=text_only_prompts,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_tokens=max_tokens,
                conversation_history=text_only_history
            )
        
        elif prepend_mode == "insert" and cached_turns < num_turns-1:
            # Insert mode: save entire conversation per-image, except the last turn
            # save if cached turns is less than num_turns-1, NOTE: assumes the we have exactly one target turn at the end
            assert len(target_prompts) <= 2, "One target prompt before and one after the context"
            print(f"\nSaving {num_turns} turns to insert cache for image: {os.path.basename(image_path)}...")
            save_to_insert_cache(
                model_short_name=model_short_name,
                prompts_name=prompts_name,
                target_prompts_name=target_prompts_name or "default",
                image_path=image_path,
                prompts=prompts,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_tokens=max_tokens,
                conversation_history=conversation.conversation_history[:-2]
            )
    
    # Save conversation and arguments if save_path is specified
    if save_dir:
        # Create directory if it doesn't exist
        save_dir_convs = os.path.join(save_dir, "convs")
        os.makedirs(save_dir_convs, exist_ok=True)
        
        save_path = os.path.join(save_dir_convs, f"{os.path.basename(image_path).split('.')[0]}.json")

        # Save conversation history
        conversation.save_conversation(save_path)
        
        # Save arguments to args.json with timestamp
        args_file_path = os.path.join(save_dir, "args.json")
        # Load existing args if file exists, otherwise start with empty list
        try:
            with open(args_file_path, 'r', encoding='utf-8') as f:
                args_list = json.load(f)
        except (FileNotFoundError):
            args_list = []
        # Add new args entry
        args_data = {
            "timestamp": args_dict["timestamp"],
            "image_path": args_dict["image_path"],
            "args": args_dict
        }
        args_list.append(args_data)
        # Save updated list
        with open(args_file_path, 'w', encoding='utf-8') as f:
            json.dump(args_list, f, indent=2, ensure_ascii=False)
        
        print(f"Arguments saved to: {args_file_path}")
        print(f"Conversation saved to: {save_path}")
    
    return conversation.conversation_history


def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversation with vision-language models.")
    parser.add_argument("--model", type=str, help="Q3, Q2p5", required=True)
    parser.add_argument("--transfer_from_model", type=str, default=None)
    parser.add_argument("--images", type=str)
    parser.add_argument("--prompts_name", type=str)
    parser.add_argument("--n_subprompts", type=int, nargs='+', default=[-2], help="Use subprompts of length [arg1, arg2, ...]. If -1, use all subprompts.")
    parser.add_argument("--target_prompts", type=str, default="default", help="Target prompts to use. 'default': read from args.json, 'alt3', 'alt4', ...")
    parser.add_argument("--prepend_mode", type=str, default="insert", choices=["prepend", "insert"], help="Mode of prepend messages.")
    args = parser.parse_args()


    # validations
    # if args.target_prompts != "default":
    #     assert args.target_prompts.split("-")[0] in args.images  # make sure we use the correct target prompts (phone, car, ...)

    MODEL_PATHS = { 
        "Q3": "Qwen/Qwen3-VL-8B-Instruct", "Q2p5": "Qwen/Qwen2.5-VL-7B-Instruct",
        "llava-ov": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "Q3-sealion": "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "Q3-med3": "ddvd233/QoQ-Med3-VL-8B",
    }

    model_path = MODEL_PATHS[args.model]
    transfer_from_model_path = MODEL_PATHS[args.transfer_from_model] if args.transfer_from_model is not None else model_path

    # generation config
    generation_config = GENERATION_CONFIGS[model_path]
    enable_flash_attn = generation_config["enable_flash_attn"]
    max_tokens = generation_config["max_tokens"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    do_sample = generation_config["do_sample"]
    system_message = generation_config["system_message"]

    seed = 0
    use_cache = False   
    save_to_cache = False

    model_short_name = get_model_short_name(model_path)
    transfer_from_model_short_name = get_model_short_name(transfer_from_model_path)
    if transfer_from_model_short_name != model_short_name:
        # raise NotImplementedError(f"Transfer not implemented, need to adapt save_dir (currently gets saved in transfer_from_model dir but should be saved in model_short_name dir)")
        print(f"\n\n######## WARNING: TRANSFERRING FROM {transfer_from_model_short_name} TO {model_short_name}\n\n")
    ######
    image_dir = f"./logs/{transfer_from_model_short_name}/{args.images}/adversarial-images"
    image_paths = [os.path.join(image_dir, el) for el in os.listdir(image_dir) if el.endswith(".jpg") or el.endswith(".png")]
    ######

    start_time = time.time()

    # read the attacked prompts from saved args
    args_file_path = os.path.join(os.path.dirname(image_dir), "args.json")
    if args.target_prompts == "default":
        target_prompts = json.load(open(args_file_path, "r"))[0]["args"]["prompts"].split(";")
    else:
        target_prompts = TARGET_PROMPTS_DICT[args.target_prompts]

    target_prompts[0] = "<image>" + target_prompts[0]  # add <image> marker to the first prompt
    print(f"\nIdentified the attacked prompts: {target_prompts}.\nAttention: Inserted <image> marker to the first prompt (as printed above).")

    # Define prompts for the conversation
    prompts_all = PROMPTS_DICT[args.prompts_name]["prompts"]
    
    # create subprompts, i.e. first k prompts, then k+1, etc
    if args.n_subprompts == [-2]:  # use a sparse subset of the prompts
        args.n_subprompts = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32]  # -1 = no context
        args.n_subprompts = [i for i in args.n_subprompts if i < len(prompts_all)-1]
        args.n_subprompts.append(len(prompts_all)-1)  # make sure to include the last prompt
        
    subprompts = []
    for i in args.n_subprompts:
        if args.prepend_mode == "prepend":
            subprompts_cur = prompts_all[:i+1] + target_prompts  # put attacked prompts last
        elif args.prepend_mode == "insert":
            subprompts_cur = target_prompts[:1] + prompts_all[:i+1] + target_prompts[1:]  # put attacked prompts last
        else:
            raise ValueError(f"Invalid prepend mode: {args.prepend_mode}")
        subprompts.append(subprompts_cur)
    if len(subprompts) == 0:
        assert args.prompts_name == "base"
        subprompts = [target_prompts]
    # start with the longest subprompts, so that we cache the longest subprompts first, can be reused then
    subprompts = subprompts[::-1]

    print(f"Using {len(subprompts)} subprompts: {args.n_subprompts}")

    # Set seeds
    # gets also set again in MultiTurnConversation.__init__(), so that output does not depend on the order of the prompts
    set_seeds(seed)  


    for i, prompts_cur in enumerate(subprompts):
        print("\n\n" + "=" * 80)
        print(f"Processing prompts: {i+1}/{len(subprompts)}")

        if transfer_from_model_path == model_path:
            save_dir = os.path.join(
                os.path.dirname(image_dir), f"multi-turn-{args.prompts_name}-{args.target_prompts}-{len(prompts_cur)-len(target_prompts)}"
                )
        else:
            save_dir = (
                f"./logs/{transfer_from_model_short_name}_TO_{model_short_name}/{args.images}/"
                f"multi-turn-{args.prompts_name}-{args.target_prompts}-{len(prompts_cur)-len(target_prompts)}"
            )
        
        # Set up logging to capture all output
        log_file_path = os.path.join(save_dir, "multi_turn_conversations.log")
        os.makedirs(save_dir, exist_ok=True)
        # Create logger and redirect stdout
        logger = Logger(log_file_path)
        original_stdout = sys.stdout
        # sys.stdout = logger
        
        try:
            logger.write(f"Found {len(image_paths)} images")
            logger.write(f"Logging all output to: {log_file_path}")
            logger.write("=" * 80)

            for step, image_path in enumerate(image_paths):
                logger.write(f"Processing image: {image_path} ({step+1}/{len(image_paths)})")
                try:
                    # Run the conversation
                    conversation_history = run_n_turn_conversation(
                        image_path=image_path,
                        prompts_name=args.prompts_name,
                        prompts=prompts_cur,
                        target_prompts=target_prompts,
                        target_prompts_name=args.target_prompts,
                        prepend_mode=args.prepend_mode,
                        model_path=model_path,
                        enable_flash_attn=enable_flash_attn,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,  
                        system_message=system_message,
                        seed=seed,
                        save_dir=save_dir,
                        logger=logger,
                        use_cache=use_cache,
                        save_to_cache=save_to_cache
                    )
                    
                    logger.write("\n=== Summary ===")
                    logger.write(f"Total turns: {len(conversation_history)}")
                    logger.write(f"Conversation saved to: {save_dir}")
                    logger.write("=" * 80)
                    logger.write("\n\n")
                    
                except Exception as e:
                    logger.write(f"Error processing {image_path}: {e}")
                    logger.write("=" * 80)
                    logger.write("\n\n")
                    raise e
            
            logger.write(f"All processing completed. Log saved to: {log_file_path}")
            end_time = time.time()
            logger.write(print_runtime(end_time - start_time, desc="Time taken"))
            
        finally:
            # Restore original stdout and close logger
            sys.stdout = original_stdout
            logger.close()
        
    return 0


if __name__ == "__main__":
    exit(main())
