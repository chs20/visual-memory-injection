import argparse
import torch
import numpy as np
import os
from datetime import datetime
import time
from utils.general import str2bool, print_gpu_memory, print_runtime, timing_decorator
from utils.messages import construct_messages
from attacks.apgd import apgd_attack_on_image_input
import json
from utils.data import load_google_landmarks_dataset, load_handselected_landmarks_dataset, load_coco_dataset
from utils.processor import load_model_and_processor, sanity_check_normalizer
from utils.logger import Logger
import matplotlib.pyplot as plt
import random
import uuid
from PIL import Image


def parse_semicolon_separated(string_arg, name="strings"):
    """Parse semicolon-separated strings from command line argument.
    
    Args:
        string_arg: String containing semicolon-separated values
        name: Name of the argument for logging purposes
        
    Returns:
        List of strings
    """
    if string_arg is None:
        return []
    
    strings = [s for s in string_arg.split(';') if s]
    print(f"Using {len(strings)} {name}: {strings}")
    return strings

def substitute_placeholders(template_string, landmark_name, city):
    """Substitute placeholders in template string with actual values.
    
    Args:
        template_string: String with placeholders like {place_name} and {city_name}
        landmark_name: Actual landmark name
        city: Actual city name
        
    Returns:
        String with placeholders substituted
    """
    return template_string.replace('{place_name}', landmark_name).replace('{city_name}', city)

def get_target_strings_from_clean_outputs(target_strings_template, image_path, model_path, prompt):
    """Get target strings from clean outputs."""
    clean_outputs_file_name = f"clean_outputs_{model_path.split('/')[-1]}.json"
    clean_outputs_file_path = os.path.join(os.path.dirname(image_path), clean_outputs_file_name)
    with open(clean_outputs_file_path, "r") as f:
        clean_outputs = json.load(f)
    assert clean_outputs["metadata"]["prompt"] == prompt, f"Prompt mismatch: {clean_outputs['metadata']['prompt']} != {prompt}"
    clean_output = clean_outputs["outputs"][image_path.split('/')[-1]]
    # substitue "clean_output" with the actual clean output
    target_strings = target_strings_template.copy()
    for i, template in enumerate(target_strings_template):
        if template == "{clean_output}":
            target_strings[i] = clean_output
    return target_strings


def main():
    parser = argparse.ArgumentParser(description="Run the attack.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Path to the model.")
    parser.add_argument("--enable_flash_attn", type=str2bool, default=True, help="Enable flash-attention for better acceleration and memory saving.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g., 'landmarks'). Mutually exclusive with --image_path.")
    parser.add_argument("--n_data", type=int, default=None, help="Limit number of images to process from dataset.")
    parser.add_argument("--prompts", type=str, default="Describe this image.", help="The input prompt.")
    parser.add_argument("--prepend_type", type=str, default=None, choices=["None", "long1", "long2", "diverse-v1"], help="Type of prepend messages to use.")
    parser.add_argument("--prepend_mode", type=str, default="prepend", choices=["prepend", "insert"], help="Mode of prepend messages.")
    parser.add_argument("--prepend_conversation", type=int, default=0, help="Prepend n conversation steps as history to the prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature of generate")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p of generate")
    parser.add_argument("--response_prefix", type=str, default="", help="Text to prefill at the start of the model's response.")
    parser.add_argument("--do_sample", type=str2bool, default=True, help="Do sample for the model.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model.")
    parser.add_argument("--target_strings", type=str, default=None, help="Target output strings separated by semicolons (e.g., 'target1;target2;target3').")
    parser.add_argument("--attack_type", type=str, choices=["none", "pgd", "apgd"], default="apgd", help="Type of adversarial attack to use (none, pgd, apgd).")
    parser.add_argument("--epsilon", type=float, default=32, help="Epsilon value for adversarial attacks, gets divided by 255.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for adversarial attacks (default: 100).")
    parser.add_argument("--alpha_apgd", type=float, default=2.0, help="Alpha value for APGD attack.")
    parser.add_argument("--use_best_loss", type=str2bool, default=True, help="Use the best loss adversarial image, otherwise use the final adversarial image.")
    parser.add_argument("--cycle_context_frequency", type=int, default=0, help="How often to cycle through conversation context during APGD attack (in attack steps).")
    parser.add_argument("--cycle_mode", type=str, choices=["linear", "circular"], help="Mode of context cycling during APGD attack.")
    parser.add_argument("--use_all_contexts_at_once", type=str2bool, default=False, help="Compute loss on all contexts at once (sum of losses) instead of cycling through them.")
    parser.add_argument("--context_update_frequency", type=int, default=-1, help="How often to update conversation context during APGD attack (in attack steps) by generating new responses. (Currently not supported.)")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory path to save logs. Image filename will be automatically generated as '{original_name}_adv_{timestamp}.png'.")
    parser.add_argument("--verbose_apgd", type=str2bool, default=True, help="Verbose output for APGD attack.")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set up logging to capture all output    
    timestamp = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file_path = os.path.join(args.log_dir, f"vision_attack_{timestamp}.log")
    else:
        # use temp file
        log_file_path = "/tmp/vision_attack_{timestamp}.log"
    logger = Logger(log_file_path)
    logger.write(f"Logging all output to: {log_file_path}")
    logger.write(f"Arguments: {args}")
    
    # Validate arguments
    if args.dataset is not None and args.image_path is not None:
        raise ValueError("Cannot specify both --dataset and --image_path. Use one or the other.")
    if args.dataset is None and args.image_path is None:
        raise ValueError("Either --dataset or --image_path must be specified.")
    
    # resolve args
    args.epsilon = args.epsilon / 255
    if args.context_update_frequency > 0:
        assert args.prepend_conversation > 0, "Context update frequency is only supported with prepended conversation."
    if args.prepend_type == "diverse-v1":
        args.max_message_length = 256
    else:
        args.max_message_length = None
    logger.write(f"Max message length: {args.max_message_length}")

    # load model and processor
    model, (processor, normalizer) = load_model_and_processor(args.model_path, args.enable_flash_attn, do_normalize=False)


    prompts = parse_semicolon_separated(args.prompts, "prompts")
    target_strings_template = parse_semicolon_separated(args.target_strings, "target strings")
    if len(target_strings_template) == 0:
        raise ValueError("No target strings provided for APGD attack")
    
    # Determine processing mode
    if args.dataset is not None:
        # Dataset mode
        if args.dataset == "landmarks":
            assert args.n_data in [None, 20]
            dataset_images = load_handselected_landmarks_dataset()
        elif args.dataset == "google-landmarks":
            dataset_images = load_google_landmarks_dataset(args.n_data)
        elif args.dataset.startswith("coco"):
            dataset_images = load_coco_dataset(args.dataset)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        logger.write(f"Processing {len(dataset_images)} images from {args.dataset} dataset")
        
        # Do sanity check on first image for dataset mode
        sanity_check_normalizer(processor, normalizer, dataset_images[0]['image_path'], model.device)
        
        # Process each image in the dataset
        for i, image_data in enumerate(dataset_images):
            logger.write(f"\n--- Processing image {i+1}/{len(dataset_images)}: {image_data['image_path']} ---")
            
            # Handle placeholder substitution based on dataset type
            if args.dataset == "landmarks":
                logger.write(f"Landmark: {image_data['landmark_name']}, City: {image_data['city']}")
                # Substitute placeholders in target strings
                target_outputs = [substitute_placeholders(template, image_data['landmark_name'], image_data['city']) 
                                for template in target_strings_template]
            else:
                # For COCO datasets or landmarks use saved clean outputs
                target_outputs = get_target_strings_from_clean_outputs(
                    target_strings_template=target_strings_template, image_path=image_data['image_path'], model_path=args.model_path, prompt=prompts[0]
                    )
            
            logger.write(f"Target strings: {target_outputs}")
            
            # Process this image
            logger.write(print_gpu_memory())
            process_single_image(
                model=model, processor=processor, normalizer=normalizer,
                image_path=image_data['image_path'], prompts=prompts, target_outputs=target_outputs,
                args=args, logger=logger
            )
            torch.cuda.empty_cache()
    else:
        # Sanity-check: compare processor vs. custom normalizer outputs once.
        sanity_check_normalizer(processor, normalizer, args.image_path, model.device)
        # Single image mode
        target_outputs = target_strings_template
        process_single_image(
            model=model, processor=processor, normalizer=normalizer,
            image_path=args.image_path, prompts=prompts, target_outputs=target_outputs,
            args=args, logger=logger
        )


@timing_decorator
def process_single_image(model, processor, normalizer, image_path, prompts, target_outputs, args, logger):
    """Process a single image through the attack pipeline."""
    
    messages = construct_messages(
        image_path=image_path, prompts=prompts, targets=target_outputs, model_path=args.model_path, prepend_type=args.prepend_type, prepend_mode=args.prepend_mode,
        prepend_conversation_steps=args.prepend_conversation, max_message_length=args.max_message_length
    )
    # logger.write(f"Messages: {json.dumps(messages, indent=2)}", to_console=False)

    n_targets = len(target_outputs)
    messages_list = []
    # get messages list for cycle context
    if args.cycle_context_frequency > 0 or args.use_all_contexts_at_once:
        # we construct messages_list by removing context conversation steps
        assert args.prepend_conversation > 0, "Cycle context frequency is only supported with prepended conversation."
        # handle system message
        system_message = None
        if messages[0]["role"] == "system":
            system_message = messages[0]
            messages = messages[1:]
        # construct messages list
        for i in range(0,args.prepend_conversation+1):
            if args.prepend_mode == "prepend":
                messages_curr = messages[:2*i] + messages[-2*n_targets:]  # we keep the first i conversation steps and the ones that are attacked
            elif args.prepend_mode == "insert":  # first prompt and answer need to be kept
                messages_curr = messages[:2] + messages[2:2*(i+1)] + messages[-2*n_targets + 2:]
            else:
                raise ValueError(f"Invalid prepend mode: {args.prepend_mode}")
            if system_message is not None:
                messages_curr = [system_message] + messages_curr
            messages_list.append(messages_curr)
    else:
        messages_list = [messages]
    logger.write(f"Messages list: {json.dumps(messages_list, indent=2)}", to_console=False)


    # Preparation 
    image_inputs = [Image.open(image_path).convert("RGB")]
    video_inputs = None
    assert processor.image_processor.do_normalize == False, "Processor should be configured to NOT normalize images"


    # --- attack ---
    start_time = time.time()
    if args.attack_type == "apgd":
        image_inputs, loss_steps = apgd_attack_on_image_input(
            processor=processor,
            model=model,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            normalizer=normalizer,
            target_outputs=target_outputs,
            messages_list=messages_list,
            epsilon=args.epsilon,
            num_steps=args.iterations,
            alpha=args.alpha_apgd,
            use_best_loss=args.use_best_loss,
            verbose=args.verbose_apgd,
            context_update_frequency=args.context_update_frequency,
            cycle_context_frequency=args.cycle_context_frequency,
            cycle_mode=args.cycle_mode,
            use_all_contexts_at_once=args.use_all_contexts_at_once,
            logger=logger,
            generation_params={
                "prepend_mode": args.prepend_mode,
                "prepend_conversation_steps": args.prepend_conversation,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
                "max_tokens": args.max_tokens,
                "max_message_length": args.max_message_length,
            },
        )
    else:
        loss_steps = None
        logger.write("No attack applied.")
    attack_time = time.time() - start_time
    logger.write(print_runtime(attack_time, desc="Attack time"))
    
    
    # If attack_type is "none", no perturbation is applied.
    # --- End PGD Adversarial Attack ---

    # Generate timestamp for logging
    timestamp_inner = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    
    # Save loss values and plot if attack was performed
    if loss_steps is not None and args.log_dir is not None:
        # Create losses directory
        losses_dir = os.path.join(args.log_dir, "losses")
        os.makedirs(losses_dir, exist_ok=True)
        
        # Generate filenames
        original_image_name = os.path.splitext(os.path.basename(image_path))[0]
        loss_filename = f"{original_image_name}_losses_{timestamp_inner}.npy"
        plot_filename = f"{original_image_name}_losses_{timestamp_inner}.pdf"
        loss_path = os.path.join(losses_dir, loss_filename)
        plot_path = os.path.join(losses_dir, plot_filename)
        
        # Save loss values as numpy array
        np.save(loss_path, loss_steps)
        logger.write(f"Loss values saved to: {loss_path}")
        
        # Create and save plot
        plt.figure(figsize=(10, 6))
        # loss_steps is (n_iter, batch_size), we take the first batch element
        iterations = np.arange(len(loss_steps))
        plt.plot(iterations, loss_steps[:, 0], marker=".", linestyle="-", linewidth=0.75, markersize=2)
        plt.yscale('log')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'APGD Attack Loss - {original_image_name}. Final: {loss_steps[-1][0]:.3e}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        logger.write(f"Loss plot saved to: {plot_path}")
    
    # Save adversarial image if requested
    if args.log_dir is not None and len(image_inputs) > 0 and args.attack_type != "none":
        # Ensure the directory exists
        save_dir = os.path.join(args.log_dir, "adversarial-images")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Generate automatic filename based on original image name and timestamp
        original_image_name = os.path.splitext(os.path.basename(image_path))[0]
        uuid_cur = str(uuid.uuid4())[:8]
        filename = f"{original_image_name}_adv_{timestamp_inner}_{uuid_cur}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Save the adversarial image
        adversarial_image = image_inputs[0]
        adversarial_image.save(save_path)
        logger.write(f"Adversarial image saved to: {save_path}")
    else:
        save_path = "N/A"

    first_prompt_idx = 2*args.prepend_conversation if args.prepend_mode == "prepend" else 0
    if messages[first_prompt_idx]["role"] == "system":  # if system prompt is used, increment index
        first_prompt_idx += 1
    first_prompt_text = processor.apply_chat_template(
        messages[:first_prompt_idx+1], tokenize=False, add_generation_prompt=True
    )
    prefix_text = args.response_prefix
    prompt_inputs = processor(
        text=[first_prompt_text + prefix_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,  
        return_tensors="pt",
    )

    # Inference: Generation of the output
    prompt_inputs = prompt_inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=prompt_inputs.input_ids,
            pixel_values=normalizer(prompt_inputs["pixel_values"].to(model.device)),
            image_grid_thw=prompt_inputs.get("image_grid_thw"),
            attention_mask=prompt_inputs.attention_mask,
            do_sample=args.do_sample,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            # min_new_tokens=100
        )
    # Remove the prompt and prefix from the output, so only the new generation is shown after the prefix
    generated_ids_trimmed = [
        out_ids[prompt_inputs.input_ids.shape[1]:] for out_ids in generated_ids
    ]
    output_text = [prefix_text + processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids_trimmed]
    logger.write(output_text)
    
    # Clean up GPU memory after first generation
    del generated_ids, prompt_inputs
    torch.cuda.empty_cache()

    # Second target prompt (is at the end)
    second_prompt_text = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    second_prompt_inputs = processor(
        text=[second_prompt_text + prefix_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,  
        return_tensors="pt",
    )
    second_prompt_inputs = second_prompt_inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=second_prompt_inputs.input_ids,
            pixel_values=normalizer(second_prompt_inputs["pixel_values"].to(model.device)),
            image_grid_thw=second_prompt_inputs.get("image_grid_thw"),
            attention_mask=second_prompt_inputs.attention_mask,
            do_sample=args.do_sample,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    # Remove the prompt and prefix from the output, so only the new generation is shown after the prefix
    generated_ids_trimmed = [
        out_ids[second_prompt_inputs.input_ids.shape[1]:] for out_ids in generated_ids
    ]
    output_text2 = [prefix_text + processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids_trimmed]
    logger.write(output_text2)
    
    # Clean up GPU memory after second generation
    del generated_ids, second_prompt_inputs
    torch.cuda.empty_cache()

    # save the output to generated_outputs.tsv
    if args.log_dir is not None:
        # create the log_dir
        os.makedirs(args.log_dir, exist_ok=True)
        # save all args to args.json with timestamp
        args_file_path = os.path.join(args.log_dir, "args.json")
        # Load existing args if file exists, otherwise start with empty list
        try:
            with open(args_file_path, 'r') as f:
                args_list = json.load(f)
        except (FileNotFoundError):
            args_list = []
        # Add new args entry
        args_data = {
            "timestamp": timestamp_inner,
            "image_adv_name": filename,
            "target_strings": target_outputs,
            "args": args.__dict__
        }
        args_list.append(args_data)
        # Save updated list
        with open(args_file_path, 'w') as f:
            json.dump(args_list, f, indent=2)        
    else:
        logger.write("No log directory provided. Output not saved.")

if __name__ == "__main__":
    main() 