

import json
import os

from tqdm import tqdm
from utils.processor import load_processor
from utils.naming import DESIGN_ABLATION_RUNS, ITERATION_ABLATION_RUNS, MAIN_PAPER_RUNS, ATTACK_SHORT_NAMES, SHORT_NAME_TO_MODEL_PATH, TRANSFER_RUNS


def count_tokens_conversation(conversation, tokenizer):
    n_tokens_total = 0
    for message in conversation:
        if message["role"] == "user":
            # if image, use second content
            text = message["content"][0]["text"] if len(message["content"]) == 1 else message["content"][1]["text"]
        elif message["role"] == "assistant":
            text = message["content"]
        elif message["role"] in ["evaluation", "system"]:
            continue
        else:
            raise ValueError(f"Invalid role: {message['role']}")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        message["n_tokens"] = n_tokens
        n_tokens_total += n_tokens

    # add evaluation message
    if conversation[-1]["role"] == "evaluation":
        conversation[-1]["n_tokens_total"] = n_tokens_total
    else:
        conversation.append(
            {
                "role": "evaluation",
                "n_tokens_total": n_tokens_total
            }
        )
    return conversation



if __name__ == "__main__":
    print()

    save_dir_name = "conv_token_counts"

    runs = MAIN_PAPER_RUNS
    # runs = DESIGN_ABLATION_RUNS
    # runs = TRANSFER_RUNS
    # runs = ITERATION_ABLATION_RUNS

    for res_name in runs:
        print(f"Processing {res_name}...")

        model_short_name = res_name.split("/")[0]
        if "_TO_" in res_name:  # transfer run
            model_short_name = model_short_name.split("_TO_")[0]
        model_path = SHORT_NAME_TO_MODEL_PATH[model_short_name]
        processor = load_processor(model_path, use_fast=True, do_normalize=True, device="cpu")
        print(f"Model: {model_path}")
        # print(f"Processor: {processor}")
        tokenizer = processor.tokenizer

        res_dir = os.path.join("./logs", res_name)
        for conv_name in tqdm(os.listdir(res_dir), desc=f"Processing {res_name}"):
            if not conv_name.startswith("multi-turn"):
                continue
            convs_dir = os.path.join(res_dir, conv_name, "evaluation_results", "convs")
            if not os.path.exists(convs_dir):
                continue
            save_dir = os.path.join(res_dir, conv_name, save_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            for conv_file_name in os.listdir(convs_dir):
                conv_file_path = os.path.join(convs_dir, conv_file_name)
                with open(conv_file_path, "r") as f:
                    conversation = json.load(f)
                conversation = count_tokens_conversation(conversation, tokenizer)
                
                save_conv_file_path = os.path.join(save_dir, conv_file_name)
                with open(save_conv_file_path, "w") as f:
                    json.dump(conversation, f, indent=2)
                
                # print(f"Conversation saved to: {save_conv_file_path}")
                # print(json.dumps(conversation, indent=2))
                # exit()


