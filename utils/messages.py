import json

def get_prepend_messages(prepend_type, prepend_conversation_steps, model_path):
    if prepend_conversation_steps == 0:
        return []
    if prepend_type != "diverse-v1":
        raise NotImplementedError(f"Prepend type {prepend_type} not implemented")
    
    if model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
        prepend_messages = json.load(
            open("./cache/Qwen-2p5-7B/Qwen-2p5-7B_diverse-v1_temp0.600_p0.950_sampleTrue_max512.json")
        )["conversation_history"]
    elif model_path == "Qwen/Qwen3-VL-8B-Instruct":
        prepend_messages = json.load(
            open("./cache/Qwen-3-8B/Qwen-3-8B_diverse-v1_temp0.600_p0.950_sampleTrue_max512.json")
        )["conversation_history"]
    elif model_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct":
        prepend_messages = json.load(
            open("./cache/LLaVA-OV-1.5-8B/LLaVA-OV-1.5-8B-diverse-v1-True-512-True-0.6-0.95-0.json")
        )["conversation_history"]
    else:
        raise NotImplementedError(f"Model path {model_path} not supported")

    # drop system message
    if prepend_messages[0]["role"] == "system":
        prepend_messages = prepend_messages[1:]

    assert 2*prepend_conversation_steps <= len(prepend_messages), "Insufficient prepend messages"
    return prepend_messages[:2*prepend_conversation_steps]


def construct_messages(image_path, prompts, targets, model_path, prepend_type=None, prepend_mode="prepend", prepend_conversation_steps=0, max_message_length=None):
    """
    Construct the conversation messages based on the provided arguments.
    
    Args:
        image_path: Path to the image
        prompts: List of prompts
        targets: List of target strings
        prepend_type: Type of prepend messages to use
        prepend_mode: Mode of prepend ("prepend" or "insert")
        prepend_conversation_steps: Number of conversation steps to prepend
        max_message_length: Maximum length of a message
    
    Returns:
        tuple: (messages, messages_2) where messages_2 is None if not needed
    """
    assert prepend_conversation_steps >= 0
    prepend_messages = get_prepend_messages(prepend_type, prepend_conversation_steps, model_path)
    
    # truncate messages if max_message_length is specified
    if max_message_length is not None:
        for message in prepend_messages:
            message["content"] = message["content"][:max_message_length]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": prompts[0],
                },
            ],
        },
        {
            "role": "assistant",
            "content": targets[0],
        },
    ]
    for i in range(1, len(prompts)):
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": prompts[i],
                },
            ],
        })
        messages.append({
            "role": "assistant",
            "content": targets[i],
        })

    # prepend or insert prepended messages
    if prepend_mode == "prepend":
        messages = prepend_messages + messages
    elif prepend_mode == "insert":
        messages = messages[:2] + prepend_messages + messages[2:]
    else:
        raise ValueError(f"Invalid prepend mode: {prepend_mode}")

    return messages

