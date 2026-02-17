class CONFIG:
    handselected_landmarks_data_path = "./clean_images/landmarks/handselected20_v1.json"


GENERATION_CONFIGS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "max_tokens": 512,
        "enable_flash_attn": True,
        "system_message": None,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "max_tokens": 512,
        "enable_flash_attn": True,
        "system_message": None,
    },
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "max_tokens": 512,
        "enable_flash_attn": True,
        "system_message": None,
    },
    "aisingapore/Qwen-SEA-LION-v4-8B-VL": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "max_tokens": 512,
        "enable_flash_attn": True,
        "system_message": None,
    },
    "ddvd233/QoQ-Med3-VL-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "max_tokens": 512,
        "enable_flash_attn": True,
        "system_message": None,
    },
}