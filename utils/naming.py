#!/usr/bin/env python3
import sys

MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen-2p5-7B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen-3-8B",
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "LLaVA-OV-1.5-8B",
    "aisingapore/Qwen-SEA-LION-v4-8B-VL": "Q3-sealionv4-8B",
    "ddvd233/QoQ-Med3-VL-8B": "Q3-med3-8B",
}

# Reverse mapping: short name -> full model path
SHORT_NAME_TO_MODEL_PATH = {v: k for k, v in MODEL_SHORT_NAMES.items()}

MAIN_PAPER_RUNS = [
    "Qwen-2p5-7B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "Qwen-2p5-7B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "Qwen-2p5-7B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "Qwen-2p5-7B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    ##
    "Qwen-3-8B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    ##
    "LLaVA-OV-1.5-8B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "LLaVA-OV-1.5-8B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "LLaVA-OV-1.5-8B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "LLaVA-OV-1.5-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
]

TRANSFER_RUNS = [
    # from Qwen 3
    "Qwen-3-8B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    ## -> sealion
    "Qwen-3-8B_TO_Q3-sealionv4-8B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-sealionv4-8B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-sealionv4-8B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-sealionv4-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    ## -> med3
    "Qwen-3-8B_TO_Q3-med3-8B/coco20-attack2000-eps8-motorola-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-med3-8B/coco20-attack2000-eps8-car-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-med3-8B/landmarks-attack2000-eps8-party-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B_TO_Q3-med3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
]

DESIGN_ABLATION_RUNS = [
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-single-target",
]

ITERATION_ABLATION_RUNS = [
    "Qwen-3-8B/landmarks-attack8000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear",
    "Qwen-3-8B/landmarks-attack500-eps8-stock-insert-diverse-v1-6-cycle5-linear",
]

ATTACK_SHORT_NAMES = {
}


def get_model_short_name(model_path):
    """Get the short name for a given model path."""
    return MODEL_SHORT_NAMES.get(model_path, "unknown")

