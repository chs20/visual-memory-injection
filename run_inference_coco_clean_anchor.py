import json
import os
from datetime import datetime
from multi_turn_conversation import MultiTurnConversation
from utils.general import set_seeds

"""
Run inference on clean images, save the output so it can be used as the anchor for the attack.
"""



if __name__ == "__main__":
    # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "Qwen/Qwen3-VL-8B-Instruct"
    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    enable_flash_attn = True
    max_tokens = 512
    do_sample = False
    temperature = None
    top_p = None
    seed = 0

    clean_images_dir = "./clean_images/coco20"
    image_paths = [os.path.join(clean_images_dir, el) for el in os.listdir(clean_images_dir) if el.endswith(".jpg") or el.endswith(".png")]

    prompt = "Provide a short caption for this image"  # TODO

    set_seeds(seed)

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
        "metadata": {
            "model_path": model_path,
            "prompt": prompt,
            "enable_flash_attn": enable_flash_attn,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        },
        "outputs": {}
    }
    for i, image_path in enumerate(image_paths):
        print(f"######## Processing image: {image_path} ({i+1}/{len(image_paths)})")
        model.add_message("user", [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}])
        response = model.get_response()
        print(f"Response: {response}\n\n")
        results["outputs"][image_path.split('/')[-1]] = response
        model.clear_history()

    # save results to json in coco20/clean_outputs_{model_path}.json
    file_name = f"clean_outputs_{model_path.split('/')[-1]}.json"
    with open(os.path.join(clean_images_dir, file_name), "w") as f:
        json.dump(results, f, indent=2)