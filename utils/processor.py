import torch
from PIL import Image
from transformers import AutoProcessor
import torchvision
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoModelForCausalLM

class ImageProcessorResize:
        def __init__(self, image_processor, resizer):
            self.image_processor = image_processor
            self.resizer = resizer
        def __call__(self, images, **kwargs):
            images = [self.resizer(image) for image in images]
            return self.image_processor(images=images, **kwargs)
        def __getattr__(self, name):
            # Avoid infinite recursion when accessing our own attributes during deepcopy/pickle
            if name in ("image_processor", "resizer"):
                raise AttributeError(f"'{type(self).__name__}' object does not allow access to attribute '{name}'")
            return getattr(self.image_processor, name)
        def __setattr__(self, name, value):
            if name in ["image_processor", "resizer"]:
                super().__setattr__(name, value)
            else:
                setattr(self.image_processor, name, value)
        def __delattr__(self, name):
            if name in ["image_processor", "resizer"]:
                super().__delattr__(name)
            else:
                delattr(self.image_processor, name)
        

def load_processor(model_path, use_fast, do_normalize, device):
    model_path_to_processor_func = {
        "Qwen/Qwen2.5-VL-7B-Instruct": load_processor_qwen,
        "Qwen/Qwen3-VL-8B-Instruct": load_processor_qwen,
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": load_processor_qwen,
        "aisingapore/Qwen-SEA-LION-v4-8B-VL": load_processor_qwen,
        "ddvd233/QoQ-Med3-VL-8B": load_processor_qwen,
    }
    load_processor_func = model_path_to_processor_func[model_path]
    return load_processor_func(model_path, use_fast, do_normalize, device)


def load_processor_qwen(model_path, use_fast, do_normalize, device):
    processor = AutoProcessor.from_pretrained(model_path, use_fast=use_fast)
    # set resizing to 224x224
    # processor.image_processor.size = {"shortest_edge": 224, "longest_edge": 224}  # this seems to have no effect with current version of transformers
    processor.image_processor.min_pixels = 224 * 224  #  this should ensure the smart resize does not resize after our resize
    processor.image_processor.max_pixels = 224 * 224
    # get our own resize function, as they do weird stuff ("smart resize")
    processor.do_resize = False
    resizer = torchvision.transforms.Resize((224, 224))
    # define our own image processor that resizes the images before passing them to the original image processor
    image_processor_custom = ImageProcessorResize(processor.image_processor, resizer)
    processor.image_processor = image_processor_custom

    if not do_normalize:
        # disable normalization return separate processor and normalizer
        processor.image_processor.do_normalize = False
        mean = processor.image_processor.image_mean
        std = processor.image_processor.image_std
        # define normalization transform, sanity check performed later
        normalizer = get_normalizer_qwen(mean=mean, std=std, processor=processor, device=device)
        return processor, normalizer
    else:
        # normalization is performed by the processor
        return processor


def get_normalizer_qwen(mean, std, processor, device):
    """Return a lambda that normalizes flattened patch tokens exactly the same
    way as ``Qwen2VLImageProcessor``.

    The processor produces flattened patches with channel-major ordering, i.e.
    all *R* pixel values for every (t, h, w) first, then all *G*, then *B*.
    This utility builds broadcastable mean / std tensors that follow the same
    layout so one can call ``normalizer(pixel_values)`` to match the internal
    preprocessing.
    """
    patch_size = processor.image_processor.patch_size
    temporal_patch_size = processor.image_processor.temporal_patch_size

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)

    elems_per_channel = temporal_patch_size * (patch_size**2)
    mean_flat = mean_t.repeat_interleave(elems_per_channel)
    std_flat = std_t.repeat_interleave(elems_per_channel)

    # Shape (1, 1, patch_dim) → broadcast over (B, N, D)
    mean_flat = mean_flat.view(1, 1, -1)
    std_flat = std_flat.view(1, 1, -1)

    return lambda x: (x - mean_flat) / std_flat


def sanity_check_normalizer(
    processor,
    normalizer,
    image_path: str,
    device,
    tol: float = 1e-5,
):
    """Print absolute diff stats between processor and custom normalizer."""

    prev_flag = processor.image_processor.do_normalize

    # Reference path: processor does its own normalisation
    processor.image_processor.do_normalize = True
    pv_ref = processor.image_processor(
        images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt",
    ).pixel_values.to(device)

    # Custom path: raw pixels + our normalizer
    processor.image_processor.do_normalize = False
    pv_raw = processor.image_processor(
        images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt",
    ).pixel_values.to(device)
    pv_custom = normalizer(pv_raw)

    # Restore flag for downstream usage
    processor.image_processor.do_normalize = prev_flag

    abs_diff = (pv_ref - pv_custom).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    median_diff = abs_diff.median().item()

    if max_diff > tol:
        print(
            f"[sanity_check_normalizer] WARNING: max |diff| = {max_diff:.6f} (> {tol})\n"
            f"[sanity_check_normalizer] mean |diff| = {mean_diff:.6f}, median |diff| = {median_diff:.6f}"
        )
    else:
        print(
            "[sanity_check_normalizer] Normalizer sanity check passed:\n"
            f"[sanity_check_normalizer] max |diff| = {max_diff:.6f}\n"
            f"[sanity_check_normalizer] mean |diff| = {mean_diff:.6f}, median |diff| = {median_diff:.6f}"
        )


def load_model_and_processor(model_path, enable_flash_attn, do_normalize, device="cuda"):
    """Load the model and processor based on model path."""
    if model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        use_fast = True
    elif model_path in [
        "Qwen/Qwen3-VL-8B-Instruct", 
        "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "ddvd233/QoQ-Med3-VL-8B",
        ]:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        use_fast = True
    elif model_path == "aisingapore/Qwen-SEA-LION-v4-8B-VL":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        use_fast = True
    elif model_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # _attn_implementation="sdpa",
        )
        use_fast = False
    else:
        raise ValueError(f"Model path {model_path} not supported.")
    
    processor = load_processor(model_path, use_fast=use_fast, do_normalize=do_normalize, device=model.device)
    return model, processor