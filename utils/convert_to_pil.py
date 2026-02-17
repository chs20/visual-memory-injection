import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def convert_adv_tensor_to_pil_image(adv_tensor: torch.Tensor, prompt_inputs: dict, processor, model_path, epsilon, verbose=False) -> Image.Image:
    """Convert adversarial tensor back to PIL image.
    
    Args:
        adv_tensor: Adversarial tensor 
        prompt_inputs: Dictionary containing processor outputs including image_grid_thw
        processor: The image processor used for preprocessing
        model_path: The path to the model
        epsilon: The epsilon value for the l-inf threat model, e.g. 8/255.
        verbose: Whether to print verbose output
    Returns:
        PIL.Image: The reconstructed adversarial image
    """
    if (model_path is not None and model_path.startswith("Qwen")) or (model_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"):
        adv_pil_image = convert_adv_tensor_to_pil_image_qwen(adv_tensor, prompt_inputs, processor, verbose)
    else:
        adv_pil_image = convert_adv_tensor_to_pil_image_qwen(adv_tensor, prompt_inputs, processor, verbose)

    # ------------------------------------------------------------------
    # Project back to threat model
    # ------------------------------------------------------------------
    # Converting from float [0,1] to uint8 PIL introduces quantisation
    # noise that can push individual pixels outside the L-inf ball.
    # We recover the clean PIL image from the original pixel_values,
    # then clamp the perturbation to [-eps_pixels, +eps_pixels].
    clean_tensor = prompt_inputs["pixel_values"]
    clean_pil_image = convert_adv_tensor_to_pil_image_qwen(clean_tensor, prompt_inputs, processor, verbose=False)
    adv_np = np.array(adv_pil_image, dtype=np.float32)
    clean_np = np.array(clean_pil_image, dtype=np.float32)
    eps_pixels = epsilon * 255.0
    perturbation = np.clip(adv_np - clean_np, -eps_pixels, eps_pixels)
    adv_projected = np.clip(clean_np + perturbation, 0, 255).astype(np.uint8)
    adv_pil_image = Image.fromarray(adv_projected)

    if verbose:
        linf = np.max(np.abs(np.array(adv_pil_image, dtype=np.float32) - clean_np))
        print(f"Threat model projection: L-inf = {linf:.1f}/255 (budget: {eps_pixels:.1f}/255)")

    # ------------------------------------------------------------------
    # Sanity check: PIL -> processor round-trip fidelity
    # ------------------------------------------------------------------ 
    reprocessed = processor.image_processor(images=[adv_pil_image], return_tensors="pt")["pixel_values"]
    reprocessed = reprocessed.to(dtype=adv_tensor.dtype, device=adv_tensor.device)
    assert reprocessed.squeeze().shape == adv_tensor.squeeze().shape, f"Shape mismatch: re_pix {reprocessed.shape} vs adv {adv_tensor.shape}"
    abs_diff = (reprocessed - adv_tensor).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    if epsilon is None:
        assert torch.allclose(reprocessed, adv_tensor, atol=5e-3), (
            f"Unpatchify/patchify mismatch (max abs diff={max_abs_diff:.4e}, mean abs diff={mean_abs_diff:.4e})"
        )
    if verbose:
        print(f"Unpatchify/patchify sanity check (max abs diff={max_abs_diff:.4e}, mean abs diff={mean_abs_diff:.4e})")
    
    return adv_pil_image



def convert_adv_tensor_to_pil_image_qwen(adv_tensor: torch.Tensor, prompt_inputs: dict, processor, verbose=False) -> Image.Image:
    """Convert adversarial tensor back to PIL image for Qwen.
    
    Args:
        adv_tensor: Adversarial tensor in patch format with shape (n_patches, patch_dim)
        prompt_inputs: Dictionary containing processor outputs including image_grid_thw
        processor: The image processor used for preprocessing
        verbose: Whether to print verbose output
    Returns:
        PIL.Image: The reconstructed adversarial image
    """
    with torch.no_grad():
        # ------------------------------------------------------------------
        # 1. Gather meta-information needed for the inverse transformation
        # ------------------------------------------------------------------
        grid_thw = prompt_inputs["image_grid_thw"]  # (1, 3) or (3,)
        if isinstance(grid_thw, torch.Tensor):
            grid_thw = grid_thw.squeeze(0).tolist()
        grid_t, grid_h, grid_w = map(int, grid_thw)

        # Fetch processor-specific patch hyper-parameters (fallback to sane defaults)
        patch_size = int(getattr(processor.image_processor, "patch_size", 14))
        temporal_patch_size = int(getattr(processor.image_processor, "temporal_patch_size", 1))
        merge_size = int(getattr(processor.image_processor, "merge_size", 1))
        channel = 3  # RGB only

        # Sanity check – make sure shapes are consistent
        flatten_patches = adv_tensor.squeeze(0).detach().cpu()  # (n_patches, patch_dim)
        assert flatten_patches.shape[0] == grid_t * grid_h * grid_w, (
            "Mismatch between #patches and grid dimensions: "
            f"{flatten_patches.shape[0]} vs {grid_t}×{grid_h}×{grid_w}"
        )

        # ------------------------------------------------------------------
        # 2. Undo the flattening/transpose done in the image processor
        # ------------------------------------------------------------------
        patches = flatten_patches.view(
            grid_t,
            grid_h // merge_size,
            grid_w // merge_size,
            merge_size,
            merge_size,
            channel,
            temporal_patch_size,
            patch_size,
            patch_size,
        )
        # Inverse of `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` used in `_preprocess`
        patches = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()

        # (grid_t * temporal_patch_size, C, H, W)
        images_tensor = patches.view(
            grid_t * temporal_patch_size,
            channel,
            grid_h * patch_size,
            grid_w * patch_size,
        )

        # We only deal with single-frame images, so take the first frame.
        # recon_img = images_tensor[0].clamp(0.0, 1.0)
        recon_img = images_tensor[0]
        assert recon_img.max() <= 1.0 and recon_img.min() >= 0.0, f"Recon image is not in [0, 1]: {recon_img.max()} {recon_img.min()}"

        # ------------------------------------------------------------------
        # 3. Convert to PIL.Image
        # ------------------------------------------------------------------
        adv_pil_image = transforms.ToPILImage()(recon_img)

        return adv_pil_image



