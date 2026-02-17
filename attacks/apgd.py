import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from attacks.utils_attacks import L2_norm, check_zero_gradients
from utils.convert_to_pil import convert_adv_tensor_to_pil_image
from utils.general import find_subtensor_indices


class ForwardPass:
    """Encapsulates the forward pass for APGD attack optimization.
    """
    
    def __init__(self, model, processor, normalizer, 
                 messages_list, image_inputs, video_inputs, target_outputs,
                 full_prompt_inputs, target_positions, 
                 channels, temporal_patch_size, patch_size, 
                 device, cycle_context_frequency, cycle_mode, logger, verbose=False, use_all_contexts_at_once=False,
                 generation_params=None):
        """Initialize predictor with required components.
        
        Args:
            model: The vision-language model
            processor: Tokenizer/processor for reprocessing messages
            normalizer: Image normalization function
            messages_list: List of conversation histories with varying context lengths
            image_inputs: Vision inputs for reprocessing
            video_inputs: Video inputs for reprocessing
            target_outputs: List of target strings to locate in new contexts
            full_prompt_inputs: Dict containing input_ids, attention_mask, image_grid_thw
            target_positions: List of (start_idx, end_idx) tuples for target tokens
            channels: Number of image channels (3)
            temporal_patch_size: Temporal dimension of patches
            patch_size: Spatial dimension of patches
            device: CUDA device
            cycle_context_frequency: How often to cycle (in optimization steps)
            cycle_mode: Mode of context cycling ("linear" or "circular")
            verbose: Whether to print verbose output (optional, default False)
            use_all_contexts_at_once: Whether to compute loss on all contexts simultaneously (default False)
            generation_params: Dict containing generation parameters for update_context:
                - prepend_mode: Mode of prepending messages ("prepend" or "insert")
                - prepend_conversation_steps: Number of prepended conversation turns
                - temperature: Temperature for generation
                - top_p: Top-p for generation
                - do_sample: Whether to sample during generation
                - max_tokens: Maximum tokens to generate
                - max_message_length: Maximum length of a message (for truncation)
        """
        self.model = model
        self.model_path = model.name_or_path
        self.processor = processor
        self.normalizer = normalizer
        
        # Input data
        self.messages_list = messages_list
        self.image_inputs = image_inputs
        self.video_inputs = video_inputs
        self.target_outputs = target_outputs
        
        # Processed inputs
        self.full_prompt_inputs = full_prompt_inputs
        self.target_positions = target_positions
        
        # Model configuration
        self.channels = channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.device = device
        
        # Behavior control
        self.cycle_context_frequency = cycle_context_frequency
        self.logger = logger
        self.verbose = verbose
        self.current_context_index = 0
        self.use_all_contexts_at_once = use_all_contexts_at_once
        self.cycle_mode = cycle_mode
        self._cycle_direction = 1
        
        # Generation parameters for update_context (with defaults)
        self.generation_params = generation_params
        
        # Pre-process all contexts once and store on CPU or GPU depending on mode
        self.preprocessed_contexts = []
        if messages_list is not None and len(messages_list) > 1:
            if self.verbose:
                print(f"[Init] Pre-processing {len(messages_list)} contexts...")
            
            for idx, messages in enumerate(messages_list):
                # Apply chat template and process
                conversation_string = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_inputs = self.processor(
                    text=[conversation_string],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # Find target positions for this context
                context_target_positions = []
                for target_output in target_outputs:
                    target_tokens = self.processor.tokenizer(
                        target_output,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.to(device)
                    
                    # Find the position of this target in the sequence
                    target_pos = find_subtensor_indices(prompt_inputs["input_ids"][0], target_tokens[0])
                    if target_pos is not None:
                        context_target_positions.append(target_pos)
                    else:
                        raise ValueError(f"Target token '{target_output}' not found in context {idx}")
                
                context_data = {
                        'input_ids': prompt_inputs['input_ids'],
                        'attention_mask': prompt_inputs['attention_mask'],
                        'image_grid_thw': prompt_inputs.get('image_grid_thw'),
                        'target_positions': context_target_positions,
                    }

                self.preprocessed_contexts.append(context_data)
                
                if self.verbose:
                    print(f"[Init] Context {idx}: {len(messages)} messages, "
                          f"{prompt_inputs['input_ids'].shape[1]} tokens, "
                          f"{len(context_target_positions)} targets")
            
            if self.verbose:
                storage_location = "GPU" if use_all_contexts_at_once else "CPU"
                print(f"[Init] All contexts pre-processed and stored on {storage_location}")

    def prepare_inputs(self, image_preprocessed: torch.Tensor) -> torch.Tensor:
        if self.model_path.startswith("Qwen") or (self.model_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"):
            return self.expand_temporal_dimension(image_preprocessed)
        else:
            raise ValueError(f"Unknown model path: {self.model_path}")

    def expand_temporal_dimension(self, image_preprocessed: torch.Tensor) -> torch.Tensor:
        if self.model_path.startswith("Qwen") or (self.model_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"):
            # Qwen needs at least 2 temporal patches
            # -> expand to 2 temporal patches
            B, Np, D = image_preprocessed.shape
            assert D == self.channels * self.patch_size * self.patch_size, (
                f"Unexpected image_preprocessed dim {D}; expected {self.channels * self.patch_size * self.patch_size}")
            image_preprocessed = image_preprocessed.view(B, Np, self.channels, 1, self.patch_size, self.patch_size).expand(
                -1, -1, -1, self.temporal_patch_size, -1, -1
            ).reshape(B, Np, self.channels * self.temporal_patch_size * self.patch_size * self.patch_size)
            return image_preprocessed
        else:
            raise ValueError(f"Unknown model path: {self.model_path}")

    
    def __call__(self, z: torch.Tensor, y: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass on reparameterized variable z.
        
        Args:
            z: Qwen: (B, n_patches, C * p * p) tensor
            y: (B, T) tensor of target token ids
            
        Returns:
            loss: (B,) tensor of loss values (negative cross-entropy for targeted attack)
        """        
        # Prepare inputs
        x = self.prepare_inputs(z)

        # Use the full conversation history and extract target positions
        outputs = self.model(
            input_ids=self.full_prompt_inputs["input_ids"],
            attention_mask=self.full_prompt_inputs["attention_mask"],
            pixel_values=self.normalizer(x),
            image_grid_thw=self.full_prompt_inputs.get("image_grid_thw"),
            return_dict=True,
        )
        
        # Extract logits for all target positions
        target_logits = []
        for target_pos in self.target_positions:
            start_idx, end_idx = target_pos[0], target_pos[1]
            target_logits.append(outputs.logits[:, start_idx-1:end_idx-1, :])
        
        # Concatenate all target logits
        logits_seq = torch.cat(target_logits, dim=1)
        if return_logits:
            return logits_seq
        
        # Compute loss (negative cross-entropy for targeted attack)
        b, t, v = logits_seq.shape
        loss = -F.cross_entropy(
            logits_seq.flatten(0, 1),  # (B*T, V)
            y.flatten(0, 1),            # (B*T)
            reduction='none'
        )
        loss = loss.view(b, t).mean(1)  # (B,)
        
        # Clean up to prevent memory leaks
        del outputs, target_logits
        
        return loss


    # ---------------------------------------------------------------
    # Context callback function
    # ---------------------------------------------------------------
    def cycle_context(self, z_adv: torch.Tensor):
        """Cycle to the next conversation context.
        
        Updates full_prompt_inputs and target_positions by cycling through
        messages_list to use different conversation histories.
        
        Args:
            z_adv: Current adversarial perturbation in compressed format (B, Np, C*p*p)
        """
        if self.messages_list is None or len(self.messages_list) <= 1:
            # No context cycling needed if only one context
            return
        
        if self.cycle_mode == "linear":
            # Cycle to next context
            self.current_context_index = (self.current_context_index + 1) % len(self.messages_list)
        elif self.cycle_mode == "circular":
            # alternative: cycle through context back and forth, i.e. 0 -> 1 -> ... -> n -> n-1 -> ... -> 0
            self.current_context_index += self._cycle_direction
            if self.current_context_index == len(self.messages_list) - 1:
                self._cycle_direction = -1
            elif self.current_context_index == 0:
                self._cycle_direction = 1
        else:
            raise ValueError(f"Invalid cycle mode: {self.cycle_mode}")


        if self.verbose:
            print(f"[Context Update] Cycling to context {self.current_context_index}/{len(self.messages_list)-1} ({self.cycle_mode})")
        
        # Convert z_adv to full temporal patch tensor
        z_adv_full = self.prepare_inputs(z_adv)
        
        # Load pre-processed context from CPU
        next_context = self.preprocessed_contexts[self.current_context_index]
        
        
        # Update full_prompt_inputs with pre-processed text data and new pixel_values
        self.full_prompt_inputs = {
            'input_ids': next_context['input_ids'].to(self.device),
            'attention_mask': next_context['attention_mask'].to(self.device),
            'pixel_values': z_adv_full,
            'image_grid_thw': next_context.get('image_grid_thw'),
        }
        self.target_positions = next_context['target_positions']
        
        if self.verbose:
            current_messages = self.messages_list[self.current_context_index]
            print(f"[Context Update] Updated to {len(current_messages)} messages, "
                  f"{self.full_prompt_inputs['input_ids'].shape[1]} tokens, "
                  f"{len(self.target_positions)} targets")


    def update_context(self, z_adv: torch.Tensor):
        """Update prepended conversation responses based on current adversarial image.
        
        This method regenerates the assistant responses in the prepended conversation
        based on the current perturbed image, keeping the attack context aligned with
        what the model would actually output.
        
        Args:
            z_adv: Current adversarial perturbation in compressed format (B, Np, C*p*p)
        """
        start_time = time.time()
        prepend_conversation_steps = self.generation_params["prepend_conversation_steps"]
        prepend_mode = self.generation_params["prepend_mode"]
        temperature = self.generation_params["temperature"]
        top_p = self.generation_params["top_p"]
        do_sample = self.generation_params["do_sample"]
        max_tokens = self.generation_params["max_tokens"]
        max_message_length = self.generation_params["max_message_length"]


        if self.messages_list is None or len(self.messages_list) == 0:
            raise ValueError("messages_list is None or empty")

        if prepend_mode == "prepend":
            raise ValueError("prepend mode is not supported for update_context")
        
        if prepend_conversation_steps == 0:
            if self.verbose:
                print("[update_context] No prepended conversation steps, skipping update")
            return
        
        if self.verbose:
            print(f"[update_context] Regenerating {prepend_conversation_steps} prepended responses...")
        
        # Convert z_adv to full temporal patch tensor
        z_adv_full = self.prepare_inputs(z_adv)
        
        # Convert adversarial tensor to PIL image for generation
        adv_pil_image = convert_adv_tensor_to_pil_image(
            z_adv_full, self.full_prompt_inputs, self.processor, epsilon=self.epsilon, verbose=False
        )
        adv_image_inputs = [adv_pil_image]
        
        # Determine which assistant message indices are prepended based on mode
        # prepend mode: prepend_messages come BEFORE the image message
        #   indices: 1, 3, 5, ..., 2*prepend_conversation_steps - 1
        # insert mode: prepend_messages come AFTER the first user+assistant pair
        #   indices: 3, 5, 7, ..., 2 + 2*prepend_conversation_steps - 1
        if prepend_mode == "prepend":
            prepend_assistant_indices = list(range(1, 2 * prepend_conversation_steps, 2))
        elif prepend_mode == "insert":
            prepend_assistant_indices = list(range(3, 2 + 2 * prepend_conversation_steps, 2))
        else:
            raise ValueError(f"Invalid prepend mode: {prepend_mode}")
        
        # Use the full messages (first element of messages_list has all context)
        # We need to work with a copy to avoid modifying during iteration
        messages = copy.deepcopy(self.messages_list[-1])  # Use the one with most context
        
        # Generate new responses for each prepended assistant turn
        with torch.inference_mode():
            for assistant_idx in prepend_assistant_indices:
                # Build conversation up to the assistant turn (include only messages before it)
                conversation_up_to = messages[:assistant_idx]
                prompt_text = self.processor.apply_chat_template(
                    conversation_up_to, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=adv_image_inputs,
                    videos=self.video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                
                # Generate with adversarial image
                generated_ids = self.model.generate(
                    input_ids=prompt_inputs.input_ids,
                    pixel_values=self.normalizer(prompt_inputs["pixel_values"].to(self.device)),
                    image_grid_thw=prompt_inputs["image_grid_thw"].to(self.device),
                    attention_mask=prompt_inputs.attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                
                # Decode the generated response
                generated_ids_trimmed = generated_ids[0, prompt_inputs.input_ids.shape[1]:]
                new_response = self.processor.tokenizer.decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                # Truncate if max_message_length is specified
                if max_message_length is not None:
                    new_response = new_response[:max_message_length]
                
                # Update the message content
                messages[assistant_idx]["content"] = new_response
                
                if self.verbose:
                    print(f"[update_context] Regenerated response at index {assistant_idx}: {new_response[:50]}...")
                
                # Clean up
                del generated_ids, prompt_inputs
        
        # Update all message lists with new responses
        # Each context i has i prepended conversation steps, so we need to compute
        # the correct indices for each context
        for context_i, msg_list in enumerate(self.messages_list):
            # For insert mode, context i has i prepended conversations
            # The prepended assistant indices for context i are: [3, 5, ..., 2+2*i-1] = [3, 5, ..., 2*i+1]
            if context_i == 0:
                # Context 0 has no prepended messages
                continue
            context_prepend_indices = list(range(3, 2 + 2 * context_i, 2))
            for assistant_idx in context_prepend_indices:
                if assistant_idx < len(msg_list) and assistant_idx in prepend_assistant_indices:
                    msg_list[assistant_idx]["content"] = messages[assistant_idx]["content"]
        
        # Re-preprocess all contexts with updated messages
        self.preprocessed_contexts = []
        for idx, msgs in enumerate(self.messages_list):
            conversation_string = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_inputs = self.processor(
                text=[conversation_string],
                images=adv_image_inputs,
                videos=self.video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Find target positions for this context
            context_target_positions = []
            for target_output in self.target_outputs:
                target_tokens = self.processor.tokenizer(
                    target_output,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(self.device)
                
                target_pos = find_subtensor_indices(prompt_inputs["input_ids"][0], target_tokens[0])
                if target_pos is not None:
                    context_target_positions.append(target_pos)
                else:
                    raise ValueError(f"Target token '{target_output}' not found in context {idx} after update")
            
            context_data = {
                'input_ids': prompt_inputs['input_ids'],
                'attention_mask': prompt_inputs['attention_mask'],
                'image_grid_thw': prompt_inputs['image_grid_thw'],
                'target_positions': context_target_positions,
            }
            self.preprocessed_contexts.append(context_data)
            
            if self.verbose:
                print(f"[update_context] New conversation string: {conversation_string}")
                print(f"[update_context] Re-processed context {idx}: {len(msgs)} messages, "
                      f"{prompt_inputs['input_ids'].shape[1]} tokens")
        
        # Update current full_prompt_inputs and target_positions from the current context
        current_context = self.preprocessed_contexts[self.current_context_index]
        self.full_prompt_inputs = {
            'input_ids': current_context['input_ids'],
            'attention_mask': current_context['attention_mask'],
            'pixel_values': z_adv_full,
            'image_grid_thw': current_context['image_grid_thw'],
        }
        self.target_positions = current_context['target_positions']
        
        # Update image_inputs to use the adversarial image
        self.image_inputs = adv_image_inputs
        
        if self.verbose:
            print(f"[update_context] Context update complete. Current context: {self.current_context_index}")
            print(f"[update_context] Time taken: {time.time() - start_time:.2f} seconds")
            
        torch.cuda.empty_cache()


    def cleanup(self):
        """Explicitly free GPU memory held by this predictor."""
        if hasattr(self, 'full_prompt_inputs'):
            del self.full_prompt_inputs
        torch.cuda.empty_cache()


def apgd_attack_on_image_input(
    *,
    processor,
    model,
    messages_list: list,  # List of conversation histories
    image_inputs: list,
    video_inputs: list,
    normalizer,
    target_outputs: list[str],  # List of target outputs
    epsilon: float = 32 / 255,
    num_steps: int = 1000,
    alpha: float = 2.0,
    use_best_loss: bool = True,
    verbose: bool = True,
    context_update_frequency: int = 50,
    cycle_context_frequency: int = 0,
    cycle_mode: str = "linear",
    use_all_contexts_at_once: bool = False,
    logger=None,
    generation_params: dict = None,  # Generation parameters for update_context
):
    """Run Auto-PGD (APGDAttack) on the *first* image so that the model's
    next token(s) match all targets in ``target_outputs``.
    
    The attack performs joint optimization across all provided targets.
    """

    # Nothing to do if there are no targets or no image input
    if not target_outputs or len(image_inputs) == 0:
        return image_inputs
    assert len(image_inputs) == 1, "Only one image input is supported for APGD attack"
    assert not (use_all_contexts_at_once and cycle_context_frequency > 0), "use_all_contexts_at_once and cycle_context_frequency cannot be both True"

    device = model.device
    messages = messages_list[0]

    # ---------------------------------------------------------------
    # Build prompt + target sequence for all targets
    # ---------------------------------------------------------------
    # Tokenize the full conversation history
    conversation_string = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)  # do not add generation prompt, as target outputs are already appended
    full_prompt_inputs = processor(
        text=[conversation_string],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # Tokenize each target separately and find their positions in the full sequence
    target_token_ids_list = []
    target_positions = []
    
    for target_output in target_outputs:
        target_tokens = processor.tokenizer(
            target_output,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        target_token_ids_list.append(target_tokens)
        
        # Find the position of this target in the full sequence
        target_pos = find_subtensor_indices(full_prompt_inputs["input_ids"][0], target_tokens[0])
        if target_pos is not None:
            target_positions.append(target_pos)
        else:
            raise ValueError(f"Target token {target_output} not found in the full prompt")
    
    # Combine all target sequences into one for joint optimization
    combined_target_token_ids = torch.cat(target_token_ids_list, dim=1)
    
    # Tensor that will be perturbed (processor patch format)
    full_patch_tensor = (
        full_prompt_inputs["pixel_values"].clone().detach().to(device)
    )  # Qwen: (n_patches, C * t * p * p)
    if full_patch_tensor.shape[0] != 1:  # add batch dimension if not present
        full_patch_tensor = full_patch_tensor.unsqueeze(0)

    # input preparation    
    if model.name_or_path.startswith("Qwen") or (model.name_or_path == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"):
        # Reparameterize: optimize a single temporal slice z, expand to t in forward
        channels = 3
        patch_size = int(getattr(processor.image_processor, "patch_size", 14))
        temporal_patch_size = int(getattr(processor.image_processor, "temporal_patch_size", 2))
        B, Np, D = full_patch_tensor.shape
        assert D == channels * temporal_patch_size * patch_size * patch_size, (
            f"Unexpected patch dim {D}; expected {channels * temporal_patch_size * patch_size * patch_size}")
        # Compress to z0 (single temporal slice) for efficiency
        z0 = full_patch_tensor.view(B, Np, channels, temporal_patch_size, patch_size, patch_size)
        z0 = z0.mean(dim=3).reshape(B, Np, channels * patch_size * patch_size)
    else:
        raise ValueError(f"Unknown model path: {model.name_or_path}")

    # ---------------------------------------------------------------
    # Build predictor class for z (encapsulates model forward pass)
    # ---------------------------------------------------------------
    predictor = ForwardPass(
        model=model,
        processor=processor,
        normalizer=normalizer,
        messages_list=messages_list,
        image_inputs=image_inputs,
        video_inputs=video_inputs,
        target_outputs=target_outputs,
        full_prompt_inputs=full_prompt_inputs,
        target_positions=target_positions,
        channels=channels,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size,
        device=device,
        cycle_context_frequency=cycle_context_frequency,
        cycle_mode=cycle_mode,
        logger=logger,
        verbose=verbose,
        use_all_contexts_at_once=use_all_contexts_at_once,
        generation_params=generation_params,
    )

    # ---------------------------------------------------------------
    # Forward pass once before the attack
    # ---------------------------------------------------------------
    predictor.use_all_contexts_at_once = False  # just for the forward pass
    logits_seq = predictor(z0, y=combined_target_token_ids, return_logits=True)
    assert logits_seq.shape[0] == 1, "Only one batch is supported"
    assert logits_seq.shape[1] == combined_target_token_ids.shape[1], "Number of target positions must match"
    # print(f"logits_seq: {logits_seq}")
    # print(f"combined_target_token_ids: {combined_target_token_ids}")
    # print(f"logits_seq.shape: {logits_seq.shape}")
    # print(f"combined_target_token_ids.shape: {combined_target_token_ids.shape}")
    # print(f"decoded: {processor.tokenizer.decode(logits_seq[0].argmax(dim=-1), skip_special_tokens=False)}")
    print(f"image_inputs: {image_inputs}")
    print(f"z shape (bs, n_patches, patch_dim): {z0.shape}, logits_seq shape: {logits_seq.shape}")
    predictor.use_all_contexts_at_once = use_all_contexts_at_once  # restore the original value
    
    # ---------------------------------------------------------------
    # Instantiate and run APGDAttack with multiple target support
    # ---------------------------------------------------------------
    attacker = APGDAttack(
        predictor,
        n_iter=num_steps,
        eps=epsilon,
        norm="Linf",
        n_restarts=1,
        loss="computed-by-model",
        alpha=alpha,
        verbose=verbose,
        device=device,
        context_update_frequency=context_update_frequency,
        cycle_context_frequency=cycle_context_frequency,
        target_seq_ids=combined_target_token_ids,
    )

    # Run attack on z0
    # z_adv = attacker.perturb(z0, y=combined_target_token_ids, best_loss=True)
    attack_result = attacker.attack_single_run(z0, y=combined_target_token_ids)
    if use_best_loss:
        z_adv = attack_result["x_best"]
    else:
        z_adv = attack_result["x_final"]
    logger.write(f"[best loss] {attack_result['loss_best']}, [final loss] {attack_result['loss_final']}")

    if predictor.model_path.startswith("Qwen"):
        # Expand z_adv back to full temporal patch tensor for saving and downstream
        adv_tensor = predictor.expand_temporal_dimension(z_adv)
    else:
        adv_tensor = z_adv

    # ---------------------------------------------------------------
    # Convert adversarial tensor back to PIL and update image_inputs
    # ---------------------------------------------------------------
    # Note: ``adv_tensor`` is in *patch* format with shape (n_patches, patch_dim).
    # We need to invert the patchification performed by the image processor to
    # recover the full RGB image so that downstream calls expecting a PIL image
    # continue to work.
    adv_pil_image = convert_adv_tensor_to_pil_image(adv_tensor, full_prompt_inputs, processor, epsilon=epsilon, model_path=model.name_or_path)
    image_inputs[0] = adv_pil_image
    
    # Clean up attack artifacts to free GPU memory
    predictor.cleanup()  # Explicitly cleanup predictor's tensors
    del predictor, attacker, z0, z_adv, combined_target_token_ids, target_token_ids_list
    del full_prompt_inputs, adv_tensor
    torch.cuda.empty_cache()
    
    return image_inputs, attack_result['loss_steps']



class APGDAttack():
    """
    AutoPGD from https://github.com/fra31/auto-attack
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            alpha=2.0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None,
            context_update_frequency=0,
            cycle_context_frequency=0,
            target_seq_ids=None,
            ):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
        self.context_update_frequency = context_update_frequency
        self.cycle_context_frequency = cycle_context_frequency
        
        # Dual target support
        self.target_seq_ids = target_seq_ids


        assert self.norm in ['Linf', 'L2']
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    
    def attack_single_run(self, x, y, x_init=None):
        self.init_hyperparam(x)

        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)        
        
        
        if not x_init is None:
            x_adv = x_init.clone()
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)
        track_accuracy = True

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'ce-targeted-cfts':
            criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                reduction='none')
        elif self.loss == 'ce-seq-targeted':
            def _seq_ce(logits, y):
                # logits: (B, T, V) ; y: (B, T)
                b, t, v = logits.shape
                l = -F.cross_entropy(
                        logits.flatten(0, 1),  # (B·T, V)
                        y.flatten(0, 1),      # (B·T)
                        # ignore_index=ignore_index,
                        reduction='none'
                    )
                l = l.view(b, t).mean(1)
                return l
            criterion_indiv = _seq_ce
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        elif self.loss == 'dlr-targeted':
            criterion_indiv = self.dlr_loss_targeted
        elif self.loss == 'ce-targeted':
            criterion_indiv = self.ce_loss_targeted
        elif self.loss == 'computed-by-model':
            # the model returns the loss directly
            criterion_indiv = lambda x, y: x
            track_accuracy = False
        else:
            raise ValueError('unknowkn loss')
        
          
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                if self.model.use_all_contexts_at_once:
                    loss_indiv = 0.
                    for context_idx in range(len(self.model.preprocessed_contexts)):
                        logits = self.model(x_adv, y)
                        loss_indiv_cur = criterion_indiv(logits, y)
                        loss = loss_indiv_cur.sum()
                        grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                        self.model.cycle_context(x_adv)
                        loss_indiv += loss_indiv_cur.detach()
                    loss_indiv /= len(self.model.preprocessed_contexts)
                    grad /= len(self.model.preprocessed_contexts)
                else:
                    logits = self.model(x_adv, y)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            
            logits = logits.detach()
            loss_indiv = loss_indiv.detach()
            del loss

        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        if track_accuracy:
            if self.loss != 'ce-seq-targeted':
                acc = logits.max(1)[1] == y
            else:
                acc = (logits.argmax(-1) != y).any(-1)
            acc_steps[0] = acc + 0

        loss_best = loss_indiv.clone()
        del logits, loss_indiv

        alpha = self.alpha
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    if self.model.use_all_contexts_at_once:
                        loss_indiv = 0.
                        for context_idx in range(len(self.model.preprocessed_contexts)):
                            logits = self.model(x_adv, y)
                            loss_indiv_cur = criterion_indiv(logits, y)
                            loss = loss_indiv_cur.sum()
                            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                            self.model.cycle_context(x_adv)
                            loss_indiv += loss_indiv_cur.detach()
                        loss_indiv /= len(self.model.preprocessed_contexts)
                        grad /= len(self.model.preprocessed_contexts)
                    else:
                        logits = self.model(x_adv, y)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
                        grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                
                logits = logits.detach()
                loss_indiv = loss_indiv.detach()
                del loss
            
            grad /= float(self.eot_iter)

            if track_accuracy:
                if self.loss != 'ce-seq-targeted':
                    pred = logits.max(1)[1] != y
                else:
                    pred = (logits.argmax(-1) != y).any(-1)
                acc = torch.min(acc, pred)
                acc_steps[i + 1] = acc + 0
                ind_pred = (pred == 0).nonzero().squeeze()
                x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            else:
                acc = None
            
            # Context cycling - cycle through different conversation contexts
            if self.cycle_context_frequency > 0 and (i % self.cycle_context_frequency == 0) and i > 0:
                assert not self.model.use_all_contexts_at_once
                if self.verbose:
                    print(f"[Iteration {i}] Cycling conversation context")
                self.model.cycle_context(x_adv)
            
            # Context update 
            if self.context_update_frequency > 0 and(i % self.context_update_frequency == 0) and i > 0:
                if self.verbose:
                    print(f"Calling context update callback at iteration {i}")
                # Call the context update callback with current adversarial example
                self.model.update_context(x_adv)
                if self.verbose:
                    print(f"Context updated successfully at iteration {i}")

            
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print(
                    f'iteration: {i} - best loss: {loss_best.sum()} - cur loss: {loss_indiv.sum()}' 
                    # f'- robust accuracy: {acc.float().mean()}{str_stats} - output: {logits.argmax(-1).cpu().tolist()}'
                )
            
            # Periodic memory cleanup
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        
        torch.cuda.empty_cache()
        
        # return (x_best, acc, loss_best, x_best_adv)
        return {
            "x_best": x_best,  # perturbed images that achieved the best loss
            "acc": acc,  # robust accuracy
            "loss_best": loss_best,  # best loss
            "x_best_adv": x_best_adv,  # perturbed images that cause misclassification / align with target. if not found, returns the initial perturbed image
            "x_final": x_adv,  # final perturbed image
            "loss_final": loss_indiv,  # final loss
            "loss_steps": -loss_steps.cpu().numpy(),  # loss at each iteration
        }

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """
        raise NotImplementedError("Not adapted to dict output format of attack_single_run")
        assert self.loss in ['ce', 'dlr', 'ce-seq-targeted'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x)
            if not self.loss == 'ce-seq-targeted':
                y_pred = y_pred.max(1)[1]
            else:
                y_pred = y_pred.argmax(-1)
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss == 'ce-targeted':
            acc = y_pred != y
        elif self.loss == 'ce-seq-targeted':
            acc = (y_pred != y).any(-1)  # all tokens must match
        else:
            acc = y_pred == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            torch.cuda.empty_cache()
            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            torch.cuda.empty_cache()
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)