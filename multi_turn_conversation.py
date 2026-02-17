#!/usr/bin/env python3
"""
Multi-turn conversation script for vision-language models.
Supports both text and image inputs in a conversational format.
"""

import argparse
import torch
from utils.processor import load_model_and_processor
import os
from datetime import datetime
from utils.general import str2bool, set_seeds
import json
from typing import List, Dict, Any, Optional, Union
import time
from PIL import Image



class MultiTurnConversation:
    def __init__(self, model_path: str, enable_flash_attn: bool, 
                 max_tokens: int, temperature: float, 
                 top_p: float, do_sample: bool, system_message: str = None, seed: int = 0):
        """
        Initialize the multi-turn conversation system.
        
        Args:
            model_path: Path to the model
            enable_flash_attn: Whether to enable flash attention
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            system_message: System message to add to the conversation history
            seed: Random seed
        """
        self.model_path = model_path
        self.enable_flash_attn = enable_flash_attn
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.system_message = system_message

        # Set seeds
        set_seeds(seed)
        
        # Load model and processor
        self.model, self.processor = load_model_and_processor(model_path, enable_flash_attn, do_normalize=True)
        
        # Initialize conversation history
        self.conversation_history = []
        # add system message, depending on the model
        self.add_system_message()
        
        print(f"Model loaded successfully: {model_path}")
        print(f"Device: {self.model.device}")
    
    def add_system_message(self):
        """Add a system message to the conversation history."""
        if self.system_message is not None:
            system_message = [{"type": "text", "text": self.system_message}]
            self.conversation_history = [{"role": "system", "content": system_message}] + self.conversation_history
    
    def add_message(self, role: str, content: Union[List[Dict[str, Any]], str]):
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: For 'user' role: List of content items (text, image, etc.)
                    For 'assistant' role: Single string content
        """
        # For assistant messages, content should be a single string
        if role == "assistant":
            message = {
                "role": role,
                "content": content,
            }
        else:
            # For user messages, content should be a list
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
        
        self.conversation_history.append(message)
    
    def add_text_message(self, role: str, text: str):
        """Add a text-only message."""
        if role == "assistant":
            content = text
        else:
            content = [{"type": "text", "text": text}]
        self.add_message(role, content)
    
    def add_image_message(self, role: str, image_path: str, text: str = ""):
        """Add a message with image and optional text."""
        assert role == "user"
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        if text:
            content.append({"type": "text", "text": text})
        
        if content:
            self.add_message(role, content)
    
    def get_response(self) -> str:
        """
        Get response from the model based on current conversation history.
            
        Returns:
            Generated response text
        """
        if not self.conversation_history:
            return "No conversation history to respond to."

        # Prepare messages for the model
        messages = self.conversation_history.copy()
        
        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information
        image_path = messages[0]["content"][0]["image"]  # NOTE: assumes only one image, which is in the first message
        image_inputs = [Image.open(image_path).convert("RGB")]
        video_inputs = None
        
        # Prepare inputs for generation
        # Only include images/videos if they exist
        processor_kwargs = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            
        prompt_inputs = self.processor(**processor_kwargs)
        

        
        prompt_inputs = prompt_inputs.to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            # Prepare generation kwargs based on available inputs
            generation_kwargs = {
                "input_ids": prompt_inputs.input_ids,
                "attention_mask": prompt_inputs.attention_mask,
                "do_sample": self.do_sample,
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            # Add vision inputs if available (model-specific)
            if "pixel_values" in prompt_inputs and prompt_inputs["pixel_values"] is not None:
                generation_kwargs["pixel_values"] = prompt_inputs["pixel_values"].to(self.model.device)
            
            if "image_grid_thw" in prompt_inputs and prompt_inputs["image_grid_thw"] is not None:
                generation_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"].to(self.model.device)
        
            
            generated_ids = self.model.generate(**generation_kwargs)
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[prompt_inputs.input_ids.shape[1]:] for out_ids in generated_ids
        ]
        response_text = self.processor.tokenizer.decode(
            generated_ids_trimmed[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        
        return response_text
    
    def chat(self, attack_type: str = "none", target_string: Optional[str] = None,
             epsilon: float = 32, iterations: int = 100, alpha_apgd: float = 2.0):
        """
        Interactive chat loop.
        
        Args:
            attack_type: Type of adversarial attack
            target_string: Target output string for attacks
            epsilon: Epsilon value for attacks
            iterations: Number of attack iterations
            alpha_apgd: Alpha value for APGD attack
        """
        print("\n=== Multi-turn Conversation Started ===")
        print("Commands:")
        print("  /image <path> [text] - Add image with optional text")
        print("  /text <message> - Add text message")
        print("  /attack <type> [epsilon] [iterations] - Set attack parameters")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /quit - Exit conversation")
        print("  /help - Show this help")
        print("=" * 40)
        
        current_attack_type = attack_type
        current_target_string = target_string
        current_epsilon = epsilon
        current_iterations = iterations
        current_alpha_apgd = alpha_apgd
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("Goodbye!")
                        break
                    elif user_input == '/help':
                        print("Commands:")
                        print("  /image <path> [text] - Add image with optional text")
                        print("  /text <message> - Add text message")
                        print("  /attack <type> [epsilon] [iterations] - Set attack parameters")
                        print("  /history - Show conversation history")
                        print("  /clear - Clear conversation history")
                        print("  /quit - Exit conversation")
                        print("  /help - Show this help")
                        continue
                    elif user_input == '/clear':
                        self.conversation_history.clear()
                        print("Conversation history cleared.")
                        continue
                    elif user_input == '/history':
                        self._print_history()
                        continue
                    elif user_input.startswith('/image '):
                        parts = user_input[7:].split(' ', 1)
                        image_path = parts[0]
                        text = parts[1] if len(parts) > 1 else ""
                        if os.path.exists(image_path):
                            self.add_image_message("user", image_path, text)
                            print(f"Image added: {image_path}")
                            if text:
                                print(f"With text: {text}")
                        else:
                            print(f"Image file not found: {image_path}")
                    elif user_input.startswith('/text '):
                        text = user_input[6:]
                        self.add_text_message("user", text)
                        print(f"Text message added: {text}")
                    elif user_input.startswith('/attack '):
                        raise NotImplementedError()
                        parts = user_input[8:].split()
                        if len(parts) >= 1:
                            current_attack_type = parts[0]
                            if len(parts) >= 2:
                                current_epsilon = float(parts[1])
                            if len(parts) >= 3:
                                current_iterations = int(parts[2])
                            print(f"Attack settings: type={current_attack_type}, epsilon={current_epsilon}, iterations={current_iterations}")
                        continue
                    else:
                        print("Unknown command. Type /help for available commands.")
                        continue
                else:
                    # Regular user input
                    self.add_text_message("user", user_input)
                
                # Get model response
                print("\nGenerating response...")
                start_time = time.time()
                
                response = self.get_response()
                
                generation_time = time.time() - start_time
                
                # Add response to history
                self.add_message("assistant", response)
                
                # Print response
                print(f"\nAssistant ({generation_time:.2f}s): {response}")
                
            except KeyboardInterrupt:
                # quit the conversation
                print("\n\nGoodbye!")
                break
            # except Exception as e:
            #     print(f"\nError: {e}")
            #     print("Please try again or type /help for commands.")
    
    def _print_history(self):
        """Print conversation history."""
        if not self.conversation_history:
            print("No conversation history.")
            return
        
        print("\n=== Conversation History Raw ===")
        print(self.conversation_history)
        print("\n=== Conversation History Formatted ===")
        for i, message in enumerate(self.conversation_history):
            role = message["role"].capitalize()
            timestamp = message["timestamp"]
            content = message["content"]
            
            print(f"\n{i+1}. {role} ({timestamp}):")
            if role.lower() == "assistant":
                # Assistant messages have string content
                print(f"   Text: {content}")
            else:
                # User messages have list content
                for item in content:
                    if item["type"] == "text":
                        print(f"   Text: {item['text']}")
                    elif item["type"] == "image":
                        print(f"   Image: {item['image']}")
        print("=" * 40)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        print(f"Conversation saved to: {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"Conversation loaded from: {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except json.JSONDecodeError:
            print(f"Invalid JSON file: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversation with vision-language models.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct", 
                       help="Path to the model.")
    parser.add_argument("--enable_flash_attn", type=str2bool, default=True, 
                       help="Enable flash-attention for better acceleration and memory saving.")
    parser.add_argument("--max_tokens", type=int, default=128, 
                       help="Max tokens of model generation")
    parser.add_argument("--temperature", type=float, default=0.6, 
                       help="Temperature of generate")
    parser.add_argument("--top_p", type=float, default=0.95, 
                       help="top_p of generate")
    parser.add_argument("--do_sample", type=str2bool, default=True, 
                       help="Do sample for the model.")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Seed for the model.")
    parser.add_argument("--attack_type", type=str, choices=["none", "pgd", "apgd"], 
                       default="none", help="Type of adversarial attack to use.")
    parser.add_argument("--epsilon", type=float, default=32, 
                       help="Epsilon value for adversarial attacks.")
    parser.add_argument("--iterations", type=int, default=100, 
                       help="Number of iterations for adversarial attacks.")
    parser.add_argument("--alpha_apgd", type=float, default=2.0, 
                       help="Alpha value for APGD attack.")
    parser.add_argument("--target_string", type=str, default=None, 
                       help="Target output string for adversarial attack.")
    parser.add_argument("--save_path", type=str, default=None, 
                       help="Path to save conversation history.")
    parser.add_argument("--load_path", type=str, default=None, 
                       help="Path to load conversation history from.")
    
    args = parser.parse_args()
    
    try:
        # Initialize conversation system
        conversation = MultiTurnConversation(
            model_path=args.model_path,
            enable_flash_attn=args.enable_flash_attn,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            seed=args.seed
        )
        
        # Load previous conversation if specified
        if args.load_path:
            conversation.load_conversation(args.load_path)
        
        # Start interactive chat
        conversation.chat(
            attack_type=args.attack_type,
            target_string=args.target_string,
            epsilon=args.epsilon,
            iterations=args.iterations,
            alpha_apgd=args.alpha_apgd
        )
        
        # Save conversation if specified
        if args.save_path:
            conversation.save_conversation(args.save_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # exit(main())
    main()