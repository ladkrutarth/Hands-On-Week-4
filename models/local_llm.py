import os
import anyio
from pathlib import Path
from mlx_lm import load, generate, sample_utils, stream_generate

class LocalLLM:
    """
    Wrapper for mlx-lm to provide a simple interface for local LLM generation.
    Highly optimized for Apple Silicon (M1/M2/M3 chips).

    Tuning for speed vs quality:
    - Default: Llama-3 8B Instruct 4bit (higher quality, slower).
    - Set VERISCAN_FAST_MODE=1 to use TinyLlama 1.1B Chat (much faster).
    - Or override explicitly with VERISCAN_LLM_MODEL.
    """
    
    # Default high-quality Llama-3 8B Instruct model for MLX
    MODEL_ID = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    FAST_MODEL_ID = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
    
    def __init__(self, model_id: str = None):
        env_model = os.environ.get("VERISCAN_LLM_MODEL")
        fast_mode = os.environ.get("VERISCAN_FAST_MODE", "").strip() == "1"
        if env_model:
            self.model_id = env_model.strip()
        elif fast_mode:
            self.model_id = self.FAST_MODEL_ID
        else:
            self.model_id = model_id or self.MODEL_ID

        print(f"Loading local MLX model: {self.model_id}...")
        
        # MLX-LM handles downloading and loading automatically
        self.model, self.tokenizer = load(self.model_id)
        print("✅ Local MLX LLM loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = 250, temp: float = 0.0) -> str:
        """Generate response from the local MLX model. Optimized for speed."""
        # Simple Instruct template for Llama-3
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # temp=0 is faster (greedy decoding) and more consistent for logic tasks
        sampler = sample_utils.make_sampler(temp=temp)
        
        response_text = ""
        # Manually stream to support stopping on special tokens
        for response in stream_generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            response_text += response.text
            if "<|eot_id|>" in response_text or "<|start_header_id|>" in response_text:
                break

        # Strip Llama-3 special tokens if they leak
        for token in ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "assistant", "user"]:
            if token in response_text:
                response_text = response_text.split(token)[0]

        response_text = response_text.strip()
        
        # Ensure response ends at the last complete sentence (period)
        if response_text and not response_text.endswith((".", "!", "?")):
            last_punct = max(response_text.rfind("."), response_text.rfind("!"), response_text.rfind("?"))
            if last_punct != -1:
                response_text = response_text[:last_punct + 1]

        return response_text

    async def generate_async(self, prompt: str, max_tokens: int = 250, temp: float = 0.0) -> str:
        """Run generation in a separate thread to avoid blocking the event loop."""
        return await anyio.to_thread.run_sync(self.generate, prompt, max_tokens, temp)

    def generate_chat(self, messages: list[dict], max_tokens: int = 512, temp: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response from a list of conversational messages.
        Supports Llama 3 special tokens natively.
        Does NOT aggressively slice the output at the last period to support code / formatting.
        """
        formatted_prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Append assistant header to prompt for generation
        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        sampler = sample_utils.make_sampler(temp=temp, top_p=top_p)
        
        response_text = ""
        for response in stream_generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            response_text += response.text
            if "<|eot_id|>" in response_text or "<|start_header_id|>" in response_text:
                break
        
        # Strip Llama-3 special tokens if they leak into generation
        for token in ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "assistant", "user", "system"]:
            if token in response_text:
                response_text = response_text.split(token)[0]

        return response_text.strip()

    async def generate_chat_async(self, messages: list[dict], max_tokens: int = 512, temp: float = 0.7, top_p: float = 0.9) -> str:
        """Run chat generation asynchronously."""
        return await anyio.to_thread.run_sync(self.generate_chat, messages, max_tokens, temp, top_p)
        


class FastDeceptionLLM:
    """
    Lightweight, fast LLM for deception grid only. Uses a smaller model so
    security/deception path stays quick and does not block the main GuardAgent.
    Set DECEPTION_LLM_MODEL env to override; set DECEPTION_LLM_DISABLED=1 to use templates only.
    """
    # Small 4-bit model for speed (alternative to main Llama-3 8B)
    DEFAULT_MODEL_ID = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

    def __init__(self, model_id: str = None):
        self.model_id = (os.environ.get("DECEPTION_LLM_MODEL") or model_id or self.DEFAULT_MODEL_ID).strip()
        self._model = None
        self._tokenizer = None
        if os.environ.get("DECEPTION_LLM_DISABLED", "").strip() == "1":
            self.model_id = None  # templates only
            return
        try:
            self._model, self._tokenizer = load(self.model_id)
            print("✅ Fast Deception LLM loaded (security path).")
        except Exception as e:
            print(f"⚠️ Fast Deception LLM not loaded ({e}); using templates only.")
            self.model_id = None

    def generate(self, prompt: str, max_tokens: int = 150, temp: float = 0.7) -> str:
        """Short, fast generation for decoy responses."""
        if self._model is None or self._tokenizer is None:
            return ""
        formatted = f"<|system|>\nYou reply briefly and professionally.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        sampler = sample_utils.make_sampler(temp=temp)
        out = generate(
            self._model, self._tokenizer, prompt=formatted, max_tokens=max_tokens,
            sampler=sampler, verbose=False
        )
        for t in ["</s>", "<|assistant|>", "<|user|>", "<|system|>"]:
            out = out.replace(t, "")
        return out.strip() or "Request received. Our team will follow up shortly."

    async def generate_async(self, prompt: str, max_tokens: int = 150, temp: float = 0.7) -> str:
        return await anyio.to_thread.run_sync(self.generate, prompt, max_tokens, temp)


if __name__ == "__main__":
    # Quick test harness (runs synchronously)
    llm = LocalLLM()
    question = "What are the common indicators of credit card fraud?"
    print(f"\nQuestion: {question}")
    response = llm.generate(question)
    print(f"\nResponse: {response}")
