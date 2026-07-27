import os
import anyio
from pathlib import Path
from PIL import Image
import mlx_vlm
from mlx_vlm import load, generate

class VisionLLM:
    """
    Local Vision-Language Model using mlx-vlm.
    Optimized for Apple Silicon.
    """
    
    # Default visual model
    MODEL_ID = "mlx-community/llava-1.5-7b-4bit" # Good balance of speed/quality
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or self.MODEL_ID
        self.model = None
        self.processor = None
        self.config = None
        print(f"VisionLLM initialized (lazy load): {self.model_id}")

    def _ensure_loaded(self):
        if self.model is None:
            print(f"Lazy loading local Vision MLX model: {self.model_id}...")
            # Load model and processor
            self.model, self.processor = load(self.model_id)
            print("✅ Local Vision MLX LLM loaded successfully.")
            
            # Patch for LLaVA 1.5 if processor has None patch_size (common bug)
            if hasattr(self.processor, "patch_size") and self.processor.patch_size is None:
                self.processor.patch_size = 14

    def analyze_image(self, image_path: str, prompt: str, max_tokens: int = 300) -> str:
        """Analyze an image with a text prompt."""
        self._ensure_loaded()
        # mlx-vlm expects a PIL Image or path
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        # run generation
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            image_path,
            max_tokens=max_tokens,
            verbose=False
        )
        return result.text.strip()

    async def analyze_image_async(self, image_path: str, prompt: str, max_tokens: int = 300) -> str:
        """Run vision analysis in a separate thread."""
        return await anyio.to_thread.run_sync(self.analyze_image, image_path, prompt, max_tokens)

if __name__ == "__main__":
    # Test harness
    # vllm = VisionLLM()
    # print(vllm.analyze_image("test.png", "What is in this image?"))
    pass
