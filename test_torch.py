import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"Bfloat16 支持: {torch.cuda.is_bf16_supported()}") # Qwen3 推荐使用 bf16