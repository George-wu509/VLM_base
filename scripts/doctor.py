"""Report environment and GPU availability without downloading any models."""
import importlib.metadata
import json
import platform
import torch

def main():
    report = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "torch_cuda": torch.version.cuda, "transformers": importlib.metadata.version("transformers")}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(), vram_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                      bf16=torch.cuda.is_bf16_supported())
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
