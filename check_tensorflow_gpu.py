"""Check whether PyTorch can use any GPU devices."""

import sys

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
    raise SystemExit(1)


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No GPU detected by PyTorch.")
        return

    device_count = torch.cuda.device_count()
    print(f"PyTorch detected {device_count} GPU(s):")
    for index in range(device_count):
        print(f"{index + 1}. {torch.cuda.get_device_name(index)} (cuda:{index})")

    print(f"Current CUDA device: cuda:{torch.cuda.current_device()}")


if __name__ == "__main__":
    main()
