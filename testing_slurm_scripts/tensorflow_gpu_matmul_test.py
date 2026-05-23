"""Run a small PyTorch matrix multiply and report whether it used a GPU."""

from __future__ import annotations

import sys
import time

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
    raise SystemExit(1)


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("No GPU detected by PyTorch. Skipping compute test.")
        raise SystemExit(1)

    device = torch.device("cuda:0")
    size = 4096
    left = torch.randn(size, size, dtype=torch.float32, device=device)
    right = torch.randn(size, size, dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    product = torch.matmul(left, right)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    checksum = float(product.sum().item())

    print("GPU compute test passed.")
    print(f"Matrix size: {size}x{size}")
    print(f"Result tensor device: {product.device}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Checksum: {checksum:.6f}")


if __name__ == "__main__":
    main()
