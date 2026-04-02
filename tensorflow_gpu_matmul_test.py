"""Run a small TensorFlow matrix multiply and report whether it used a GPU."""

from __future__ import annotations

import sys
import time

try:
    import tensorflow as tf
except Exception as exc:
    print(f"TensorFlow import failed: {exc}")
    raise SystemExit(1)


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")

    print(f"Python executable: {sys.executable}")
    print(f"TensorFlow version: {tf.__version__}")

    if not gpus:
        print("No GPU detected by TensorFlow. Skipping compute test.")
        raise SystemExit(1)

    size = 4096
    with tf.device("/GPU:0"):
        left = tf.random.normal((size, size), dtype=tf.float32)
        right = tf.random.normal((size, size), dtype=tf.float32)

        start = time.perf_counter()
        product = tf.matmul(left, right)
        checksum = float(tf.reduce_sum(product).numpy())
        elapsed = time.perf_counter() - start

    print("GPU compute test passed.")
    print(f"Matrix size: {size}x{size}")
    print(f"Result tensor device: {product.device}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Checksum: {checksum:.6f}")


if __name__ == "__main__":
    main()
