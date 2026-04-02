"""Check whether TensorFlow can use any GPU devices."""

import sys

try:
    import tensorflow as tf
except Exception as exc:
    print(f"TensorFlow import failed: {exc}")
    raise SystemExit(1)


def main() -> None:
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")

    print(f"Python executable: {sys.executable}")
    print(f"TensorFlow version: {tf.__version__}")

    if not physical_gpus:
        print("No GPU detected by TensorFlow.")
        return

    print(f"TensorFlow detected {len(physical_gpus)} physical GPU(s):")
    for index, gpu in enumerate(physical_gpus, start=1):
        print(f"{index}. {gpu}")

    print(f"TensorFlow exposed {len(logical_gpus)} logical GPU(s):")
    for index, gpu in enumerate(logical_gpus, start=1):
        print(f"{index}. {gpu}")


if __name__ == "__main__":
    main()
