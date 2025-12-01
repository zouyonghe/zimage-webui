import torch

def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        # No GPU detected or CUDA not set up; early exit.
        return

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"Device {idx}: {props.name}, "
            f"CC {props.major}.{props.minor}, "
            f"total memory {props.total_memory / (1024 ** 3):.2f} GB"
        )

if __name__ == "__main__":
    main()
