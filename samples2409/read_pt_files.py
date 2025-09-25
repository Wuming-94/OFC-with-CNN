import torch
import sys


def main():
    if len(sys.argv) < 2:
        print("files name")
        return

    filename = sys.argv[1]
    try:
        data = torch.load(filename)
        print(f"Data type: {type(data)}")

        if isinstance(data, dict):
            print("data Dict:")
            for key, value in data.items():
                print(f"{key}: {value}")
        elif isinstance(data, torch.Tensor):
            print("data tensor:")
            print(f"shape: {data.shape}")
            print(f"data type: {data.dtype}")
            print(f"values: {data}")
        else:
            print("data:")
            print(data)

    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    main()