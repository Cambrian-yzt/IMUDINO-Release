"""Load the released checkpoint and extract features from synthetic IMU data."""

import argparse
from pathlib import Path

import torch

from model_factory import imuformer_huge_patch16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).with_name("imudino.pt")
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = imuformer_huge_patch16()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # Replace this synthetic batch with your IMU data: [batch, samples, 6].
    # Channels: accelerometer X/Y/Z, then gyroscope X/Y/Z.
    inputs = {
        "imu_data": torch.randn(2, 1024, 6, device=device),
        "freqs": torch.tensor([100.0, 200.0], device=device),
        "masks": torch.ones(2, 1024, dtype=torch.bool, device=device),
    }
    # The second sequence has 800 valid samples, followed by zero padding.
    inputs["masks"][1, 800:] = False
    inputs["imu_data"][1, 800:] = 0

    with torch.inference_mode():
        outputs = model(inputs)

    for name, tokens in outputs.items():
        print(f"{name}: {tuple(tokens.shape)}")


if __name__ == "__main__":
    main()
