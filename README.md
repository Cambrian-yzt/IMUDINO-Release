# IMUDINO

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/Octopus1/IMUDINO)
[![arXiv](https://img.shields.io/badge/arXiv-2611.07437-b31b1b)](http://arxiv.org/abs/2611.07437)

Official PyTorch release of IMUDINO, a general purpose foundation model for inertial sensing.
See the [paper](http://arxiv.org/abs/2611.07437) for details.

## Download the checkpoint

The weights are hosted on Hugging Face: [Octopus1/IMUDINO](https://huggingface.co/Octopus1/IMUDINO).
Download `imudino.pt` into this repository's root directory:

```bash
# huggingface-cli (pip install "huggingface_hub[cli]")
huggingface-cli download Octopus1/IMUDINO imudino.pt --local-dir .

# or plain download
wget https://huggingface.co/Octopus1/IMUDINO/resolve/main/imudino.pt
```

The example scripts expect `imudino.pt` next to them by default.

The backbone uses 16-sample patches, 12 transformer layers, 24 attention heads,
768-dimensional features, and 4 register tokens. It embeds all six sensor channels
jointly and uses standard GELU MLPs. Frequency-aware positional encoding always
uses the supplied sampling rates and a fixed 100 Hz reference. The factory retains
its original name, `imuformer_huge_patch16`, for compatibility.
The implementation lives in `imudino.py`; constructing `IMUFormer()` directly
uses the same settings as the factory.

Install PyTorch and timm in a Python 3.10+ environment:

```bash
pip install -r requirements.txt
python example_forward.py
```

Use `--device cuda` for a CUDA device or `--checkpoint /path/to/imudino.pt` for a
different checkpoint location. The example loads weights strictly, switches to
evaluation mode, and runs inference on a synthetic batch with padding. Tested
with Python 3.12, PyTorch 2.15.0, and timm 1.0.30.

To use the model in your own code:

```python
import torch
from model_factory import imuformer_huge_patch16

model = imuformer_huge_patch16()
model.load_state_dict(
    torch.load("imudino.pt", map_location="cpu", weights_only=True), strict=True
)
model.eval()

with torch.inference_mode():
    features = model({
        "imu_data": torch.randn(2, 1024, 6),
        "freqs": torch.tensor([100.0, 200.0]),
        "masks": None,
    })
```

Input `imu_data` has shape `[B, L, 6]`, with accelerometer XYZ followed by gyroscope
XYZ. `L` must be a positive multiple of 16. `freqs` contains one positive sampling
frequency in Hz per sequence. Use floating-point inputs on the same device as the
model.

The optional `masks` tensor has shape `[B, L]`: `True`/`1` means valid and
`False`/`0` means padding. A patch is valid only when all 16 samples are valid.
Omit `masks` or pass `None` for unpadded input. Padded patches remain in the output;
exclude them when pooling or using patch features downstream.

The output contains only the final transformer layer's tokens after layer normalization:

| Key | Shape | Description |
| --- | --- | --- |
| `cls_token` | `[B, 768]` | Sequence representation |
| `reg_tokens` | `[B, 4, 768]` | Register representations |
| `patch_tokens` | `[B, L / 16, 768]` | Temporal patch representations |

For the example batch, these shapes are `(2, 768)`, `(2, 4, 768)`, and `(2, 64, 768)`.
