# Installation

All commands below are expected to run from the project root:

```bash
export PROJECT_ROOT=/path/to/repo
cd "$PROJECT_ROOT"
```

## 1. Create the conda environment

```bash
conda create -n cadspotting python=3.10
conda activate cadspotting
```

## 2. Install build helpers

`cairosvg` and CUDA extension builds need these helpers:

```bash
conda install -n cadspotting -y pkg-config cmake cairo git ninja
```

## 3. Install PyTorch cu121

```bash
conda run -n cadspotting python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install compatibility packages first

Keep NumPy on the 1.x line for this checked path. This avoids the `matplotlib` / `mmdet3d` ABI error seen with NumPy 2.x.

```bash
conda run -n cadspotting python -m pip install \
  "numpy<2" "opencv-python==4.10.0.84" psutil
```

## 5. Install the OpenMMLab stack

```bash
conda run -n cadspotting python -m pip install \
  mmengine==0.10.6 mmdet==3.2.0
```

`mmcv 2.1.0` does not currently have a matching prebuilt wheel for this exact `torch 2.5 + cu121` path, so this step builds it locally:

```bash
conda run -n cadspotting env MMCV_WITH_OPS=1 FORCE_CUDA=1 \
  python -m pip install -v --no-build-isolation \
  mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.5.0/index.html
```

```bash
conda run -n cadspotting python -m pip install mmdet3d==1.4.0
```

## 6. Install CUDA extensions used by the default PTv3 path

```bash
conda run -n cadspotting python -m pip install spconv-cu121
```

```bash
conda run -n cadspotting python -m pip install \
  torch_scatter \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

```bash
conda run -n cadspotting python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation
```

## 7. Install the remaining project dependencies

```bash
conda run -n cadspotting python -m pip install -r requirements.txt
```

## 8. Optional extensions

The validated PTv3 configs do not need these packages:

- `MinkowskiEngine`
- `pointops`

If you need legacy or alternate backbones that depend on them, install them separately with a build or wheel that matches your CUDA and PyTorch stack.

## 9. Validate the environment

```bash
conda run -n cadspotting env PYTHONPATH=$(pwd) \
  python -c "import torch, mmengine, mmdet3d, oneformer3d; print('env ok')"
```

```bash
conda run -n cadspotting env PYTHONPATH=$(pwd) python tools/train.py --help
conda run -n cadspotting env PYTHONPATH=$(pwd) python tools/test.py --help
conda run -n cadspotting python -m pip check
```

## Notes

- `requirements.txt` is intentionally not version locked.
- The checked combination is `torch 2.5.1+cu121`, `mmcv 2.1.0`, `mmdet 3.2.0`, `mmengine 0.10.6`, `mmdet3d 1.4.0`.
- `mmcv 2.1.0` is compiled locally in this path and may take several minutes.
- The public smoke checkpoint is downloaded with `bash download_public_ckpt.sh`.
- The real floorplan dataset is not bundled. See `DATASET.md` for the expected layout under `data/newfloorplan/`.
- If you accidentally upgrade to NumPy 2.x and hit `numpy.core.multiarray failed to import` or `_ARRAY_API not found`, run:

```bash
conda run -n cadspotting python -m pip install \
  "numpy<2" "opencv-python==4.10.0.84"
```
