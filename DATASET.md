# Floorplan Dataset

## Status

The full floorplan dataset is not bundled with this public repository.

- Public sample kept in repo: `data/smoke/newfloorplan/`
- Real training root expected by the main configs: `data/newfloorplan/`

## Expected Layout

Prepare the real dataset under:

```text
data/newfloorplan/
├── points/
├── instance_mask/
├── semantic_mask/
├── reverse_map/
├── s3dfloorplan_infos_Area_1.pkl
├── s3dfloorplan_infos_Area_2.pkl
├── s3dfloorplan_infos_Area_3.pkl
└── s3dfloorplan_infos_Area_4.pkl
```

Each sample is expected to have matching files across:

- `points/`
- `instance_mask/`
- `semantic_mask/`
- `reverse_map/`

## Default Area Splits

The main public training config is:

```text
configs/aaai_nocolor_1024_nopooling.py
```

Its default split is:

- training: `train_area = [1]`
- validation / test: `test_area = [3]`

Info generation expects areas `1..4`.

## Generate Info PKLs

After the prepared point, mask, and reverse-map files are in place, generate the metadata files with:

```bash
python tools/create_data.py s3dfloorplan \
  --root-path data/newfloorplan \
  --out-dir data/newfloorplan \
  --extra-tag s3dfloorplan
```

## Raw Preprocess Entry Points

If you start from raw floorplan assets instead of prepared `.bin` / mask files, use these entry points:

- `tools/inkscape.sh`
- `tools/svg_tsfm_scale.py`
- `tools/preprocess_floorplan_large.py`
- `tools/readme.md`

## Additional Evaluation Assets

Some COCO-style evaluation helpers expect extra assets under:

```text
data/floorplan_color/test_color/npz_gt/
```

These assets are not bundled with the public repo.
