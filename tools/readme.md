## Script Descriptions

### Data Processing

- **preprocess_floorplan.py** — Convert SVG floorplans to NPZ format with point sampling and semantic/instance labeling
- **floorplan_data_utils.py** — Utility class for loading and processing Floorplan dataset information
- **indoor_converter.py** — Create dataset info files (PKL) for Floorplan datasets
- **create_data.py** — Main data preparation pipeline that orchestrates dataset conversion and info file generation
- **update_infos_to_v2.py** — Convert annotation PKL files to OpenMMLab V2.0 standard format

### Evaluation & Visualization

- **create_coco_gt.py** — Generate COCO-format ground truth annotations from NPZ files for evaluation
- **test_map.py** — Run COCO-style bbox evaluation on prediction JSON files
- **visual_result.py** — Visualize model predictions by rendering SVG and PNG outputs with bounding boxes

## Workflow

### SVG -> NPZ -> PKL

```bash
# Step 1: Convert SVG to NPZ with point sampling
python tools/preprocess_floorplan.py \
  --datasets train,test,val \
  --raw_base_path data/floorplan_raw \
  --d_samp 0.14285714285714285

# Step 2: Generate PKL info files
python tools/create_data.py s3dfloorplan \
  --root-path data/newfloorplan \
  --out-dir data/newfloorplan \
  --extra-tag s3dfloorplan
```

### Evaluation

```bash
# Generate COCO-format annotations
python tools/create_coco_gt.py \
  --root-dir data/floorplan_color/test_color/npz_gt \
  --output-json aaai_test/annotations_test.json

# Run COCO evaluation
python tools/test_map.py \
  --ann-file aaai_test/annotations_test.json \
  --res-file aaai_test/pred_test_coco_mixedpooling.json
```
