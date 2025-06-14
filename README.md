# Tiny YOLO NAS

This repository implements a lightweight NAS (Neural Architecture Search) framework on top of YOLOv8, designed to search for efficient object detection models under resource constraints. It is the official implementation of the paper "Tiny YOLO NAS: Resource-aware Multi-Objective Neural Architecture Search for YOLO model"

## Project Structure

- `search.py` - Main NAS pipeline using Optuna.
- `yaml_generator.py` - Dynamically generates YOLOv8 model YAML files based on the searched architecture.
- `search_constraints.py` - Defines hardware and model constraints (size, GFLOPs, parameters).
- `dataset.yaml` - Dataset configuration file for training.
- `train.py` - Training the best architectures fully.
- `generated_yamls/` - Auto-generated model YAML files. (Created later)
- `bo_runs/` - Stores training results. (Created later)

---

## Dataset Preparation

The dataset follows YOLOv8 directory structure. You can modify `dataset.yaml` accordingly.

Example directory structure:

```bash
/dataset
├── train
│   └── images
│   └── labels
├── val
│   └── images
│   └── labels
```

Make sure your `dataset.yaml` contains:

```yaml
path: dataset
train: train/images
val: val/images

names:
  0: buffalo
  1: elephant
  2: rhino
  3: zebra
```

---

## Environment Setup

Install dependencies:

```bash
# Create virtual env (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Ultralytics YOLOv8
pip install -r requirements.txt

```

> **Note:** The code is tested with Ultralytics YOLOv8 official package.

---

## Running NAS

To start the architecture search:

```bash
python search.py
```

- The number of trials, image size, and number of epochs are controlled inside `search.py`.
- Modify variables in `search.py`:

```python
epochs = 3
imgsz = 96
trials = 5
nc = 4  # number of classes (adjust based on your dataset)
k = 3   # top-k architectures to display
```

- Search space is defined inside `search_constraints.py` and partially inside `search.py` using Optuna's sampling functions.

- After search completion, the best models' metrics and architectures are saved in `sorted_trials_user_attrs.txt`.

---

## Training a Selected Model

After NAS, you can retrain any selected architecture:

1. Pick the generated YAML path from the NAS output.

2. Run YOLOv8 training manually:

```bash
python train.py
```

- Replace `model_trial_XX.yaml` with your desired architecture.
- Adjust epochs, image size, batch size as needed.

---

## Customizing Constraints

Edit `search_constraints.py` to modify hardware constraints:

```python
constraints = {
    "max_model_size_kb": 900,
    "max_params_m": 0.5,
    "max_gflops": 2.0
}
```

NAS will discard models exceeding these limits.

---

## Notes

- This framework is fully standalone and does not require external NAS frameworks.
- Search space combines stage depth, channel scaling, C2F repeats, and inclusion of SPPF.
- The YAML generation logic strictly follows YOLOv8 model structure.

---

## Future Improvements

- Add multi-objective search support (latency-aware search)
- Add better visualization of searched architectures
- Support larger and more complicated datasets
- Support more advanced versions of YOLO

---

## License

[MIT License]