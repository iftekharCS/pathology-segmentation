# FCN Segmentation Baseline

A modular FCN (Fully Convolutional Network) pipeline for binary segmentation of tumor regions in histopathology images.

---

## Setup

```bash
conda env create -f environment.yml
conda activate pathseg
```

---

## Data Format

The dataset is not included. Organize your data as follows before running:

```
data/
  train/
    images/    ← RGB patches (.png or .tif)
    masks/     ← Binary masks (.png) — 255 = tumor, 0 = background
  val/
    images/
    masks/
  test/
    images/
    masks/
```

---

## Running

Open `pipeline_1_fcn.ipynb` and update the **Config cell** at the top:

```python
DATA_DIR = Path('data')   # point to your data folder
IMG_SIZE = 512            # adjust to match your patch size
```

Then run all cells top to bottom.

---

## MoNuSeg (Optional Test Dataset)

To test on the public [MoNuSeg benchmark](https://monuseg.grand-challenge.org/),
download the dataset and run the preprocessing script to convert XML annotations to PNG masks:

```bash
python preprocess_monuseg.py \
    --train_img  "monuseg_raw/train/Tissue Images" \
    --train_ann  "monuseg_raw/train/Annotations" \
    --test_dir   "monuseg_raw/test" \
    --output_dir "data"
```
