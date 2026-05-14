"""
MoNuSeg Preprocessing Script
==============================
Converts MoNuSeg XML polygon annotations → binary PNG masks
and organizes data into the folder structure expected by the notebooks.

Expected input structure (what you downloaded):
    monuseg_raw/
        train/
            Tissue Images/       ← .tif images
            Annotations/         ← .xml annotation files
        test/
            <images and .xml files in same folder>

Output structure (what the notebooks expect):
    data/
        train/
            images/   ← .png images
            masks/    ← binary .png masks (255=nucleus, 0=background)
        val/
            images/
            masks/
        test/
            images/
            masks/

Usage:
    python preprocess_monuseg.py \
        --train_img   "monuseg_raw/train/Tissue Images" \
        --train_ann   "monuseg_raw/train/Annotations" \
        --test_dir    "monuseg_raw/test" \
        --output_dir  "data" \
        --val_split   0.2 \
        --seed        42
"""

import os
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from lxml import etree
from PIL import Image
import cv2

# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='MoNuSeg preprocessing')
    parser.add_argument('--train_img',  type=str,
                        default='monuseg_raw/train/Tissue Images',
                        help='Path to training tissue images folder')
    parser.add_argument('--train_ann',  type=str,
                        default='monuseg_raw/train/Annotations',
                        help='Path to training annotations (XML) folder')
    parser.add_argument('--test_dir',   type=str,
                        default='monuseg_raw/test',
                        help='Path to test folder (images + XML together)')
    parser.add_argument('--output_dir', type=str,   default='data')
    parser.add_argument('--val_split',  type=float, default=0.2,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--seed',       type=int,   default=42)
    return parser.parse_args()


# ── XML → binary mask conversion ────────────────────────────────────────────

def xml_to_mask(xml_path: Path, image_shape: tuple) -> np.ndarray:
    """
    Parse a MoNuSeg XML annotation file and render all nucleus polygons
    into a binary mask of shape (H, W), dtype uint8.

    MoNuSeg XML structure:
        <Annotation>
          <Regions>
            <Region>
              <Vertices>
                <Vertex X="..." Y="..." />
                ...
              </Vertices>
            </Region>
          </Regions>
        </Annotation>

    Returns:
        mask: np.ndarray shape (H, W), values {0, 255}
              255 = nucleus (tumor cell), 0 = background
    """
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Handle both direct Annotation root and wrapped structures
    regions = root.findall('.//Region')
    if len(regions) == 0:
        print(f'  Warning: no regions found in {xml_path.name}')
        return mask

    for region in regions:
        vertices = region.findall('.//Vertex')
        if len(vertices) < 3:
            # Need at least 3 points to form a polygon
            continue

        # Extract (x, y) coordinates
        polygon = []
        for v in vertices:
            x = float(v.get('X'))
            y = float(v.get('Y'))
            polygon.append([x, y])

        polygon = np.array(polygon, dtype=np.int32)

        # Fill the polygon on the mask
        cv2.fillPoly(mask, [polygon], color=255)

    return mask


# ── Utility: save image as PNG ───────────────────────────────────────────────

def save_image_as_png(src_path: Path, dst_path: Path):
    """Load any image format (tif, png, jpg) and save as PNG."""
    img = Image.open(src_path).convert('RGB')
    img.save(dst_path)


# ── Main processing ──────────────────────────────────────────────────────────

def process_split(img_paths, ann_paths, out_img_dir, out_mask_dir, split_name):
    """
    Process a list of (image_path, annotation_path) pairs into a split folder.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for img_path, ann_path in zip(img_paths, ann_paths):
        stem = img_path.stem   # filename without extension

        # ── Load image to get dimensions ──────────────────────────────────
        img = np.array(Image.open(img_path).convert('RGB'))
        H, W = img.shape[:2]

        # ── Convert XML annotation → binary mask ──────────────────────────
        mask = xml_to_mask(ann_path, (H, W))

        # ── Save image as PNG ──────────────────────────────────────────────
        out_img_path  = out_img_dir  / f'{stem}.png'
        out_mask_path = out_mask_dir / f'{stem}.png'

        Image.fromarray(img).save(out_img_path)
        Image.fromarray(mask).save(out_mask_path)

        nucleus_pct = (mask > 0).mean() * 100
        print(f'  [{split_name}] {stem}.png  '
              f'size={W}×{H}  nucleus={nucleus_pct:.1f}%  '
              f'nuclei regions={count_regions(mask)}')
        success += 1

    print(f'  → {success} images processed into {split_name}/')
    return success


def count_regions(mask):
    """Count number of connected nucleus regions in mask (for sanity check)."""
    _, n = cv2.connectedComponents((mask > 0).astype(np.uint8))
    return n - 1   # subtract background component


def find_annotation(ann_dir: Path, stem: str) -> Path:
    """
    Find the XML annotation file matching an image stem.
    MoNuSeg annotation filenames sometimes have different suffixes.
    """
    # Try exact match first
    candidates = [
        ann_dir / f'{stem}.xml',
        ann_dir / f'{stem}.XML',
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fuzzy match: annotation stem might differ slightly
    for xml_file in ann_dir.glob('*.xml'):
        if xml_file.stem.startswith(stem[:10]):   # first 10 chars
            return xml_file

    raise FileNotFoundError(
        f'Could not find annotation for {stem} in {ann_dir}')


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    train_img_dir = Path(args.train_img)
    train_ann_dir = Path(args.train_ann)
    test_dir      = Path(args.test_dir)
    output_dir    = Path(args.output_dir)

    # ── Validate input paths ───────────────────────────────────────────────
    for p, name in [(train_img_dir, 'train_img'),
                    (train_ann_dir, 'train_ann'),
                    (test_dir,      'test_dir')]:
        if not p.exists():
            raise FileNotFoundError(f'{name} path not found: {p}')

    print(f'\n{"="*60}')
    print(f'MoNuSeg Preprocessing')
    print(f'{"="*60}')
    print(f'Train images : {train_img_dir}')
    print(f'Train annots : {train_ann_dir}')
    print(f'Test dir     : {test_dir}')
    print(f'Output dir   : {output_dir}')
    print(f'Val split    : {args.val_split}')
    print(f'{"="*60}\n')

    # ── Gather training images and annotations ─────────────────────────────
    # MoNuSeg training images are .tif
    all_train_imgs = sorted(
        list(train_img_dir.glob('*.tif')) +
        list(train_img_dir.glob('*.png')) +
        list(train_img_dir.glob('*.jpg'))
    )
    if len(all_train_imgs) == 0:
        raise FileNotFoundError(f'No images found in {train_img_dir}')

    print(f'Found {len(all_train_imgs)} training images.')

    # Match each image to its annotation
    all_train_anns = []
    for img_path in all_train_imgs:
        ann_path = find_annotation(train_ann_dir, img_path.stem)
        all_train_anns.append(ann_path)

    # ── Train / val split (by patient, not by slide) ──────────────────────
    indices = list(range(len(all_train_imgs)))
    random.shuffle(indices)
    n_val   = max(1, int(len(indices) * args.val_split))
    n_train = len(indices) - n_val

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_imgs = [all_train_imgs[i] for i in train_indices]
    train_anns = [all_train_anns[i] for i in train_indices]
    val_imgs   = [all_train_imgs[i] for i in val_indices]
    val_anns   = [all_train_anns[i] for i in val_indices]

    print(f'Split: {len(train_imgs)} train / {len(val_imgs)} val')

    # ── Process training split ─────────────────────────────────────────────
    print(f'\nProcessing training set...')
    process_split(
        train_imgs, train_anns,
        output_dir / 'train' / 'images',
        output_dir / 'train' / 'masks',
        'train'
    )

    # ── Process validation split ───────────────────────────────────────────
    print(f'\nProcessing validation set...')
    process_split(
        val_imgs, val_anns,
        output_dir / 'val' / 'images',
        output_dir / 'val' / 'masks',
        'val'
    )

    # ── Process test set ───────────────────────────────────────────────────
    # MoNuSeg test: images and XMLs are in the same folder
    print(f'\nProcessing test set...')
    test_imgs = sorted(
        list(test_dir.glob('*.tif')) +
        list(test_dir.glob('*.png')) +
        list(test_dir.glob('*.jpg'))
    )
    test_anns = []
    for img_path in test_imgs:
        ann_path = find_annotation(test_dir, img_path.stem)
        test_anns.append(ann_path)

    print(f'Found {len(test_imgs)} test images.')
    process_split(
        test_imgs, test_anns,
        output_dir / 'test' / 'images',
        output_dir / 'test' / 'masks',
        'test'
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Done! Output structure:')
    print(f'  {output_dir}/train/images/  ({len(train_imgs)} images)')
    print(f'  {output_dir}/train/masks/   ({len(train_imgs)} masks)')
    print(f'  {output_dir}/val/images/    ({len(val_imgs)} images)')
    print(f'  {output_dir}/val/masks/     ({len(val_imgs)} masks)')
    print(f'  {output_dir}/test/images/   ({len(test_imgs)} images)')
    print(f'  {output_dir}/test/masks/    ({len(test_imgs)} masks)')
    print(f'\nYou can now run the FCN or U-Net notebooks.')
    print(f'Set DATA_DIR = Path("{output_dir}") in the Config cell.')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
