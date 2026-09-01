# -*- coding: utf-8 -*-
"""Validate a local DRIVE dataset layout for eval_test.py.

DRIVE must be downloaded by the user (registration required):
  https://drive.grand-challenge.org/  (or the ISBI 2012 challenge download)

Expected layout under --root:
  DRIVE/
  ├── training/
  │   ├── images/      20 files *.tif          (e.g. 21_training.tif)
  │   └── 1st_manual/  20 files *_manual1.gif
  └── test/
      ├── images/      20 files *.tif          (e.g. 01_test.tif)
      └── 1st_manual/  20 files *_manual1.gif

Both splits are checked so the directory also supports training.

Usage:
  python scripts/prepare_drive.py --root data/DRIVE
"""
import argparse
import glob
import os
import sys

SPLIT_EXPECT = {'training': (20, 20), 'test': (20, 20)}


def check_split(root, split):
    images = sorted(glob.glob(os.path.join(root, split, 'images', '*.tif')))
    masks = sorted(glob.glob(os.path.join(root, split, '1st_manual', '*.gif')))
    ok = True
    if len(images) != SPLIT_EXPECT[split][0]:
        print(f'  [x] {split}/images: expected {SPLIT_EXPECT[split][0]} .tif files, found {len(images)}')
        ok = False
    else:
        print(f'  [ok] {split}/images: {len(images)} .tif')
    if len(masks) != SPLIT_EXPECT[split][1]:
        print(f'  [x] {split}/1st_manual: expected {SPLIT_EXPECT[split][1]} .gif files, found {len(masks)}')
        ok = False
    else:
        print(f'  [ok] {split}/1st_manual: {len(masks)} .gif')
    if images and masks:
        # image/mask pairs: NN_training.tif / NN_test.tif <-> NN_manual1.gif
        pairs = all(
            os.path.basename(i).rsplit('.', 1)[0].rsplit('_', 1)[0] in
            os.path.basename(m) for i, m in zip(images, masks))
        if not pairs:
            print('  [x] image/mask file names do not pair up (NN_training.tif <-> NN_manual1.gif)')
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/DRIVE', help='DRIVE dataset root')
    args = ap.parse_args()

    root = args.root
    print(f'checking DRIVE layout at: {root}')
    if not os.path.isdir(root):
        print(f'[x] directory not found. Download DRIVE (registration required) from '
              f'https://drive.grand-challenge.org/ and unpack it to this path.')
        sys.exit(1)

    ok = all(check_split(root, s) for s in SPLIT_EXPECT)
    if ok:
        print('\n[ok] DRIVE layout is valid. Run: python eval_test.py --test-root ' + root)
    else:
        print('\n[x] DRIVE layout is invalid; fix the errors above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
