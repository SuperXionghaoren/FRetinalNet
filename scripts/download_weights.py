# -*- coding: utf-8 -*-
"""Download the pretrained weights from Hugging Face Hub.

Defaults fetch both files into ./weights/:
  - dice_fdconv_drive_ori.pth  (final DRIVE model, used by eval_test.py)
  - STAGE_FDCONV.pth           (stage-1 encoder, only needed for training)

Usage:
  python scripts/download_weights.py
  python scripts/download_weights.py --repo-id <your-hf-username>/FRetinalNet
"""
import argparse
import os

DEFAULT_REPO_ID = 'waspwallbvb/FRetinalNet'
FILES = ['dice_fdconv_drive_ori.pth', 'STAGE_FDCONV.pth']


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', default=DEFAULT_REPO_ID,
                    help='Hugging Face repo id hosting the weights')
    ap.add_argument('--out-dir', default=os.path.join(here, 'weights'))
    ap.add_argument('--files', nargs='+', default=FILES)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('huggingface_hub is required: pip install huggingface_hub')

    for name in args.files:
        print(f'[download] {args.repo_id}/{name} -> {args.out_dir}')
        path = hf_hub_download(repo_id=args.repo_id, filename=name,
                               local_dir=args.out_dir)
        print(f'[download] done: {path}')


if __name__ == '__main__':
    main()
