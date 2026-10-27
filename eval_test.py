# -*- coding: utf-8 -*-
"""Standalone DRIVE test-set evaluation for FRetinalNet.

Protocol:
  - DRIVE official test split: all 20 test images, 512x512, CLAHE(2.0, 8x8)
  - per-image metrics binarized at threshold 0.5, averaged over images
  - model returns (fused_out, fused_out_low); the reported metric is the main (image-branch) head

Usage:
  python eval_test.py                                        # defaults below
  python eval_test.py --weights weights/dice_fdconv_drive_ori.pth --test-root data/DRIVE
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from skimage.morphology import skeletonize

from data.drive import DRIVE_Test_Dataset, transform
from fretinalnet_notebook import FRetinalNet

KEYS = ['ACC', 'Dice', 'IoU', 'AUC', 'SE', 'SP', 'clDice', 'Conn']


# ---- per-image metrics ----
def compute_batch_metrics(pred, gt, threshold=0.5, eps=1e-8):
    """ACC/Dice/IoU/AUC/SE/SP/clDice/Conn, computed per image then batch-averaged."""
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    B = pred_np.shape[0]

    accs, dices, ious, aucs, sens, specs = [], [], [], [], [], []
    cldices, conns = [], []

    for i in range(B):
        p = pred_np[i, 0]
        g = gt_np[i, 0]

        p_bin = (p >= threshold).astype(np.uint8)
        g_bin = (g >= threshold).astype(np.uint8)

        TP = np.sum((p_bin == 1) & (g_bin == 1))
        TN = np.sum((p_bin == 0) & (g_bin == 0))
        FP = np.sum((p_bin == 1) & (g_bin == 0))
        FN = np.sum((p_bin == 0) & (g_bin == 1))

        accs.append((TP + TN) / (TP + TN + FP + FN + eps))
        sens.append(TP / (TP + FN + eps))
        specs.append(TN / (TN + FP + eps))

        if TP + FP + FN == 0:
            dices.append(1.0); ious.append(1.0)
        else:
            dices.append(2 * TP / (2 * TP + FP + FN + eps))
            ious.append(TP / (TP + FP + FN + eps))

        if len(np.unique(g_bin.ravel())) == 1:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(g_bin.ravel(), p.ravel()))

        skel_p = skeletonize(p_bin).astype(bool)
        skel_g = skeletonize(g_bin).astype(bool)
        t_prec = np.sum(skel_p & g_bin) / (skel_p.sum() + eps)
        t_rec = np.sum(skel_g & p_bin) / (skel_g.sum() + eps)
        cld = (2 * t_prec * t_rec) / (t_prec + t_rec + eps)
        cldices.append(cld)

        comps_gt = label(skel_g, connectivity=2)
        n_gt = comps_gt.max()
        if n_gt == 0:
            conn_score = 1.0
        else:
            matched = 0
            for comp_id in range(1, n_gt + 1):
                if np.any(skel_p[comps_gt == comp_id]):
                    matched += 1
            conn_score = matched / n_gt
        conns.append(conn_score)

    return {k: v for k, v in zip(
        KEYS, [np.mean(accs), np.mean(dices), np.mean(ious), np.mean(aucs),
               np.mean(sens), np.mean(specs), np.mean(cldices), np.mean(conns)])}


def evaluate(model, weights_path, test_loader, device):
    print(f'\n[eval] loading weights: {weights_path}')
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    metrics = []
    # forward returns (fused_out, fused_out_low); the paper reports the
    # main (image-branch) head.
    with torch.no_grad():
        for image, mask_od in test_loader:
            image = image.to(device)
            mask_od = mask_od.to(device)
            out1, _ = model(image)
            out1 = torch.sigmoid(out1)
            metrics.append(compute_batch_metrics(out1, mask_od))

    arr = np.array([[m[k] for k in KEYS] for m in metrics])
    return dict(zip(KEYS, arr.mean(axis=0).tolist()))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=os.path.join(here, 'weights', 'dice_fdconv_drive_ori.pth'))
    ap.add_argument('--test-root', default=os.path.join(here, 'data', 'DRIVE'),
                    help='DRIVE root containing test/images and test/1st_manual')
    ap.add_argument('--test-images', default=None, help='override: dir of test .tif images')
    ap.add_argument('--test-masks', default=None, help='override: dir of manual1.gif masks')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--out', default=None, help='where to write the metrics JSON')
    args = ap.parse_args()

    images_dir = args.test_images or os.path.join(args.test_root, 'test', 'images')
    masks_dir = args.test_masks or os.path.join(args.test_root, 'test', '1st_manual')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[eval] device = {device}')

    model = FRetinalNet(num_classes=1)
    test_dataset = DRIVE_Test_Dataset(images_dir, masks_dir, transform=transform)
    print(f'[eval] DRIVE test images: {len(test_dataset)}')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    results = {os.path.basename(args.weights): evaluate(model, args.weights, test_loader, device)}

    for w, m in results.items():
        print('\n' + '=' * 72)
        print(f'weights: {w}   (test = all {len(test_dataset)} DRIVE test images, threshold 0.5)')
        print('  ' + ', '.join(f'{k}={m[k]:.4f}' for k in KEYS))

    out_path = args.out or os.path.join(here, 'results', 'eval_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n[eval] metrics saved to: {out_path}')


if __name__ == '__main__':
    main()
