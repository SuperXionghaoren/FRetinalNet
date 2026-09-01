# -*- coding: utf-8 -*-
"""Checkpoint compatibility test: the released checkpoint must load into
FRetinalNet with strict=True and cover every parameter/buffer.

Run: pytest tests/ -q          (from the repo root; set FRETINALNET_CKPT
if the checkpoint is not at the default weights/ location)
"""
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get(
    'FRETINALNET_CKPT', os.path.join(REPO, 'weights', 'dice_fdconv_drive_ori.pth'))


def test_checkpoint_loads_strict():
    if not os.path.exists(CKPT):
        pytest.skip(f'checkpoint not present: {CKPT} (set FRETINALNET_CKPT)')

    from fretinalnet_notebook import FRetinalNet
    model = FRetinalNet(num_classes=1)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    # one forward pass at the protocol resolution
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out1, out2 = model(x)
    assert out1.shape == (1, 1, 512, 512)
    assert out2.shape == (1, 1, 512, 512)
    print(f'checkpoint ok: {os.path.basename(CKPT)}, {len(state)} tensors, forward 512x512 pass')
