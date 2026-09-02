"""
gradcam_visual.py — Grad-CAM for the visual-manipulation detector
----------------------------------------------------------------------
Parallel to explain.gradcam.py (the lip-sync model's explainability), but
for FaceManipulationNet's full-face input instead of the mouth-only crop.

Deliberately does NOT modify app/models/visual.py. FaceManipulationNet
already exposes self.last_conv_maps as an instance attribute, but that
tensor is non-leaf and never had .retain_grad() called on it, so
.grad would be None after backward() without extra work. Rather than
editing the model file (touching a trained, checkpointed model's source
even in a way that doesn't change its weights still means every existing
checkpoint's provenance gets muddier -- better to keep it untouched), this
uses a standard PyTorch forward hook on model.trunk to capture the same
tensor externally and call retain_grad() on it at explain-time only. Same
result, zero risk to the model file or any existing checkpoint.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.models.visual import FaceManipulationNet


def _b64_png(img_rgb_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb_u8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def explain_face(model: FaceManipulationNet, face_crop_float01_hwc: np.ndarray) -> dict:
    """
    face_crop_float01_hwc: (224, 224, 3) float32 in [0,1], RGB -- same
    format as what dataset_visual.py caches and score_visual() consumes.

    Returns {"heatmap_b64": <PNG>, "fake_probability": float} -- a single
    overlay image (warm = pixels that most increased the manipulation
    score), not multiple frames, since this explains ONE face crop at a
    time (the caller picks which frame -- see inference_fusion.py).
    """
    model.eval()
    x = torch.from_numpy(face_crop_float01_hwc).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)

    captured = {}

    def _hook(module, inp, out):
        out.retain_grad()
        captured["maps"] = out

    handle = model.trunk.register_forward_hook(_hook)
    try:
        logit = model(x)                     # (1,)
        model.zero_grad(set_to_none=True)
        logit.backward()
    finally:
        handle.remove()

    maps = captured["maps"]                  # (1, 512, 7, 7)
    grads = maps.grad                        # (1, 512, 7, 7)

    # Standard Grad-CAM: global-average-pool the gradients per channel to
    # get an importance weight for each of the 512 channels, then take a
    # weighted sum of the (unweighted) feature maps -- channels the
    # gradient says mattered more contribute more to the final heatmap.
    weights = grads.mean(dim=(2, 3), keepdim=True)       # (1, 512, 1, 1)
    cam = F.relu((weights * maps).sum(dim=1))            # (1, 7, 7)
    cam = cam.squeeze(0).detach().numpy()                 # (7, 7)

    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heat_u8 = (cam_resized * 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    face_u8 = (np.clip(face_crop_float01_hwc, 0, 1) * 255).astype(np.uint8)
    overlay = (0.55 * face_u8 + 0.45 * heat_color_rgb).astype(np.uint8)

    with torch.no_grad():
        prob = float(torch.sigmoid(logit).item())

    return {"heatmap_b64": _b64_png(overlay), "fake_probability": prob}
