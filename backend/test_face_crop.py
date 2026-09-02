"""
Phase 1 smoke test: confirms face_crop extraction works end-to-end on a
real video before committing to Phase 2's multi-hour cache build.

Run from backend/ with the venv active:
    python test_face_crop.py "path\\to\\some_video.mp4"
"""
import sys

import cv2
import numpy as np

from app.preprocessing.video_utils import extract_lip_sequence

if __name__ == "__main__":
    video_path = sys.argv[1]
    seq = extract_lip_sequence(video_path)

    n_lip = sum(1 for r in seq if r.lip_crop is not None)
    n_face = sum(1 for r in seq if r.face_crop is not None)
    print(f"{len(seq)} frames total")
    print(f"  lip crops : {n_lip} ({n_lip/len(seq):.1%})")
    print(f"  face crops: {n_face} ({n_face/len(seq):.1%})")

    # Save one face crop as a real image file so it can be visually inspected.
    example = next((r for r in seq if r.face_crop is not None), None)
    if example is not None:
        img = (example.face_crop * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("face_crop_sample.jpg", img_bgr)
        print(f"\nSaved a sample face crop (t={example.timestamp:.2f}s) "
              f"to face_crop_sample.jpg -- open it to check the crop looks right.")
    else:
        print("\nNo face crops were extracted at all -- something's wrong.")
