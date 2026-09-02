"""
Quick standalone check that the trained checkpoint + Grad-CAM explainability
layer work end-to-end. Not part of the permanent codebase -- just a smoke
test. Run from backend/ with the venv active:

    python test_gradcam.py "E:\\TruthSeeker-SSD\\FakeAVCeleb_v1.2\\...\\some_fake.mp4"
"""
import json
import sys

from app.inference import analyze_video

if __name__ == "__main__":
    video_path = sys.argv[1]
    result = analyze_video(video_path)

    print(f"model_trained     : {result['model_trained']}")
    print(f"verdict           : {result['verdict']}")
    print(f"fake_probability  : {result['fake_probability']}")
    print(f"n_windows         : {result['video']['n_windows']}")
    print(f"face_detect_rate  : {result['video']['face_detection_rate']}")
    print()
    print("explanation summary:")
    print(" ", result["explanation"]["summary"])
    print()
    print("explanation keys  :", list(result["explanation"].keys()))

    # Save the full JSON so we can inspect the Grad-CAM image data (base64)
    # without flooding the terminal.
    with open("gradcam_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nfull result written to gradcam_result.json")
