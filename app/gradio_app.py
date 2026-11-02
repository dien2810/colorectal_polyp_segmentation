"""Demo: upload a colonoscopy image, get the predicted polyp mask overlaid on it.

    pip install gradio
    python app/gradio_app.py --run outputs/runs/cenet            # add --share on Colab

Academic demo on public data only. It is not a medical device.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.inference import load_run, predict_proba, to_original_size
from src.utils import get_device


def build_demo(run_dir, checkpoint="best.pt", tta=True):
    import gradio as gr
    device = get_device()
    cfg, model, transform = load_run(run_dir, device, checkpoint)
    thr = cfg.get("eval", {}).get("threshold", 0.5)

    def segment(image: np.ndarray, threshold: float):
        if image is None:
            return None, "Upload an image."
        x = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = to_original_size(predict_proba(model, x, tta=tta), image.shape[:2])
        mask = prob > threshold
        out = image.astype(float).copy()
        out[mask] = 0.5 * out[mask] + 0.5 * np.array([0, 255, 0])
        share = 100 * mask.mean()
        return out.astype(np.uint8), f"Predicted polyp area: {share:.1f}% of the image"

    return gr.Interface(
        fn=segment,
        inputs=[gr.Image(type="numpy", label="Colonoscopy image"),
                gr.Slider(0.1, 0.9, value=thr, step=0.05, label="Threshold")],
        outputs=[gr.Image(label="Predicted mask (green)"), gr.Textbox(label="Summary")],
        title=f"Polyp segmentation demo ({cfg['experiment']})",
        description="Academic demo trained on public datasets (Kvasir-SEG, CVC-ClinicDB). "
                    "Not a medical device.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--share", action="store_true", help="public link (useful on Colab)")
    args = ap.parse_args()
    build_demo(args.run, args.checkpoint, tta=not args.no_tta).launch(share=args.share)


if __name__ == "__main__":
    main()
