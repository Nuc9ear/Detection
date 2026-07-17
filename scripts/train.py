from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROFILES = {
    "fast": {"base": "yolo11n.pt", "imgsz": 640, "epochs": 80},
    "accurate": {"base": "yolo11s.pt", "imgsz": 960, "epochs": 100},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one VisionGuard detector")
    parser.add_argument("--task", choices=["powerline", "fracture", "vehicle"], required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--data", required=True, help="Path to data.yaml or VisDrone.yaml")
    parser.add_argument("--project", default="runs/detect")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    run_name = f"{args.task}_{args.profile}"
    model = YOLO(profile["base"])
    model.train(
        data=args.data,
        epochs=profile["epochs"],
        imgsz=1280 if args.task == "vehicle" and args.profile == "accurate" else profile["imgsz"],
        batch=-1,
        optimizer="AdamW",
        cos_lr=True,
        patience=20,
        seed=42,
        deterministic=True,
        project=args.project,
        name=run_name,
    )
    best = Path(args.project) / run_name / "weights" / "best.pt"
    target = Path("models") / f"{args.task}_yolo11{'n' if args.profile == 'fast' else 's'}.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()

