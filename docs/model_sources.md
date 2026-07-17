# Sources of bundled demo checkpoints

The fracture checkpoints are third-party research models and are used for an
educational prototype only:

- `fracture_yolov8_fast.pt`: `adeebaai/bone-fracture-yolov8` on Hugging Face;
- `fracture_yolov8l.pt`: `DimitriVavoulisPortfolio/x-ray-bone-fracture-detection-app`
  on GitHub, Apache-2.0 repository, trained on the referenced Kaggle dataset.

The upstream YOLOv8l model card reports mAP50 0.37, mAP50-95 0.18, precision
0.46 and recall 0.26. These values must not be presented as clinical validation.
The application is not a medical device and does not provide a diagnosis.
