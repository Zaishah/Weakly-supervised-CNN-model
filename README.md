# Weakly Supervised Video Anomaly Detection with 3D CNN

This project implements a weakly supervised deep learning pipeline to detect anomalies (accidents) in videos using a 3D Convolutional Neural Network (3D CNN). It processes raw video datasets of normal and accident traffic scenarios, applies data augmentation, segments videos, trains the model, and saves it for inference.

---
## Dataset

This project uses the Car Crash Dataset originally introduced by Bao et al. It contains videos of normal and accident traffic scenarios.
Visit the [GitHub Repository](https://github.com/Cogito2012/CarCrashDataset) for more details.
## Citation
```bibtex
@InProceedings{BaoMM2020,
    author = {Bao, Wentao and Yu, Qi and Kong, Yu},
    title  = {Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning},
    booktitle = {ACM Multimedia Conference},
    month  = {May},
    year   = {2020}
}




## Features

- Download video datasets from Google Drive  
- Extract a subset of videos (first 500 from each class)  
- Preprocess videos into fixed-length segments of frames  
- Apply temporal and spatial augmentations to video segments  
- Train a 3D CNN to classify normal vs accident videos  
- Early stopping and learning rate reduction callbacks  
- Save the trained model for later use  

---

## Requirements

- Python 3.x  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- tqdm  
- scikit-learn  
- gdown (for Google Drive downloads)  

Install dependencies via pip if needed:

```bash
pip install tensorflow opencv-python numpy tqdm scikit-learn gdown
