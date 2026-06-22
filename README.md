# rice_leaf_detection
# Rice Leaf Disease Detection and Classification

## Overview

This project presents a comparative study of **classical machine learning, hybrid learning, and deep learning approaches** for automated rice leaf disease classification. Using the **Paddy Doctor dataset**, the study evaluates different learning paradigms based on classification performance and computational efficiency to support precision agriculture applications.

---

## Dataset

The experiments were conducted on the **Paddy Doctor dataset**, which contains **10,407 rice leaf images** across **10 classes**:

* Bacterial Leaf Blight
* Bacterial Leaf Streak
* Bacterial Panicle Blight
* Blast
* Brown Spot
* Dead Heart
* Downy Mildew
* Hispa
* Normal
* Tungro

The dataset was split into **8,325 training images** and **2,082 validation images** using stratified sampling.

---

## Methodology

Three learning paradigms were evaluated:

* **Classical Machine Learning:** HOG, LBP, and HSV features with SVM, Random Forest, XGBoost, and LightGBM.
* **Deep Learning:** Transfer learning using ResNet34.
* **Hybrid Learning:** ResNet34 feature embeddings combined with traditional machine learning classifiers.

---

## Experimental Setup

* **Backbone:** ResNet34
* **Optimizer:** AdamW
* **Batch Size:** 64
* **Epochs:** 20
* **Input Size:** 224 × 224
* **Hardware:** NVIDIA Tesla T4 GPU

---

## Results

| Method                | Accuracy (%) | Macro-F1 (%) |
| --------------------- | ------------ | ------------ |
| ResNet34              | **97.60**    | **97.47**    |
| HOG + LBP + HSV → SVM | 91.88        | 92.10        |
| ResNet34 → SVM        | 91.79        | 91.38        |

The results demonstrate that **end-to-end transfer learning with ResNet34 outperforms both classical and hybrid approaches**, while handcrafted feature-based SVM remains a strong lightweight alternative.

---

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* OpenCV
* NumPy
* Pandas
* Matplotlib
* XGBoost
* LightGBM
* Google Colab

---

## Future Work

* Evaluate models on real-world field data
* Integrate explainable AI techniques such as Grad-CAM
* Explore lightweight models for edge deployment
* Extend the framework to disease severity estimation
