# ♻️ Waste Classification using MobileNetV2 + MixUp

This project contains code to train and evaluate a deep learning model for classifying waste into **metal**, **paper**, and **plastic** using **MobileNetV2**, **MixUp data augmentation**, and **transfer learning**. The model is designed for smart recycling systems and is compatible with low-power edge devices like the Raspberry Pi.

---

## 📁 Files Included

- `train_model.py` – Full training script that includes:
  - Data augmentation
  - MixUp regularization
  - Class weighting to handle imbalance
  - Two-phase training: frozen base + fine-tuning
  - Model saving

- `evaluate_model.py` – Evaluation script that:
  - Loads the trained model
  - Calculates test accuracy
  - Prints classification report
  - Displays confusion matrix using seaborn

---

## ✅ Final Test Performance

- **Accuracy:** 95.03%
- **Precision / Recall / F1-Score:** ~95% (weighted average)

---


## 📦 Requirements

Install the required Python packages before running the scripts:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## 👩‍💻 Author
Dunia Aljafare,
Smart Systems Engineering,
Palestine Ahliya University – Graduation Project 2025
