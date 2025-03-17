# Knowledge Distillation with PyTorch (MNIST Classification)
This work is inspired by the paper [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).


This repository implements **Knowledge Distillation** in PyTorch using the **MNIST dataset**. The goal is to train a **Student model** to achieve similar accuracy to a **Teacher model** while being significantly smaller and more efficient.

## Overview

* A **Teacher model** (larger network) is trained on MNIST.
* A **Student model** (smaller network) learns from both the **Teacher's soft labels** and the ground-truth labels using **Distillation Loss**.
* The **Student model** achieves **~97.68% accuracy**, nearly matching the **Teacher's 97.69% accuracy**, with **73.5% fewer parameters**.

## Model Architectures

| Model | Architecture (Fully Connected Layers) | Parameters | Accuracy (%) |
|-------|--------------------------------------|------------|--------------|
| **Teacher** | FC(784 → 1200) → FC(1200 → 1200) → FC(1200 → 10) | **2,395,210** | **97.69%** |
| **Student** | FC(784 → 800) → FC(800 → 10) | **636,010** (**73.5% smaller**) | **97.68%** |




### Train the Student Model with Distillation
* Learns from **soft labels** (teacher's predictions) and **ground-truth labels**.
* Uses **Distillation Loss**: $\text{Loss} = \alpha \cdot \text{KL-Divergence} + (1 - \alpha) \cdot \text{CrossEntropyLoss}$
* **Temperature (T=5.0)** controls softening of teacher logits.
* **Alpha (α=0.7)** balances hard vs. soft targets.

## Results

### Performance Comparison

| Model | Accuracy (%) | Parameters | Size Reduction (%) |
|-------|--------------|------------|---------------------|
| **Teacher** | **97.69%** | **2,395,210** | - |
| **Student** | **97.68%** | **636,010** | **73.5% smaller than teacher** |

* **Minimal Accuracy Drop (~0.01%)**
* **4x Reduction in Model Size**


## References

* Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
