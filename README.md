# EMG Spectrogram Classification — Latin Alphabet Fingerspelling

Classification of the 26 letters of the fingerspelling (dactylology) alphabet from **sEMG + IMU signals**, converted into **Mel spectrograms** and classified with **fine-tuned CNNs** (InceptionV3, ResNet50, ConvNeXt).

**Best result:** InceptionV3 reached **82% accuracy and 0.82 macro F1** across 26 classes, up from 56% after adding synthetic data.

> Thesis project — Electronic and Telecommunications Engineering, Universidad del Cauca (2025).

---

## Problem

Fingerspelling is used to spell names and terms that have no dedicated sign. Camera-based recognition depends on lighting and image quality. Wearable EMG sensors avoid those limits, so this project tests whether muscle-activity signals are enough to recognize each letter.

## Dataset

| Property | Value |
|---|---|
| Samples | 780 (26 letters × 30 recordings) |
| Alphabet | Italian fingerspelling (Latin alphabet) |
| Sensor | Myo armband: 8 EMG channels + IMU (accelerometer, gyroscope, orientation) |
| Sampling rate | 200 Hz |
| Sample shape | 400 × 8 (time × EMG channel) per recording |
| Format | JSON, one file per recording |

> The raw data is not included in this repository. Source: *[https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet]*

## Pipeline

```
EMG + IMU (JSON) → Z-score normalization → PCA fusion to 1D signal
      → Mel spectrogram (350×350 px) → Fine-tuned CNN → Letter (A–Z)
```

### 1. Dimensionality reduction (PCA)
All synchronized channels are fused into a single 1D signal using three strategies:

| Approach | Method |
|---|---|
| **1** | Projection onto the first principal component (max variance baseline) |
| **2** | Variance-weighted sum of the first K components (95% explained variance) |
| **3** | Local PCA over sliding windows (adapts to non-stationary dynamics) |

Approach 1 was rendered with two colormaps (*viridis* and *jet*), giving **4 spectrogram datasets** in total.

### 2. Spectrogram generation
- STFT + Mel filter bank, log-power scale
- Hann window of 32 samples (0.16 s), 50% overlap, NFFT = 256
- Clean 350×350 px images, one folder per letter
- Intra- and inter-class similarity checked with Pearson correlation and SSIM

### 3. Model training (transfer learning)
- Pretrained ImageNet weights; final layer replaced with a 26-class head
- Only the last convolutional stage and the classifier are unfrozen
- Adam optimizer (lr 1e-3 for the new head, 1e-4 for fine-tuned layers), dropout 0.3
- Early stopping (patience 3), stratified 80/20 split with a fixed seed

### 4. Evaluation
- **Supervised:** accuracy, recall, F1-score, confusion matrix
- **Embedding quality:** Silhouette, Davies–Bouldin and Fowlkes–Mallows indices on last-layer features (PCA + KNN)
- **Statistical test:** Kruskal–Wallis across the 12 architecture × dataset configurations

## Results

### Architecture comparison
| Model | Performance | Clustering (Approach 2) |
|---|---|---|
| **InceptionV3** | Best on accuracy, recall and F1 across all datasets | Silhouette 0.001 · DB 2.02 · FM 0.45 |
| ResNet50 | Intermediate; most sensitive to the PCA approach | Silhouette −0.17 · DB 3.48 · FM 0.12 |
| ConvNeXt | Underfitted with the available data | Silhouette −0.21 · DB 4.57 · FM 0.16 |

**Kruskal–Wallis test:** the architecture has a statistically significant effect on performance (p ≈ 0.007). The spectrogram variant does not (p > 0.94). Model choice matters more than the PCA preprocessing.

### Synthetic data augmentation
With only 30 samples per class, the best configuration (InceptionV3 + Approach 2) was retrained with **180 synthetic samples per class**. Each synthetic sample is a linear interpolation (α = 0.5) of two random real training signals.

| Dataset | Samples per class | Accuracy | Macro F1 |
|---|---|---|---|
| Original | 30 | 56% | — |
| **Original + synthetic** | 210 | **82%** | **0.82** |

- **Best-classified letters:** D, J, L, Y, Z (F1 ≥ 0.91). These have very distinctive hand shapes or motion trajectories.
- **Hardest letters:** X (F1 0.61), V (0.68), Q (0.70). These have similar shapes or low muscle activation.


## Key takeaways
- Pretrained image CNNs transfer well to EMG spectrograms when the time–frequency representation is designed carefully.
- InceptionV3's multi-scale parallel convolutions fit spectrograms better than purely sequential architectures.
- The main bottleneck was data volume: synthetic augmentation raised accuracy by 26 points.

## Limitations and future work
- Data from a **single user** and one sign-language alphabet (Italian), so multi-user generalization is untested.
- Next steps: compare STFT vs. CWT vs. wavelet scattering, test CNN-LSTM hybrids for real-time use, and try generative models for synthetic data.

## Tech stack
Python · PyTorch · scikit-learn · NumPy · Pandas · Matplotlib · MATLAB (spectrogram generation) · Jupyter Notebook

## Repository structure
```
├── CNNs_with_4_differents_datasets.ipynb   # Training and evaluation of the three CNNs
└── README.md
```

## Authors
- **Maria Paula Guaca Campo** — [LinkedIn](https://linkedin.com/in/mariapaulaguacacampo)
- Carlos Andrés Hurtado Bedoya

Advisor: MSc. María Manuela Silva Zambrano. GNTT Research Group, Universidad del Cauca.
