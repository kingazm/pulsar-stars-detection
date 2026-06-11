# Pulsar Star Detection

This project develops a Multi-Layer Perceptron (MLP) from scratch in PyTorch to classify astronomical signals as either genuine pulsars or background noise, addressing class imbalance and missing data through robust preprocessing techniques.

## Overview

Pulsars are a rare type of neutron star that emit detectable radio waves. Identifying these from vast amounts of observational data is crucial for astronomical research but challenging due to significant noise and interference. This project tackles this as a binary classification problem using a dataset of statistical features derived from candidate signals.

## Dataset

The dataset comprises 12,528 labelled training samples and 5,370 unlabelled test samples. Each candidate has eight continuous features. A key characteristic is the strong class imbalance, with only about 9.2% of training samples being genuine pulsars.

<img width="515" height="411" alt="image" src="https://github.com/user-attachments/assets/8dad69af-6a75-45de-82dc-be46d849ac6f" />

## Preprocessing

The preprocessing pipeline includes:
- **Data Cleaning**: Stripping white spaces from column names.
- **Train/Eval Split**: Classic 80/20 split with stratification to maintain class distribution.
- **Missing Value Imputation**: Using `IterativeImputer` to estimate missing values based on feature correlations.
- **Feature Scaling**: Applying `RobustScaler` due to heavy tails and outliers in features, which makes it more robust than `StandardScaler` or `MinMaxScaler`.
- **Class Imbalance Handling**: Using a `pos_weight` in the loss function to penalize missed pulsars more heavily.

## Model Architecture

The model is a Multi-Layer Perceptron (MLP) with two hidden layers, built using PyTorch. Key aspects include:
- **Depth and Width**: Two hidden layers of 16 neurons each, balancing capacity without overfitting.
- **Activation Function**: ReLU for hidden layers, chosen for efficiency and vanishing gradient prevention.
- **Batch Normalisation**: `BatchNorm1d` applied after each linear layer and before activation to stabilize training.
- **Dropout**: A dropout rate of 0.2 after activations for light regularization.
- **Output Layer**: A single raw logit, with sigmoid applied internally by `BCEWithLogitsLoss` for numerical stability.

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/edd51c3a-c93c-485c-a9ac-fd06314477dc" />

Multiple experiments condicted were conducted to determine the best architecture and hyperparameters, as described in detail in the project report (overleaf) linked in the notebook.

## Training

- **Loss Function**: `nn.BCEWithLogitsLoss` with `pos_weight` to handle class imbalance.
- **Optimizer**: `Adam` with a learning rate of 0.001.
- **Learning Rate Scheduler**: `ReduceLROnPlateau` to dynamically adjust the learning rate.
- **Early Stopping**: To prevent overfitting and save the best model based on validation loss.

## Model Performance

The model demonstrates strong performance, especially on F1-Score (88.79%) and ROC-AUC (97.86%), which are more indicative metrics for imbalanced datasets than raw accuracy (96.69%).

<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/cfb78cf4-2512-48c6-a135-0576372a9b30" />
<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/d7bbf1fe-66c4-4144-9a98-aec7b36aa88a" />

*Originally developed in Google Colab environment.*

Authors: Kinga Żmuda (@kingazm), Wiktor Godyń (@Budyn13441)
