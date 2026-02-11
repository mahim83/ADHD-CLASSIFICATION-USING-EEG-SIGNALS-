# ADHD Classification Using EEG Signals

## Author
Mahim Katiyar (102303958)  
Thapar Institute of Engineering and Technology, Patiala  

---

## Project Overview

This project focuses on detecting Attention Deficit Hyperactivity Disorder (ADHD) using EEG (Electroencephalogram) signals through Machine Learning and Deep Learning techniques.

Traditional ADHD diagnosis relies on behavioral observation, which can be subjective and time-consuming. This project provides a data-driven and objective approach by analyzing brain signal patterns using computational models.

---

## Dataset Description

- Total Participants: 121  
  - 61 ADHD  
  - 60 Healthy Controls  
- EEG Channels: 19 electrodes  
  (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz)  
- Sampling Rate: 128 Hz  
- Total Records: ~2.16 million rows  
- Total Columns: 21 (19 EEG channels + ID + Class)

The EEG signals were recorded during a visual attention task using the 10–20 international electrode placement system.

---

## Problem Statement

Manual interpretation of EEG signals is challenging due to their high dimensionality and complex non-linear patterns.

The objective of this project is to develop an automated classification system that can distinguish between ADHD and non-ADHD subjects using Machine Learning and Deep Learning models.

---

## Project Workflow

### 1. Data Preprocessing

- Checked and handled missing values  
- One-hot encoded categorical columns (ID and Class)  
- Defined:
  - X (Features): 19 EEG channel readings  
  - y (Target): Class_ADHD  
- Performed 80–20 stratified train-test split  
- Applied StandardScaler (fitted only on training data to prevent data leakage)

---

### 2. Models Implemented

#### Logistic Regression
- Used as a baseline linear classifier  
- PCA (95% variance retention) was tested  
- Performance remained similar after PCA  
- Limited in handling non-linear EEG patterns  

#### Random Forest
- Ensemble learning method using bagging  
- Captures non-linear relationships  
- Reduced overfitting compared to single decision trees  

#### Hybrid LSTM + CNN Model
- CNN layers extract spatial dependencies across EEG channels  
- LSTM layers capture temporal dependencies in EEG sequences  
- Input reshaped into 3D tensors (samples, time_steps, features)  
- Implemented using TensorFlow / Keras  

---

## Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|----------|--------|----------|---------|
| Logistic Regression  | 0.50     | 0.56     | 0.50   | 0.53     | 0.50    |
| Random Forest        | 0.72     | 0.72     | 0.82   | 0.77     | 0.77    |
| LSTM + CNN           | 0.77     | 0.76     | 0.85   | 0.80     | 0.84    |

---

## Key Observations

- Logistic Regression performed close to random guessing due to the non-linear nature of EEG signals.
- Random Forest significantly improved performance by capturing complex feature interactions.
- The LSTM + CNN model achieved the best results:
  - 76.9% Accuracy
  - 0.84 ROC-AUC
  - High ADHD Recall (0.85)

Deep learning models proved more suitable for modeling temporal and spatial EEG dependencies.

---

## Evaluation Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC Curve & AUC  
- Confusion Matrix  

Since the dataset is slightly imbalanced, ROC-AUC was considered a more reliable metric than accuracy alone.

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Seaborn  

---

## Key Learnings

- Importance of preventing data leakage (train-test split before scaling)  
- Linear models struggle with complex biomedical signals  
- Deep learning effectively captures temporal-spatial EEG patterns  
- Proper evaluation metrics are critical in medical AI applications  

---

## Future Improvements

- Hyperparameter tuning  
- Cross-subject generalization  
- Attention-based deep learning models  
- Improved model interpretability  

---

## Conclusion

This project demonstrates that EEG-based computational modeling can assist in objective ADHD classification. The hybrid LSTM + CNN model outperformed traditional ML approaches, highlighting the effectiveness of deep learning in handling high-dimensional biomedical time-series data.
