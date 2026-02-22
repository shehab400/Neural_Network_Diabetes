# Neural Network for Pima Indians Diabetes Prediction

## Problem Statement

This project applies a deep learning neural network to predict diabetes onset using the **Pima Indians Diabetes Dataset** from Kaggle. The dataset contains medical diagnostic measurements for 768 female patients of Pima Indian heritage, and the goal is to classify whether a patient is diabetic (1) or non-diabetic (0) based on 8 clinical features.


## Dataset

- **Source**: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768
- **Features**: 8 numeric medical predictors
- **Target**: Binary (0 = No Diabetes, 1 = Diabetes)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2h oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic influence score) |
| Age | Age in years |

## Approach

The project follows a complete end-to-end machine learning pipeline:

1. **Data Loading & Exploration**: Load the CSV dataset, inspect class distribution, and examine feature value ranges.
2. **Outlier Removal**: Apply Z-score analysis (threshold = 3) to remove extreme values that could skew model training.
3. **Train-Test Split**: 80/20 stratified split to preserve class proportions.
4. **Feature Scaling**: StandardScaler normalization (fit on training data only) to bring all features to zero mean and unit variance.
5. **Model Architecture**: A Sequential neural network with 3 hidden layers (64 → 32 → 16 neurons), `tanh` activation, Dropout regularization (30%, 20%, 20%), and a `sigmoid` output layer.
6. **Training**: Adam optimizer (lr=0.0005), binary crossentropy loss, Early Stopping (patience=20) with best weight restoration, up to 200 epochs.
7. **Evaluation**: Test accuracy, confusion matrix, precision, recall, and F1-score analysis.

## Model Architecture

```
Layer (type)              Output Shape    Param #
─────────────────────────────────────────────────
Dense (64, tanh)          (None, 64)      576
Dropout (0.3)             (None, 64)      0
Dense (32, tanh)          (None, 32)      2,080
Dropout (0.2)             (None, 32)      0
Dense (16, tanh)          (None, 16)      528
Dropout (0.2)             (None, 16)      0
Dense (1, sigmoid)        (None, 1)       17
─────────────────────────────────────────────────
Total params: 3,201
```

## Results

- **Expected Test Accuracy**: 78.8%
- The Pima Diabetes dataset is a challenging real-world dataset with class imbalance and noisy features, making it harder than many benchmark datasets.
- Detailed metrics (precision, recall, F1-score) and visualizations are available in the notebook and the `results/` folder.

## Structure

```
    /
├── pima_diabetes_nn.ipynb      # Complete notebook with all steps
├── README.md                   # This file
└── results/
    ├── training_curves.png     # Accuracy & loss plots
    ├── confusion_matrix.png    # Confusion matrix heatmap
    └── metrics_summary.txt     # Classification metrics summary
```

