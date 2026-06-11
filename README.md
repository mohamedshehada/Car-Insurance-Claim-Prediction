# 🚗 Car Insurance Claim Prediction

A complete machine learning project that predicts whether a customer will file a car insurance claim. Built in three progressive parts — from classical ML to deep learning with hyperparameter optimization.

---

## 📁 Project Structure

```
Car_Insurance_Claim_Prediction_COMPLETE.ipynb   ← Full merged notebook (all 3 parts)
README.md
```

---

## 📊 Dataset

- **Size:** 10,000 customers × 19 features
- **Target:** `OUTCOME` — binary (1 = filed a claim, 0 = no claim)
- **Class balance:** ~69% No Claim / ~31% Claim (handled via stratified split + `class_weight="balanced"`)

### Feature Types

| Type | Features |
|---|---|
| Numeric | `CREDIT_SCORE`, `ANNUAL_MILEAGE`, `SPEEDING_VIOLATIONS`, `DUIS`, `PAST_ACCIDENTS` |
| Binary | `VEHICLE_OWNERSHIP`, `MARRIED`, `CHILDREN` |
| Ordinal | `AGE`, `DRIVING_EXPERIENCE`, `EDUCATION`, `INCOME` |
| Nominal | `GENDER`, `RACE`, `VEHICLE_YEAR`, `VEHICLE_TYPE`, `POSTAL_CODE` |

---

## 🧩 Part 1 — EDA, Preprocessing & Random Forest Baseline

### Goals
- Load, explore, and clean the dataset
- Build a full preprocessing pipeline (no data leakage)
- Train a Random Forest baseline model
- Analyze feature importance

### Preprocessing Pipeline

| Feature Type | Transformer |
|---|---|
| Numeric | `SimpleImputer(median)` → `StandardScaler` |
| Binary | `SimpleImputer(most_frequent)` |
| Ordinal | `OrdinalEncoder` with explicit category order |
| Nominal | `OneHotEncoder` |

### EDA Highlights
- **Driving experience** and **credit score** show the strongest separation between claimants and non-claimants
- Young drivers (16–25) have significantly higher claim rates
- Older vehicles (before 2015) correlate with more claims
- Speeding violations and past accidents are strong risk indicators

### Results

| Set | Accuracy |
|---|---|
| Train | **1.00** (overfitting — default RF with no depth limit) |
| Test | **0.81** |

### Top 10 Features (Random Forest Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `DRIVING_EXPERIENCE` | 0.1873 |
| 2 | `CREDIT_SCORE` | 0.1286 |
| 3 | `ANNUAL_MILEAGE` | ~0.10 |
| 4 | `VEHICLE_OWNERSHIP` | ~0.08 |
| 5 | `AGE` | ~0.07 |
| 6 | `INCOME` | ~0.06 |
| 7 | `SPEEDING_VIOLATIONS` | ~0.05 |
| 8 | `PAST_ACCIDENTS` | ~0.05 |
| 9 | `EDUCATION` | ~0.04 |
| 10 | `VEHICLE_YEAR` (before 2015) | ~0.03 |

### Business Insights
- **Credit Score:** Lower scores → higher claim risk (reflects financial responsibility)
- **Driving Experience:** Less experience → more accidents → higher claims
- **Annual Mileage:** More miles = more exposure to risk
- **Past Accidents:** Strongest behavioral predictor of future claims

---

## ⚙️ Part 2 — Feature Engineering & Feature Selection

### Goals
- Apply PCA + KMeans feature engineering
- Apply embedded feature selection (`SelectFromModel`)
- Compare all model variants

### Key Design Decisions
- `POSTAL_CODE` reclassified as **nominal** (only 4 unique geographic regions) and one-hot encoded — shows meaningful variation in claim rates across regions
- `class_weight="balanced"` added to handle class imbalance

### Feature Engineering

| Method | What it adds |
|---|---|
| **PCA (3 components)** | Captures main directions of variance across all preprocessed features |
| **KMeans (4 clusters)** | Assigns each customer to a risk-segment group (optimal k=4 via Elbow Method) |

> ⚠️ No data leakage: PCA and KMeans fit **only on training data**, transformed on test set.

Feature space: **24 original → +3 PCA + 1 KMeans = 28 engineered features**

### Feature Selection (Embedded — `SelectFromModel`)
- Internal RF with `threshold='median'` keeps only features with importance ≥ median
- Reduced from **28 → 14 selected features**

### Model Comparison

| Model | Features | Test Accuracy |
|---|---|---|
| Baseline RF (Part 1 re-trained, balanced) | 24 | ~0.79 |
| Engineered RF (+ PCA + KMeans) | 28 | ~0.79 |
| Final RF (SelectFromModel) | 14 | ~0.79 |

> Feature engineering maintained accuracy while cutting feature count by 50% — a more efficient and interpretable model.

### Top 10 Features — Permutation Importance (Final Model)
`DRIVING_EXPERIENCE`, `CREDIT_SCORE`, `ANNUAL_MILEAGE`, `PAST_ACCIDENTS`, `AGE`, `SPEEDING_VIOLATIONS`, `INCOME`, `VEHICLE_OWNERSHIP`, `PCA_1`, `VEHICLE_TYPE`

---

## 🧠 Part 3 — Neural Network + Keras Tuner

### Goals
- Build a binary classification neural network (1 hidden layer)
- Train with Early Stopping
- Tune 4 hyperparameters with Keras Tuner (Hyperband)

### Architecture — Base Model

```
Input  →  Dense(30, ReLU)  →  Dropout(0.3)  →  Dense(1, Sigmoid)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (lr=0.001)
- **Metrics:** Accuracy, Recall, Precision
- **Early Stopping:** patience=5 on `val_accuracy`
- **Stopped at epoch 28** (saved 22 unnecessary epochs)

### Training History Observations
- Accuracy steadily increased; train/val stayed close → good generalization, no overfitting
- Loss decreased on both curves → Dropout prevented memorization
- Recall improved from ~0.44 → ~0.76 across epochs
- Precision stabilized around ~0.78

### Base Model Results (Test Set)

| Metric | Value |
|---|---|
| **Test Accuracy** | **84.70%** |
| Test Loss | 0.3423 |
| Precision (class 1) | 0.75 |
| Recall (class 1) | 0.78 |
| F1-Score (class 1) | 0.76 |

---

### Hyperparameter Tuning — Keras Tuner (Hyperband)

| Hyperparameter | Search Space |
|---|---|
| Units (hidden layer) | 32, 64, 128, 256 |
| Dropout Rate | 0.1 → 0.5 (step 0.1) |
| Optimizer | Adam, RMSprop, SGD |
| Learning Rate | 0.0001, 0.001, 0.01 |

- **Algorithm:** Hyperband (fast — eliminates poor trials early)
- **Trials:** 90
- **Time:** ~9 minutes 47 seconds
- **Best val_accuracy:** 86.25%

### Best Hyperparameters Found

| Hyperparameter | Base Model | Best Tuned |
|---|---|---|
| Units | 30 | **256** |
| Dropout Rate | 0.3 | **0.1** |
| Optimizer | Adam | **RMSprop** |
| Learning Rate | 0.001 | **0.001** |

> The tuner preferred **more capacity** (256 units) and **lighter regularization** (dropout 0.1) — the dataset benefits from a larger model.

### Best Tuned Model — Training
- Converged faster: **stopped at epoch 13** (vs 28 for base)

### Final Comparison

| Metric | Base NN | Tuned NN |
|---|---|---|
| Test Accuracy | **84.70%** | 84.40% |
| Test Loss | 0.3423 | 0.3486 |
| Precision (class 1) | 0.75 | 0.73 |
| **Recall (class 1)** | 0.78 | **0.80** |
| F1-Score (class 1) | 0.76 | 0.76 |

> The tuned model trades a tiny accuracy drop for **higher recall (0.80 vs 0.78)**. In insurance, recall is the critical metric — missing a real claimant is more costly than a false alarm.

---

## 🏆 Overall Project Summary

| Model | Test Accuracy | Recall (class 1) | Notes |
|---|---|---|---|
| RF Baseline — Part 1 | 81% | — | Overfits (train=100%), no class balancing |
| RF Baseline — Part 2 (balanced) | ~79% | ~0.74 | `class_weight="balanced"` added |
| RF + PCA + KMeans — Part 2 | ~79% | ~0.74 | Feature engineering, same accuracy |
| RF + Feature Selection — Part 2 | ~79% | ~0.74 | 14 features, same performance |
| Base Neural Network — Part 3 | **84.70%** | 0.78 | 1 hidden layer, Early Stopping |
| **Tuned Neural Network — Part 3** | 84.40% | **0.80** | Best recall — recommended for deployment |

**Best model for deployment:** The **Tuned Neural Network** — highest recall for catching actual claimants, which is the business-critical metric for insurance pricing and risk assessment.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn` (RF, PCA, KMeans, SelectFromModel, pipelines) |
| Deep Learning | `TensorFlow / Keras` |
| Hyperparameter Tuning | `keras-tuner` (Hyperband) |
| Environment | Google Colab |

---

## ▶️ How to Run

1. Open `Car_Insurance_Claim_Prediction_COMPLETE.ipynb` in **Google Colab**
2. Mount your Google Drive and update the `file_path` variable to point to your `Car_Insurance_Claim.csv`
3. Run all cells top to bottom (**Runtime → Run All**)
4. `keras-tuner` is installed automatically via `!pip install -q keras-tuner` at the start of Part 3

---

## 👤 Author

**Mohamed Shehada**  
Intermediate Machine Learning Project — AXSOS Academy
