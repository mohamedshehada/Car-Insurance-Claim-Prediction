# 🚗 Car Insurance Claim Prediction

A complete Machine Learning project that predicts whether a customer will file a car insurance claim.

The project progresses through three stages:

1. Exploratory Data Analysis (EDA) & Random Forest Baseline
2. Feature Engineering & Feature Selection
3. Deep Learning with Hyperparameter Optimization (Keras Tuner)
 
---

# 📁 Project Structure

```text
├── Car_Insurance_Claim_Classification.ipynb
├── Car_Insurance_Claim.csv
├── README.md
├── Distribution of Claims.png
├── Claim Rate by Driving Experience.png
├── Average Credit Score by Claim Outcome.png
├── claim_rate_by_age.png
├── vehicle_type_vs_claim.png
├── correlation_heatmap.png
└── top10_features.png
```

---

# 📊 Dataset

- **Size:** 10,000 customers × 19 features
- **Target Variable:** `OUTCOME`
  - 1 = Customer filed a claim
  - 0 = No claim filed

### Feature Categories

| Type | Features |
|--------|----------|
| Numeric | CREDIT_SCORE, ANNUAL_MILEAGE, SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS |
| Binary | VEHICLE_OWNERSHIP, MARRIED, CHILDREN |
| Ordinal | AGE, DRIVING_EXPERIENCE, EDUCATION, INCOME |
| Nominal | GENDER, RACE, VEHICLE_YEAR, VEHICLE_TYPE, POSTAL_CODE |

---

# 📈 Exploratory Data Analysis (EDA)

## Distribution of Claims

![Distribution of Claims](Distribution%20of%20Claims.png)

### Insight
The dataset is moderately imbalanced — 68.67% non-claimants vs 31.33% claimants. This motivated the use of stratified sampling and `class_weight='balanced'` during modeling.

---

## Claim Rate by Driving Experience

![Claim Rate by Driving Experience](Claim%20Rate%20by%20Driving%20Experience.png)

### Insight
Drivers with 0–9 years of experience show a claim rate above 60%, dropping sharply as experience increases. Driving experience is one of the strongest predictors in the dataset.

---

## Claim Rate by Age

![Claim Rate by Age](claim_rate_by_age.png)

### Insight
The 16–25 age group shows the highest claim rate, consistent with inexperienced drivers taking more risks on the road.

---

## Average Credit Score by Claim Outcome

![Average Credit Score by Claim Outcome](Average%20Credit%20Score%20by%20Claim%20Outcome.png)

### Insight
Customers who did not file a claim have noticeably higher average credit scores (~0.52) compared to those who did (~0.44), suggesting a link between financial responsibility and driving risk.

---

## Correlation Heatmap

![Correlation Heatmap](correlation_heatmap.png)

### Insight
CREDIT_SCORE (−0.33) and VEHICLE_OWNERSHIP (−0.38) show the strongest negative correlations with OUTCOME among numeric features. SPEEDING_VIOLATIONS and PAST_ACCIDENTS are positively correlated with each other (0.44) and with claim risk.

---

# 🧩 Part 1 — EDA, Preprocessing & Random Forest Baseline

## Objectives

- Data cleaning and preprocessing
- Pipeline construction
- Baseline Random Forest model
- Feature importance analysis

## Preprocessing Pipeline

| Feature Type | Method |
|--------------|---------|
| Numeric | Median Imputation + StandardScaler |
| Binary | Most Frequent Imputation |
| Ordinal | OrdinalEncoder (natural order) |
| Nominal | OneHotEncoder |

## Model Performance

| Dataset | Accuracy |
|----------|----------|
| Train | 99.92% |
| Test | 82.05% |

### Observation
The baseline Random Forest overfit the training data (99.92% train vs 82.05% test). This is expected behavior for a default Random Forest without depth constraints — it memorizes training patterns rather than generalizing.

---

## Top 10 Feature Importances

![Top 10 Features](top10_features.png)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | CREDIT_SCORE | 0.169 |
| 2 | DRIVING_EXPERIENCE | 0.124 |
| 3 | ANNUAL_MILEAGE | 0.096 |
| 4 | AGE | 0.081 |
| 5 | VEHICLE_OWNERSHIP | 0.079 |
| 6 | SPEEDING_VIOLATIONS | 0.069 |
| 7 | INCOME | 0.061 |
| 8 | POSTAL_CODE | 0.052 |
| 9 | PAST_ACCIDENTS | 0.050 |
| 10 | VEHICLE_YEAR_after 2015 | 0.037 |

### Key Business Insights

- **Credit Score** is the top predictor — lower scores correlate with higher claim risk, reflecting financial responsibility patterns.
- **Driving Experience** is the strongest behavioral predictor — inexperienced drivers file significantly more claims.
- **Annual Mileage** increases exposure to accidents directly.
- **POSTAL_CODE** appears in the top 10, confirming that geographic region carries meaningful risk signal (claim rates vary from 27% to 100% across the 4 postal regions).

---

# ⚙️ Part 2 — Feature Engineering & Feature Selection

## Objectives

- PCA Feature Extraction
- K-Means Clustering
- Embedded Feature Selection
- Model Comparison

## Feature Engineering

| Method | Purpose |
|----------|----------|
| PCA (3 Components) | Dimensionality Reduction — captures 58.8% of total variance |
| K-Means (4 Clusters) | Customer Risk Segmentation |

Feature space expanded from:

```text
24 Original Features
↓
+ 3 PCA Features
+ 1 Cluster Feature
↓
28 Features
```

---

## Feature Selection

Using:

```python
SelectFromModel(RandomForestClassifier(), threshold='median')
```

Feature count reduced:

```text
28 Features → 14 Selected Features
```

Selected features: CREDIT_SCORE, ANNUAL_MILEAGE, SPEEDING_VIOLATIONS, PAST_ACCIDENTS, VEHICLE_OWNERSHIP, AGE, DRIVING_EXPERIENCE, INCOME, VEHICLE_YEAR_after 2015, VEHICLE_YEAR_before 2015, PCA_1, PCA_2, PCA_3, KMeans_Cluster

---

## Model Comparison

| Model | Features | Test Accuracy | Recall (class 1) |
|---------|---------|---------|---------|
| Baseline RF | 24 | 82.40% | 0.71 |
| RF + PCA + KMeans | 28 | 83.40% | 0.72 |
| RF + Feature Selection | 14 | 80.50% | 0.67 |

### Conclusion

Feature engineering (PCA + KMeans) actually improved accuracy by ~1% over baseline. Feature selection halved the feature count from 28 to 14 with only a modest drop in accuracy (from 83.4% to 80.5%) — a good trade-off for model simplicity and inference speed.

---

# 🧠 Part 3 — Neural Network & Hyperparameter Tuning

## Base Neural Network

```text
Input Layer   (24 features)
      ↓
Dense (30, ReLU)
      ↓
Dropout (0.3)
      ↓
Dense (1, Sigmoid)
```

### Configuration

- Loss: Binary Crossentropy
- Optimizer: Adam
- Learning Rate: 0.001
- Early Stopping: Patience = 5
- Training stopped at epoch 32

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 84.85% |
| Loss | 0.3414 |
| Precision (class 1) | 0.75 |
| Recall (class 1) | 0.77 |
| F1 Score (class 1) | 0.76 |

---

## Hyperparameter Tuning

### Search Space

| Parameter | Values |
|------------|----------|
| Units | 32, 64, 128, 256 |
| Dropout | 0.1 – 0.5 |
| Optimizer | Adam, RMSprop, SGD |
| Learning Rate | 0.0001, 0.001, 0.01 |

### Best Hyperparameters

| Parameter | Best Value |
|------------|------------|
| Units | 64 |
| Dropout | 0.5 |
| Optimizer | RMSprop |
| Learning Rate | 0.01 |

---

## Final Model Comparison

| Metric | Base NN | Tuned NN |
|----------|----------|----------|
| Accuracy | 84.85% | 84.40% |
| Precision | 0.75 | 0.73 |
| Recall | 0.77 | 0.80 |
| F1 Score | 0.76 | 0.76 |

### Recommendation

Although accuracy decreased slightly, the tuned model achieved **higher recall (0.80 vs 0.77)** — it catches more actual claimants. For an insurance use case, recall is the most critical metric because missing a true claimant is far more costly than a false alarm.

---

# 🏆 Final Results

| Model | Features | Test Accuracy | Recall (class 1) |
|----------|----------|----------|----------|
| RF Baseline | 24 | 82.05% | 0.70 |
| RF Balanced (Part 2) | 24 | 82.40% | 0.71 |
| RF + PCA + KMeans | 28 | 83.40% | 0.72 |
| RF + Feature Selection | 14 | 80.50% | 0.67 |
| Base Neural Network | 24 | 84.85% | 0.77 |
| **Tuned Neural Network** | 24 | **84.40%** | **0.80** |

✅ **Recommended Deployment Model:** Tuned Neural Network

Reason:
- Highest Recall (0.80) — best at identifying high-risk customers
- Strong generalization on unseen test data
- Compact architecture (64 units, 1 hidden layer)

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow / Keras
- Keras-Tuner
- Google Colab

---

# ▶️ How to Run

1. Open the notebook in Google Colab.
2. Upload `Car_Insurance_Claim.csv`.
3. Run all notebook cells.
4. The notebook automatically installs `keras-tuner`.

---

# 👤 Author

**Mohamed Shehada**

Machine Learning Project – AXSOS Academy
