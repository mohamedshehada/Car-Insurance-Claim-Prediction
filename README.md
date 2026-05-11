# Car Insurance Claim Prediction

---

## Project Overview

This project builds a machine learning classification model to predict whether a car insurance customer will file an insurance claim (`OUTCOME = 1`) or not (`OUTCOME = 0`).

Understanding which customers are likely to file claims allows insurance companies to:
- Set accurate, risk-based premium pricing
- Identify high-risk customer segments proactively
- Improve profitability through better underwriting decisions

---

## Dataset Description

| Property | Details |
|---|---|
| **Source** | Car Insurance Claim Dataset |
| **Target** | `OUTCOME` — 1 = Filed a Claim, 0 = No Claim |
| **One row represents** | A single insurance customer |
| **Total rows** | 10,000 |
| **Total features** | 17 (after dropping `ID` and `POSTAL_CODE`) |
| **Class balance** | ~69% No Claim / ~31% Claimed |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `AGE` | Ordinal | Age group of the customer |
| `GENDER` | Nominal | Gender |
| `RACE` | Nominal | Majority / Minority |
| `DRIVING_EXPERIENCE` | Ordinal | Years of driving experience |
| `EDUCATION` | Ordinal | Education level |
| `INCOME` | Ordinal | Income bracket |
| `CREDIT_SCORE` | Numeric | Credit score — has missing values (~9.8%) |
| `VEHICLE_OWNERSHIP` | Binary | Owns vehicle (1) or not (0) |
| `VEHICLE_YEAR` | Nominal | Before or after 2015 |
| `MARRIED` | Binary | Marital status |
| `CHILDREN` | Binary | Has children (1) or not (0) |
| `ANNUAL_MILEAGE` | Numeric | Miles driven per year — has missing values (~9.6%) |
| `VEHICLE_TYPE` | Nominal | Sedan or Sports Car |
| `SPEEDING_VIOLATIONS` | Numeric | Number of speeding violations |
| `DUIS` | Numeric | Number of DUI offenses |
| `PAST_ACCIDENTS` | Numeric | Number of past accidents |

---

## Methodology (CRISP-DM)

1. **Business Understanding** — Predict claim likelihood to support risk-based pricing
2. **Data Understanding** — EDA with univariate and multivariate visualizations
3. **Data Preparation** — Pipeline-based preprocessing with `ColumnTransformer`
4. **Modeling** — Random Forest Classifier (default baseline)
5. **Evaluation** — Accuracy, Classification Report, Confusion Matrix
6. **Feature Importance** — Top 10 features via permutation importance

---

## Preprocessing Pipeline

```
ColumnTransformer
├── Numeric  (CREDIT_SCORE, ANNUAL_MILEAGE, ...)          → SimpleImputer(median) → StandardScaler
├── Binary   (VEHICLE_OWNERSHIP, MARRIED, CHILDREN)       → SimpleImputer(most_frequent)
├── Ordinal  (AGE, DRIVING_EXPERIENCE, EDUCATION, INCOME) → OrdinalEncoder
└── Nominal  (GENDER, RACE, VEHICLE_YEAR, VEHICLE_TYPE)   → OneHotEncoder
          ↓
   RandomForestClassifier(random_state=42)
```

> All preprocessing is done **inside** the pipeline to prevent data leakage.

---

## Model Results

| Metric | Score |
|---|---|
| Train Accuracy | 1.00 (overfitting — next step: GridSearchCV tuning) |
| **Test Accuracy** | **0.81** |
| Top Feature #1 | `CREDIT_SCORE` (importance: 0.187) |
| Top Feature #2 | `DRIVING_EXPERIENCE` (importance: 0.129) |
| Top Feature #3 | `ANNUAL_MILEAGE` (importance: 0.107) |

---

## Top 10 Most Important Features

![Top 10 Features](top10_features.png)

### Do These Features Make Sense for the Business Case?

Yes — the top features align directly with how insurance companies price risk:

1. **Credit Score** — Lower credit scores correlate with higher claim risk, reflecting overall financial responsibility.
2. **Driving Experience** — Less experienced drivers have more accidents and higher claim rates.
3. **Annual Mileage** — More miles driven increases exposure and the likelihood of incidents.
4. **Vehicle Ownership** — Owners tend to be more cautious than non-owners.
5. **Age** — Younger drivers are statistically higher risk; very elderly drivers may also show elevated rates.
6. **Income** — Influences vehicle type and driving behavior, which in turn affects claim probability.
7. **Speeding Violations** — Risky driving behavior is a direct predictor of future claims.
8. **Past Accidents** — A customer's accident history is one of the strongest predictors of future claims.
9. **Education** — Correlates indirectly with risk awareness and driving behavior.
10. **Vehicle Year (before 2015)** — Older vehicles are less safe and more prone to mechanical failures.

---

## Explanatory Visualizations

### 1. Credit Score vs. Claim Outcome

![Average Credit Score by Claim Outcome](Average_Credit_Score_by_Claim_Outcome.png)

**Insight:** Customers with a **Very Low credit score (below 0.30)** file insurance claims at a rate of **59.4%** — more than six times higher than customers with a **Very High credit score (above 0.80)**, who file at only **9.6%**. Credit score is the single strongest predictor of claim risk in this dataset.

**Business Recommendation:** Use credit score as a primary pricing variable. Apply higher premiums for scores below 0.50 and offer competitive discounts above 0.80 as a retention strategy.

---

### 2. Driving Experience vs. Claim Rate

![Driving Experience](explanatory_driving_experience.png)

**Insight:** **Beginner drivers (0–9 years)** file claims at **62.8%** — over 30 times higher than **veterans with 30+ years of experience**, who file at just **1.9%**. This steep drop-off confirms that driving experience is one of the most actionable variables for premium setting.

**Business Recommendation:** Beginner drivers should be the highest premium tier. Veterans represent the lowest-risk segment and should be retained with competitive pricing and loyalty incentives.

---

## Repository Structure

```
Car-Insurance-Claim-Prediction/
│
├── Car_Insurance_Claim_Classification.ipynb   # Main project notebook
├── Car_Insurance_Claim.csv                    # Dataset
├── top10_features.png                         # Feature importance chart
├── explanatory_credit_score.png               # Explanatory visual 1
├── explanatory_driving_experience.png         # Explanatory visual 2
└── README.md                                  # This file
```

---

## How to Run

1. Clone this repository
2. Install requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Car_Insurance_Claim_Classification.ipynb` in JupyterLab or Google Colab
4. Run all cells in order

---

## Author

**Mohammed Shehada**
Computer Engineering Graduate | Data Science & ML Student @ AXSOSACADEMY
GitHub: [@mohamedshehada](https://github.com/mohamedshehada)
