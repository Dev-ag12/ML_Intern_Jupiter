# Credit Score Movement Prediction

## Project Overview
This project simulates a realistic dataset and builds a machine learning model to predict whether a customer's credit score will increase, decrease, or remain stable in the next 3 months.

## Project Structure
```
ML_Intern_Jupiter/
├── notebooks/
│   ├── 01_data_generation.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── data_generator.py
│   │   └── feature_engineering.py
|   |   |__credit_score_dataset.csv
|   |   |__ processed_dataset.csv
│   └── models/
│       ├── model_trainer.py
│       └── model_explainer.py
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
# Synthetic Credit-Score Movement Dataset Generator

## Project overview
This repository provides a single class `DataGenerator` that creates a 25 000-row synthetic dataset modelling individual credit behaviour over one calendar year.  
The dataset is ready for exploratory data analysis or model training and already includes the multiclass target label `target_credit_score_movement` (`increase`, `decrease`, `stable`).

```python
from data_generator import DataGenerator

gen = DataGenerator(n_customers=25_000,
                    start_date='2023-01-01',
                    months=12)
df = gen.generate_dataset()
```

## Dataset Generation

### Feature-generation logic and key assumptions 

#### Demographics

- Customer IDs follow the pattern CUST_00001 … CUST_25000.
- Ages are drawn from a normal distribution centred at 35 years (σ = 10).
- Gender split is 52 % male, 48 % female.
- Location is sampled as 50 % Urban, 30 % Suburban, 20 % Rural.

#### Financial metrics

- Base annual income depends on location
   - Urban 120 k ₹, Suburban 90 k ₹, Rural 60 k ₹.

- A gender multiplier (M 1.10, F 0.90) adjusts the mean income.

- Monthly income is sampled from a log-normal distribution with σ = 0.30.

- Monthly EMI outflow is a uniform 20 – 35 % of that income.

- Total credit limit is 2 – 4 × the monthly income.

- Current outstanding is 30 – 80 % of the limit (Beta(2, 5)); from this the credit-utilisation ratio is derived.

#### Credit-history metrics

- Days past due in the last three months follow a Poisson(λ = 2).

- Months since last default: Poisson with mean inversely related to DPD (fewer DPD → longer since default).

- Hard enquiries in the last six months: Poisson(0.5 + 5 × utilisation).

- A repayment-history score (0–100) is built from a weighted blend of
- DPD (40 %), months-since-default (30 %), utilisation (30 %),
plus Gaussian noise (σ = 5).

#### Loan-portfolio metrics

- Expected number of open loans is Poisson(1 + 3 × utilisation).

#### Recent-behaviour features

- Recent credit-card spend is 40 – 80 % of available limit times utilisation.

- A recent loan disbursement amount equals 20 – 60 % of current outstanding.

#### Target-label heuristic
Each customer’s credit-score movement is determined by a rule-based risk score:

Start at zero.

Add one risk point for each high-risk signal
• DPD in last 3 months > 3 days
• Utilisation > 0.80
• Hard enquiries > 2
• Months since last default < 6
• Recent card spend > 30 % of credit limit
• Recent loan disbursement > 50 % of monthly income

Subtract one risk point for each strong positive signal
• Repayment-history score > 80
• Income-to-EMI ratio > 2
• More than three open loans while repayment score > 90

Map the net score to classes
• Risk ≥ 2 → decrease
• Risk ≤ –2 → increase
• Otherwise → stable

This simple heuristic introduces class imbalance typical in real-world credit-score data while remaining transparent and interpretable.


The dataset contains 25,000+ rows with synthetic customer credit behavior data. Features include:
- Customer demographics (age, gender, location)
- Financial metrics (income, EMI, outstanding)
- Credit behavior (utilization, inquiries, repayment history)
- Target variable: credit_score_movement (increase, decrease, stable)

The synthetic credit‐score dataset is stored at:
`src/data/credit_score_dataset.csv`

# Exploratory Data Analysis

## What the data look like
* **Shape:** **25 000 × 17** (`rows × columns`)
* **Missing values:** none detected  
* **Target class distribution**

  | Class        | Count | Proportion |
  |--------------|------:|-----------:|
  | `stable`     | 20 898 | **83.6 %** |
  | `increase`   | 2 346  | 9.4 % |
  | `decrease`   | 1 756  | 7.0 % |

> The data are heavily imbalanced toward the *stable* category, which mirrors real-world credit-bureau skews.

## Demographics  
   * Most customers are in their mid-thirties.  
   * The gender split is almost even.  
   * Half of the customers live in urban areas, about a third in suburbs, and the rest in rural regions.

## Income and spending  
   * Urban customers earn the most on average; suburban customers earn a bit less, and rural customers earn the least.  
   * People spend roughly a quarter of their income on EMIs (loan instalments).  
   * Credit limits scale with income, so higher earners have higher limits.

## Credit utilisation  
   * The typical customer uses less than one-third of the credit available to them.  
   * Higher utilisation often goes hand-in-hand with more hard enquiries and slightly worse repayment scores.

## Key Correlations 
* `credit_utilization_ratio` ↑ correlates with  
  * `num_hard_inquiries_last_6m`   
  * `dpd_last_3_months` 
* `repayment_history_score` shows **negative** correlation with utilisation and DPD.
* Income and credit limit are strongly but intentionally correlated; other financials remain largely independent.

## Target vs. Demographics
* **Location effect**  
  * Urban customers have a slightly higher share of `increase`  and lower `decrease`  outcomes.  
  * Rural segment skews more to `stable`.
* **Gender effect** is negligible.

## Unsupervised Risk Segmentation
A K-Means (**k = 3**) on eight scaled numeric features yields:

| Cluster | Approx. Share | Key Traits | Typical Target |
|---------|--------------:|------------|----------------|
| **High Risk** | 30 % | utilisation ≈ 0.30, enquiries ≈ 2, repayment ≈ 68 | `decrease` over-represented |
| **Medium Risk** | 35 % | lowest DPD, utilisation ≈ 0.19, best repayment ≈ 73 | majority of `increase` |
| **Low Risk** | 35 % | utilisation ≈ 0.44, enquiries ≈ 3.1, repayment ≈ 60 | largely `stable` |

A 2-component PCA plot confirms good visual separation among clusters.

## 8. Takeaways for Modelling
* Strong class imbalance → use stratified splits and imbalance mitigation (e.g. SMOTE, class-weighted loss).  
* Credit-utilisation, repayment score, and DPD are the most informative predictors—they appear in every high-correlation pair and cluster boundary.  
* Location offers mild predictive value; gender offers virtually none.

# Model Implementation

## Feature Engineering Summary

The `FeatureEngineer` class performs light preprocessing on the credit score dataset to prepare it for model training.

### Key Steps:
* **Derived Features**
  - `income_to_emi_ratio`: Ratio of monthly income to EMI outflow
  - `recent_usage_ratio`: Ratio of recent credit card usage to total credit limit

* **Preprocessing Pipeline**
  - **Numerical Features**: Scaled using `StandardScaler`
  - **Categorical Features**: One-hot encoded using `OneHotEncoder`

* **Output**
  - Final processed dataset with transformed features and target column
  - Saved to: `data/processed/processed_dataset.csv`

## Model Training Class

The `ModelTrainer` class handles the training and evaluation of a credit score movement classifier using a Random Forest model with class imbalance handling.

### Key Components:

* **Label Encoding**
  - The target labels (`increase`, `stable`, `decrease`) are encoded using `LabelEncoder`.

* **Train-Test Split**
  - 80–20 stratified split to preserve class proportions across training and testing data due to heavy class imbalance.

* **Imbalance Handling**
  - Uses `SMOTETomek` (oversampling + undersampling) inside an imbalanced-learn pipeline.
  - Random Forest with `class_weight='balanced_subsample'` for internal reweighting.

* **Model Selection**
  - GridSearchCV with 5-fold cross-validation.
  - Hyperparameters tuned:
    - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
  - Optimized on **macro F1 score** to balance performance across imbalanced classes.

* **Evaluation Metrics**
  - Accuracy
  - Macro F1 Score
  - Class-wise Recall (to understand per-class sensitivity)

* **Feature Importance**
  - Extracted from the trained Random Forest model and sorted by contribution.

### Output:
- Trained and optimized Random Forest model
- Evaluation dictionary with key metrics
- Feature importance DataFrame (optional)

## Model Training code

### Training
* **Pipeline.** Combined **SMOTETomek** (to both oversample minority classes and clean borderline noise) with a **Random Forest Classifier** (`class_weight='balanced_subsample'`) inside an `ImbPipeline`.  
* **Hyper-parameter search.** 5-fold `GridSearchCV`, scored with **macro-averaged F1**, exploring:  
  * `n_estimators: 100 | 200`  
  * `max_depth: None | 10 | 20`  
  * `min_samples_split: 2 | 5`  
  * `min_samples_leaf: 1 | 2`  

### 3  Hold-out Evaluation (20 % Test Split)  
| Metric | Score |
|--------|-------|
| Accuracy | `0.967` |
| Macro F1 | `0.9235` |

**Class-wise recall**

| Class | Recall |
|-------|--------|
| Decrease | `0.8889` |
| Increase | `0.9957` |
| Stable   | `0.9703` |

We can see that model performs quite well on all the classes.

## 4 Feature Importance & Explainability  

The Random Forest classifier revealed the following as the top 5 drivers of credit score movement:

| Rank | Feature                         | Description |
|------|----------------------------------|-------------|
| 1    | `repayment_history_score`       | Historical repayment reliability score  
| 2    | `credit_utilization_ratio`      | Ratio of credit used to credit available  
| 3    | `num_hard_inquiries_last_6m`    | Hard credit checks in past 6 months  
| 4    | `recent_loan_disbursed_amount`  | Amount disbursed in most recent loan  
| 5    | `dpd_last_3_months`             | Days past due in the last 3 months  

---

# Assumptions

1. **Score Movement Logic:**
   - High DPD + High Utilization + Recent Inquiries → Score likely to **decrease**
   - Low EMI/Income ratio + High repayment history → Score likely to **increase**
   - Moderately balanced metrics → Score likely to remain **stable**

2. **Data Representativeness:**
   - Synthetic demographics follow real-world distributions of income, age, credit activity.

3. **Causality Assumption:**
   - Observed financial behaviors are primary drivers of credit score movement.

---

# Strategic Recommendations Based on Model Insights

## Proposed Product & Policy Interventions

Based on the model’s predictive insights and observed customer behavior patterns, the following interventions are recommended:

---

### 1. High-Risk Segments (Credit Score Likely to Decrease)

**Typical characteristics:**
- High debt-to-income ratio (monthly EMI > 50% of income)
- Credit utilization ratio > 80%
- Recent credit card spikes
- Multiple hard inquiries in the past 6 months
- Recent defaults or missed payments

**Proposed Interventions:**
- **Credit Wellness Programs:** Launch structured credit counseling and budgeting workshops.
- **Debt Consolidation Loans:** Offer tailored consolidation products to reduce EMI burden.
- **Utilization Alerts:** Provide app-based or SMS alerts when credit utilization exceeds thresholds.
- **Late Payment Protection:** Introduce grace periods or temporary EMI restructuring.

---

### 2. Opportunity Segments (Credit Score Likely to Improve)

**Typical characteristics:**
- Repayment history score > 80
- Low EMI-to-income ratio (< 30%)
- Credit utilization < 30%
- No recent defaults or hard inquiries

**Proposed Interventions:**
- **Credit Line Expansion:** Increase limits for responsible users to boost flexibility while keeping utilization low.
- **Loyalty Rewards:** Offer incentives for continued timely repayment and responsible credit behavior.
- **Pre-approved Loans:** Provide instant offers for low-risk users with reduced documentation.
- **Personalized Guidance:** Send proactive advice on maintaining or improving credit health.



