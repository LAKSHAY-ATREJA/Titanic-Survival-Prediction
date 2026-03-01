# Titanic Survival Prediction

> **Kaggle Competition:** [Titanic – Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

A complete end-to-end machine learning project that predicts passenger survival on the Titanic using Python and the PyData ecosystem. The project covers exploratory data analysis, feature engineering, and comparison of three supervised learning models.

---

## Features

- **Exploratory Data Analysis** — survival breakdowns by gender, class, age, and embarkation port, visualised with Matplotlib
- **Feature Engineering** — family size, is-alone flag, median imputation for missing age and fare values
- **Three ML models** — Logistic Regression, SVM (RBF kernel), and Random Forest Classifier
- **Rigorous evaluation** — 5-fold stratified cross-validation with mean accuracy and standard deviation
- **Kaggle-ready output** — generates `PassengerId, Survived` CSV files for each model
- **Standalone demo script** — reproduces the full pipeline in one command

---

## Example Output

Running `python demo.py` produces:

```
============================================================
DATASET SUMMARY
============================================================
  Training samples   : 891
  Survived           : 342 (38.4%)
  Age  (mean ± std)  : 29.4 ± 13.0

  Survival by Gender:
    Female    74.2%  (233/314)
    Male      18.9%  (109/577)

  Survival by Passenger Class:
    Class 1    63.0%  (136/216)
    Class 2    47.3%  (87/184)
    Class 3    24.2%  (119/491)

============================================================
MODEL EVALUATION  (5-Fold Stratified Cross-Validation)
============================================================
  Model                      CV Accuracy   Std Dev
  --------------------------------------------------
  Logistic Regression            0.7991    0.0169
  SVM (RBF kernel)               0.6824    0.0164
  Random Forest                  0.8227    0.0226
============================================================
```

Charts are saved to `images/`:

| Survival Analysis | Feature Importance |
|---|---|
| ![survival](images/survival_analysis.png) | ![importance](images/feature_importance.png) |

---

## Project Structure

```
Titanic-Survival-Prediction/
├── data/
│   ├── train.csv                  # Training set (891 passengers)
│   ├── test.csv                   # Test set (418 passengers)
│   └── output/                    # Generated submission CSVs
├── images/                        # EDA and result charts
├── KaggleAux/
│   ├── __init__.py
│   └── predict.py                 # Utility: statsmodels logit prediction helper
├── Python Examples/
│   ├── agc_simp_gendermodel.py    # Baseline: gender-only model
│   ├── agcgenderclassmodel.py     # Gender + class + fare lookup model
│   ├── agcfirstforest.py          # Random Forest with raw NumPy arrays
│   └── agc_embark_class_gender.py # statsmodels Logit: class, sex, age, embarkation
├── Titanic.ipynb                  # Full interactive analysis notebook
├── demo.py                        # Standalone demo script (start here)
├── requirements.txt
└── ReadMe.md
```

---

## Technologies

| Library | Purpose |
|---|---|
| **pandas** | Data loading, cleaning, feature engineering |
| **NumPy** | Array operations |
| **Matplotlib** | Visualisation |
| **scikit-learn** | Logistic Regression, SVM, Random Forest, cross-validation |
| **statsmodels** | Logit regression with statistical summaries |
| **patsy** | R-style model formula syntax |
| **Jupyter** | Interactive notebook exploration |

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/LAKSHAY-ATREJA/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction

# 2. (Optional) create a virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Quick Demo (recommended)

```bash
python demo.py
```

This runs the full pipeline — preprocessing, cross-validation, submission generation, and chart export — in under 30 seconds. Outputs land in `data/output/` and `images/`.

### Interactive Notebook

```bash
jupyter notebook Titanic.ipynb
```

The notebook walks through every step with explanations and inline plots:

1. Data loading and cleaning
2. Exploratory visualisations
3. Logistic Regression with statsmodels
4. SVM with three kernels (linear, RBF, polynomial)
5. Random Forest classifier
6. K-fold cross-validation
7. Kaggle submission export

### Individual Example Scripts

Each script in `Python Examples/` is a self-contained approach:

```bash
# Baseline gender model
python "Python Examples/agc_simp_gendermodel.py"

# Gender + class + fare lookup table
python "Python Examples/agcgenderclassmodel.py"

# Random Forest via sklearn
python "Python Examples/agcfirstforest.py"

# statsmodels Logit regression
python "Python Examples/agc_embark_class_gender.py"
```

All outputs are written to `data/output/`.

---

## Dataset

The Titanic dataset is provided by Kaggle. Download it from the [competition page](https://www.kaggle.com/c/titanic/data) and place the files in the `data/` directory.

| File | Rows | Description |
|---|---|---|
| `train.csv` | 891 | Labelled training data (includes `Survived`) |
| `test.csv` | 418 | Unlabelled test data for submission |

Key features used:

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Sex` | Gender |
| `Age` | Age in years (median-imputed where missing) |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| `FamilySize`* | Engineered: SibSp + Parch + 1 |
| `IsAlone`* | Engineered: 1 if travelling alone, else 0 |

\* Engineered features added in `demo.py` and the Python Examples.

---

## Results

| Model | CV Accuracy (5-fold) |
|---|---|
| Logistic Regression | 79.9% |
| SVM (RBF kernel) | 68.2% |
| Random Forest | **82.3%** |

---

## Project Info

- **Type:** Machine Learning Classification — tutorial / Kaggle benchmark
- **Dataset:** Kaggle Titanic: Machine Learning from Disaster
- **Python:** 3.9+
- **Original analysis:** late 2024
- **Repository published:** 2026
