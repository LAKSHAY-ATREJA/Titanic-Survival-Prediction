"""
Titanic Survival Prediction - Demo Script
==========================================
End-to-end ML pipeline: data loading, preprocessing, model training,
cross-validation evaluation, and Kaggle submission file generation.

Usage:
    python demo.py

Outputs:
    data/output/submission_logistic_regression.csv
    data/output/submission_svm_rbf_kernel.csv
    data/output/submission_random_forest.csv
    images/feature_importance.png
    images/survival_analysis.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']


def load_and_preprocess(filepath):
    """Load CSV and engineer features; returns a clean DataFrame."""
    df = pd.read_csv(filepath)

    df = df.drop(['Ticket', 'Cabin', 'Name'], axis=1, errors='ignore')

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])       # female=0, male=1
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])  # C=0, Q=1, S=2

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df


def print_dataset_summary(df):
    """Print key survival statistics from the training DataFrame."""
    print('\n' + '=' * 60)
    print('DATASET SUMMARY')
    print('=' * 60)
    print(f'  Training samples   : {len(df)}')
    print(f'  Survived           : {int(df["Survived"].sum())} ({df["Survived"].mean():.1%})')
    print(f'  Age  (mean ± std)  : {df["Age"].mean():.1f} ± {df["Age"].std():.1f}')

    print('\n  Survival by Gender:')
    for sex_val, label in [(0, 'Female'), (1, 'Male')]:
        grp = df[df['Sex'] == sex_val]
        rate = grp['Survived'].mean()
        print(f'    {label:<8}  {rate:.1%}  ({int(grp["Survived"].sum())}/{len(grp)})')

    print('\n  Survival by Passenger Class:')
    for cls in sorted(df['Pclass'].unique()):
        grp = df[df['Pclass'] == cls]
        rate = grp['Survived'].mean()
        print(f'    Class {cls}    {rate:.1%}  ({int(grp["Survived"].sum())}/{len(grp)})')
    print('=' * 60)


def evaluate_models(X_train, y_train):
    """Train and cross-validate three classifiers; return fitted models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF kernel)':    SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print('\n' + '=' * 60)
    print('MODEL EVALUATION  (5-Fold Stratified Cross-Validation)')
    print('=' * 60)
    print(f'  {"Model":<25} {"CV Accuracy":>12}  {"Std Dev":>8}')
    print('  ' + '-' * 50)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f'  {name:<25} {scores.mean():>11.4f}  {scores.std():>8.4f}')

    print('=' * 60)
    return models


def generate_submissions(models, X_train, y_train, X_test, test_df):
    """Fit each model and write a Kaggle-formatted submission CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('\nGenerating submission files...')

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test).astype(int)

        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': preds,
        })

        safe_name = (
            name.lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
        )
        path = os.path.join(OUTPUT_DIR, f'submission_{safe_name}.csv')
        submission.to_csv(path, index=False)
        print(f'  Saved: {path}')

    return models


def plot_feature_importance(rf_model, feature_names):
    """Bar chart of Random Forest feature importances."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title('Feature Importances — Random Forest', fontsize=14)
    plt.bar(range(len(feature_names)), importances[order], color='steelblue', alpha=0.8)
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in order],
               rotation=40, ha='right', fontsize=11)
    plt.ylabel('Importance', fontsize=11)
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'\n  Feature importance chart → {path}')


def plot_survival_analysis(df):
    """Grid of survival breakdown charts saved to images/."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Titanic — Survival Analysis', fontsize=15, y=1.02)

    # Overall survival counts
    df['Survived'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color=['#E74C3C', '#2ECC71'], alpha=0.8
    )
    axes[0].set_title('Overall Survival')
    axes[0].set_xticklabels(['Died (0)', 'Survived (1)'], rotation=0)
    axes[0].set_ylabel('Passengers')

    # Survival by gender (0=female, 1=male)
    for sex_val, label, color in [(0, 'Female', '#FA2379'), (1, 'Male', '#1F77B4')]:
        rates = df[df['Sex'] == sex_val]['Survived'].value_counts(normalize=True).sort_index()
        rates.plot(kind='bar', ax=axes[1], label=label, color=color, alpha=0.65)
    axes[1].set_title('Survival Rate by Gender')
    axes[1].set_xticklabels(['Died', 'Survived'], rotation=0)
    axes[1].legend()

    # Survival rate by class
    class_survival = df.groupby('Pclass')['Survived'].mean()
    class_survival.plot(kind='bar', ax=axes[2], color='steelblue', alpha=0.8)
    axes[2].set_title('Survival Rate by Passenger Class')
    axes[2].set_xticklabels([f'Class {c}' for c in class_survival.index], rotation=0)
    axes[2].set_ylabel('Survival Rate')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'survival_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Survival analysis chart → {path}')


def main():
    print('=' * 60)
    print('  Titanic Survival Prediction — Demo')
    print('=' * 60)

    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Training data not found: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Test data not found: {test_path}')

    train_df = load_and_preprocess(train_path)
    test_df = load_and_preprocess(test_path)

    print_dataset_summary(train_df)

    X_train = train_df[FEATURES].values
    y_train = train_df['Survived'].values
    X_test = test_df[FEATURES].values

    models = evaluate_models(X_train, y_train)

    generate_submissions(models, X_train, y_train, X_test, test_df)

    print('\nGenerating charts...')
    plot_survival_analysis(train_df)
    plot_feature_importance(models['Random Forest'], FEATURES)

    print('\n' + '=' * 60)
    print('  Demo complete.')
    print(f'  Submission CSVs  →  {OUTPUT_DIR}/')
    print(f'  Charts           →  {IMAGES_DIR}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
