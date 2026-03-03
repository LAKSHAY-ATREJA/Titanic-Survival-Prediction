"""
Titanic Survival Prediction — Interactive Streamlit App
========================================================
Enter your passenger details and see whether you would have survived.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

st.set_page_config(
    page_title='Titanic Survival Predictor',
    page_icon='🚢',
    layout='wide',
)


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Ticket', 'Cabin', 'Name'], axis=1, errors='ignore')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


@st.cache_resource(show_spinner='Training models on Titanic data…')
def train_models():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    df = load_and_preprocess(train_path)
    X = df[FEATURES].values
    y = df['Survived'].values

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF kernel)':    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores[name] = (cv_scores.mean(), cv_scores.std())
        model.fit(X, y)

    return models, scores, df


def make_survival_charts(_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Titanic — Survival Analysis', fontsize=13)

    _df['Survived'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color=['#E74C3C', '#2ECC71'], alpha=0.85
    )
    axes[0].set_title('Overall Survival')
    axes[0].set_xticklabels(['Died', 'Survived'], rotation=0)
    axes[0].set_ylabel('Passengers')

    for sex_val, label, color in [(0, 'Female', '#E91E8C'), (1, 'Male', '#1565C0')]:
        grp = _df[_df['Sex'] == sex_val]
        rate = grp['Survived'].mean()
        axes[1].bar(label, rate, color=color, alpha=0.85)
    axes[1].set_title('Survival Rate by Gender')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)

    class_rates = _df.groupby('Pclass')['Survived'].mean()
    axes[2].bar([f'Class {c}' for c in class_rates.index], class_rates.values,
                color='steelblue', alpha=0.85)
    axes[2].set_title('Survival Rate by Class')
    axes[2].set_ylabel('Rate')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    return fig


def make_importance_chart(_rf_model):
    importances = _rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(FEATURES)), importances[order], color='steelblue', alpha=0.85)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in order], rotation=35, ha='right')
    ax.set_title('Feature Importances — Random Forest')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    return fig


def predict_passenger(models, row_values):
    """Run all models and return a dict of {name: (prediction, probability)}."""
    x = np.array(row_values).reshape(1, -1)
    results = {}
    for name, model in models.items():
        pred = int(model.predict(x)[0])
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(x)[0][1]
        else:
            prob = None
        results[name] = (pred, prob)
    return results


def main():
    st.title('🚢 Titanic Survival Predictor')
    st.markdown(
        'Train-set cross-validation + interactive passenger prediction using '
        'three machine learning models.'
    )

    models, cv_scores, train_df = train_models()

    tab1, tab2, tab3 = st.tabs(['🔮 Predict', '📊 Dataset Analysis', '🤖 Model Performance'])

    # ── Tab 1: Predict ────────────────────────────────────────────────────────
    with tab1:
        st.subheader('Would you have survived the Titanic?')
        st.markdown('Fill in your passenger details and all three models will vote.')

        col1, col2, col3 = st.columns(3)
        with col1:
            pclass = st.selectbox('Passenger Class', [1, 2, 3],
                                  format_func=lambda x: f'Class {x}')
            sex = st.radio('Sex', ['Female', 'Male'])
            age = st.slider('Age', 1, 80, 30)
        with col2:
            sibsp = st.slider('Siblings / Spouses aboard', 0, 8, 0)
            parch = st.slider('Parents / Children aboard', 0, 6, 0)
            fare = st.slider('Fare paid (£)', 0.0, 512.0, 32.0, step=0.5)
        with col3:
            embarked_label = st.selectbox('Port of Embarkation',
                                          ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'])

        sex_enc = 0 if sex == 'Female' else 1
        embarked_enc = {'Cherbourg (C)': 0, 'Queenstown (Q)': 1, 'Southampton (S)': 2}[embarked_label]
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        row = [pclass, sex_enc, age, sibsp, parch, fare, embarked_enc, family_size, is_alone]

        if st.button('🎯 Predict My Survival', type='primary'):
            predictions = predict_passenger(models, row)

            votes_survived = sum(1 for pred, _ in predictions.values() if pred == 1)
            majority = 'SURVIVED ✅' if votes_survived >= 2 else 'DID NOT SURVIVE ❌'

            if votes_survived >= 2:
                st.success(f'### Majority verdict: {majority}')
            else:
                st.error(f'### Majority verdict: {majority}')

            st.markdown('**Individual model predictions:**')
            res_cols = st.columns(3)
            for idx, (name, (pred, prob)) in enumerate(predictions.items()):
                with res_cols[idx]:
                    label = '✅ Survived' if pred == 1 else '❌ Did not survive'
                    st.metric(label=name, value=label,
                              delta=f'Probability: {prob:.1%}' if prob is not None else '')

    # ── Tab 2: Dataset Analysis ───────────────────────────────────────────────
    with tab2:
        st.subheader('Training Data Overview')
        col_a, col_b, col_c = st.columns(3)
        col_a.metric('Total Passengers', len(train_df))
        col_b.metric('Survived', int(train_df['Survived'].sum()))
        col_c.metric('Survival Rate', f"{train_df['Survived'].mean():.1%}")

        st.pyplot(make_survival_charts(train_df))

        st.subheader('Survival breakdown')
        breakdown = train_df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'count'])
        breakdown.columns = ['Survival Rate', 'Count']
        breakdown.index = breakdown.index.map(
            lambda x: (f'Class {x[0]}', 'Female' if x[1] == 0 else 'Male')
        )
        breakdown['Survival Rate'] = breakdown['Survival Rate'].map('{:.1%}'.format)
        st.dataframe(breakdown, use_container_width=True)

    # ── Tab 3: Model Performance ──────────────────────────────────────────────
    with tab3:
        st.subheader('5-Fold Stratified Cross-Validation')

        perf_data = {
            'Model': list(cv_scores.keys()),
            'CV Accuracy': [f'{mean:.4f}' for mean, _ in cv_scores.values()],
            'Std Dev': [f'± {std:.4f}' for _, std in cv_scores.values()],
        }
        st.dataframe(pd.DataFrame(perf_data).set_index('Model'), use_container_width=True)

        st.subheader('Random Forest — Feature Importances')
        st.pyplot(make_importance_chart(models['Random Forest']))


if __name__ == '__main__':
    main()
