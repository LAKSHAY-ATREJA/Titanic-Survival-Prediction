"""
Logistic Regression: Embarkation + Class + Gender Model
=========================================================
Predicts survival using a statsmodels Logit regression with features:
passenger class, sex, age, number of siblings/spouses, and embarkation port.

Formula: Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)
"""

import os
import warnings
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'output', 'logit_embark_class_gender_predictions.csv')

# Load and clean training data
train_df = pd.read_csv(TRAIN_FILE)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
train_df = train_df.dropna(subset=['Age', 'Embarked', 'Survived'])

# Logistic regression formula
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)'

y, x = dmatrices(formula, data=train_df, return_type='dataframe')

model = sm.Logit(y, x)
results = model.fit(disp=False)

print('Logit Regression Results:')
print(results.summary())

# Load and prepare test data
test_df = pd.read_csv(TEST_FILE)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# Fill missing values in test set so dmatrices can process it
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Add a placeholder Survived column required by dmatrices
test_df['Survived'] = 0

_, x_test = dmatrices(formula, data=test_df, return_type='dataframe')

# Align columns between training and test design matrices
common_cols = [c for c in x.columns if c in x_test.columns]
x_test = x_test[common_cols]

# Generate probability predictions and convert to binary
probabilities = results.predict(x_test)
predictions = (probabilities >= 0.5).astype(int)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

output = test_df[['PassengerId']].copy()
output['Survived'] = predictions.values
output.to_csv(OUTPUT_FILE, index=False)

print('\nBegin Predictions:')
for i, (pid, pred) in enumerate(zip(output['PassengerId'], output['Survived'])):
    print(f'  Row {i:3d} | PassengerId: {pid} | Survived: {pred}')

print(f'\nPredictions saved to: {OUTPUT_FILE}')
print('Analysis complete.')
