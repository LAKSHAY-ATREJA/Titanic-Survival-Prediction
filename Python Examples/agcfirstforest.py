"""
Random Forest Classifier
==========================
Non-parametric survival prediction using a Random Forest ensemble.
"""

import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'output', 'random_forest_predictions.csv')

train_data = []
test_data = []

with open(TRAIN_FILE, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        train_data.append(row)
train_data = np.array(train_data)

with open(TEST_FILE, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        test_data.append(row)
test_data = np.array(test_data)

# Encode Sex: male=1, female=0 (train col 4, test col 3)
train_data[train_data[:, 4] == 'male', 4] = '1'
train_data[train_data[:, 4] == 'female', 4] = '0'

test_data[test_data[:, 3] == 'male', 3] = '1'
test_data[test_data[:, 3] == 'female', 3] = '0'

# Encode Embarked: C=0, S=1, Q=2 (train col 11, test col 10)
for col, arr in [(11, train_data), (10, test_data)]:
    arr[arr[:, col] == 'C', col] = '0'
    arr[arr[:, col] == 'S', col] = '1'
    arr[arr[:, col] == 'Q', col] = '2'

# Fill missing Age with median (train col 5, test col 4)
for col, arr in [(5, train_data), (4, test_data)]:
    known = arr[arr[:, col] != '', col].astype(float)
    arr[arr[:, col] == '', col] = str(np.median(known))

# Fill missing Embarked with mode
for col, arr in [(11, train_data), (10, test_data)]:
    known = arr[arr[:, col] != '', col].astype(float)
    values, counts = np.unique(known, return_counts=True)
    arr[arr[:, col] == '', col] = str(int(values[np.argmax(counts)]))

# Fill missing Fare in test data with median per class (test col 8)
for i in range(test_data.shape[0]):
    if test_data[i, 8] == '':
        pclass = test_data[i, 1]
        same_class = test_data[(test_data[:, 8] != '') & (test_data[:, 1] == pclass), 8].astype(float)
        test_data[i, 8] = str(np.median(same_class)) if same_class.size > 0 else '0'

# Drop Name (col 2), Ticket (col 8), Cabin (col 10) from train
# Columns: 0=PassengerId, 1=Survived, 2=Pclass, 3=Name, 4=Sex, 5=Age,
#          6=SibSp, 7=Parch, 8=Ticket, 9=Fare, 10=Cabin, 11=Embarked
train_data = np.delete(train_data, [3, 8, 10], axis=1)

# Drop Name (col 1), Ticket (col 7), Cabin (col 9) from test
# Columns: 0=PassengerId, 1=Pclass, 2=Name, 3=Sex, 4=Age,
#          5=SibSp, 6=Parch, 7=Ticket, 8=Fare, 9=Cabin, 10=Embarked
test_data = np.delete(test_data, [2, 7, 9], axis=1)

print('Training Random Forest...')
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(train_data[:, 2:].astype(float), train_data[:, 1].astype(float))

print('Generating predictions...')
predictions = forest.predict(test_data[:, 1:].astype(float))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(TEST_FILE, 'r', newline='') as f_in, \
     open(OUTPUT_FILE, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    writer.writerow(['PassengerId', 'Survived'])
    next(reader)  # skip header

    for i, row in enumerate(reader):
        writer.writerow([row[0], int(predictions[i])])

print('Predictions saved to:', OUTPUT_FILE)
print('Analysis complete.')
