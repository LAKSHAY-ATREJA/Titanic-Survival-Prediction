"""
Simple Gender-Based Survival Model
====================================
Predicts survival using only an individual's gender as a predictor.
Model: y = b0 + b1 * (gender)
"""

import csv
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'output', 'gender_model_predictions.csv')

data = []

with open(TRAIN_FILE, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        data.append(row)

data = np.array(data)

# Separate data by gender (column index 4 = Sex in train.csv)
women_mask = data[:, 4] == 'female'
men_mask = data[:, 4] != 'female'

# Survival is column index 1
women_survived = data[women_mask, 1].astype(float)
men_survived = data[men_mask, 1].astype(float)
all_survived = data[:, 1].astype(float)

proportion_survived = all_survived.sum() / all_survived.size
proportion_women_survived = women_survived.sum() / women_survived.size
proportion_men_survived = men_survived.sum() / men_survived.size

print('Proportion of people who survived: {:.4f}'.format(proportion_survived))
print('Proportion of women who survived:  {:.4f}'.format(proportion_women_survived))
print('Proportion of men who survived:    {:.4f}'.format(proportion_men_survived))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(TRAIN_FILE, 'r', newline='') as f_in, \
     open(OUTPUT_FILE, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)
    writer.writerow(['Predicted'] + header)

    for row in reader:
        prediction = '1' if row[4] == 'female' else '0'
        writer.writerow([prediction] + row)

print('Predictions saved to:', OUTPUT_FILE)
print('Analysis complete.')
