"""
Gender + Class + Fare Survival Model
======================================
Predicts survival based on gender, passenger class, and ticket fare.
Bins fare into brackets and uses historical survival rates per group.
"""

import os
import csv
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'output', 'gender_class_model_predictions.csv')

data = []

with open(TRAIN_FILE, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        data.append(row)
data = np.array(data)

# Bin fare values; cap anything above fare_ceiling in the last bracket
fare_ceiling = 40
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size
number_of_classes = 3  # 1st, 2nd, 3rd class

# Cap fares at the ceiling value
data[data[:, 9].astype(float) >= fare_ceiling, 9] = str(fare_ceiling - 1.0)

# Build a survival lookup table: [gender, class, fare_bracket] -> survival rate
survival_table = np.zeros([2, number_of_classes, number_of_price_brackets], dtype=float)

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        fare_low = j * fare_bracket_size
        fare_high = (j + 1) * fare_bracket_size

        female_mask = (
            (data[:, 4] == 'female') &
            (data[:, 2].astype(float) == i + 1) &
            (data[:, 9].astype(float) >= fare_low) &
            (data[:, 9].astype(float) < fare_high)
        )
        male_mask = (
            (data[:, 4] != 'female') &
            (data[:, 2].astype(float) == i + 1) &
            (data[:, 9].astype(float) >= fare_low) &
            (data[:, 9].astype(float) < fare_high)
        )

        female_survived = data[female_mask, 1].astype(float)
        male_survived = data[male_mask, 1].astype(float)

        survival_table[0, i, j] = np.mean(female_survived) if female_survived.size > 0 else 0.0
        survival_table[1, i, j] = np.mean(male_survived) if male_survived.size > 0 else 0.0

# Round to binary survival decision (>= 0.5 -> survived)
survival_table[survival_table >= 0.5] = 1
survival_table[survival_table < 0.5] = 0

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(TEST_FILE, 'r', newline='') as f_in, \
     open(OUTPUT_FILE, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    writer.writerow(['PassengerId', 'Survived'])
    next(reader)  # skip header

    for row in reader:
        passenger_id = row[0]
        pclass = int(row[1])
        sex = row[3]

        try:
            fare = float(row[8])
        except ValueError:
            fare = (3 - pclass) * fare_bracket_size  # fallback by class

        if fare >= fare_ceiling:
            bin_fare = number_of_price_brackets - 1
        else:
            bin_fare = int(fare // fare_bracket_size)

        gender_idx = 0 if sex == 'female' else 1
        prediction = int(survival_table[gender_idx, pclass - 1, bin_fare])
        writer.writerow([passenger_id, prediction])

print('Predictions saved to:', OUTPUT_FILE)
print('Analysis complete.')
