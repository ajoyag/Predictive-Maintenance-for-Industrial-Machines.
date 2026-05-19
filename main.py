import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# load data
df = pd.read_csv('dataset.csv')

# clean data
df = df.drop_duplicates()
df = df.fillna(df.median())

# split data
X = df.drop('fail', axis=1)
y = df['fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train model
logistic = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=15, min_samples_leaf=5, random_state=42)
naive_bayes = GaussianNB()

logistic.fit(X_train_scaled, y_train)
decision_tree.fit(X_train, y_train)
naive_bayes.fit(X_train_scaled, y_train)

# evaluate model
log_pred = logistic.predict(X_test_scaled)
dt_pred = decision_tree.predict(X_test)
nb_pred = naive_bayes.predict(X_test_scaled)

models = ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
predictions = [log_pred, dt_pred, nb_pred]
accuracies = []

print("\nEvaluation Metrics")
print("==================")
for name, pred in zip(models, predictions):
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"\n{name}:")
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Precision Score: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall Score: {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

# ask user for sensor inputs
print("\nProvide Sensor Inputs:")
def get_float_input(prompt, default_val):
    val = input(prompt)
    if val.strip() == "":
        print(f"Using default value: {default_val}")
        return default_val
    try:
        return float(val)
    except ValueError:
        print(f"Invalid input. Using default value: {default_val}")
        return default_val

footfall = get_float_input(f"Enter footfall (max {df['footfall'].max()}) [default 50]: ", 50)
tempMode = get_float_input(f"Enter tempMode (max {df['tempMode'].max()}) [default 40]: ", 40)
AQ = get_float_input(f"Enter AQ (max {df['AQ'].max()}) [default 100]: ", 100)
USS = get_float_input(f"Enter USS (max {df['USS'].max()}) [default 15]: ", 15)
CS = get_float_input(f"Enter CS (max {df['CS'].max():.1f}) [default 45]: ", 45)
VOC = get_float_input(f"Enter VOC (max {df['VOC'].max()}) [default 100]: ", 100)
RP = get_float_input(f"Enter RP (max {df['RP'].max()}) [default 100]: ", 100)
IP = get_float_input(f"Enter IP (max {df['IP'].max()}) [default 100]: ", 100)
Temperature = get_float_input(f"Enter Temperature (max {df['Temperature'].max()}) [default 50]: ", 50)

input_data = pd.DataFrame([[footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature]], 
                          columns=['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'])

# scale input data
input_data_scaled = scaler.transform(input_data)

# prediction
log_user_pred = logistic.predict(input_data_scaled)[0]
dt_user_pred = decision_tree.predict(input_data)[0]
nb_user_pred = naive_bayes.predict(input_data_scaled)[0]

def get_condition(pred):
    return "Failure Risk" if pred == 1 else "Healthy"

print("\nPrediction Results")
print()
print(f"Logistic Regression : {get_condition(log_user_pred)}")
print(f"Decision Tree       : {get_condition(dt_user_pred)}")
print(f"Naive Bayes         : {get_condition(nb_user_pred)}")
print()

# find best model
best_idx = np.argmax(accuracies)
best_model_name = models[best_idx]
best_pred = predictions[best_idx]

# graphs
os.makedirs('results', exist_ok=True)

# 1. Accuracy Comparison
plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center')
plt.savefig('results/accuracy_comparison.png')
plt.close()

# 2. Actual vs Predicted (Best Model)
plt.figure(figsize=(8, 4))
plt.scatter(range(30), y_test.values[:30], marker='o', label='Actual')
plt.scatter(range(30), best_pred[:30] + 0.02, marker='x', label='Predicted')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.xlabel('Sample')
plt.ylabel('Class (0=Healthy, 1=Fail)')
plt.legend()
plt.savefig('results/actual_vs_predicted.png')
plt.close()

# 3. Confusion Matrix (Best Model)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix ({best_model_name})')
plt.colorbar()
plt.xticks([0, 1], ['Healthy', 'Fail'])
plt.yticks([0, 1], ['Healthy', 'Fail'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
plt.savefig('results/confusion_matrix.png')
plt.close()

print("Graphs saved in 'results' folder.")

# explanation
print("\nNote: High accuracy may be due to clear separation between healthy and failure data in the dataset.")
