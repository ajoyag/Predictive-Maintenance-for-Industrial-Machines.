import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
X_scaled = scaler.fit_transform(X)

# train model
logistic = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=15, min_samples_leaf=5, random_state=42)
naive_bayes = GaussianNB()

logistic.fit(X_train_scaled, y_train)
decision_tree.fit(X_train, y_train)
naive_bayes.fit(X_train_scaled, y_train)

# prediction
log_pred = logistic.predict(X_test_scaled)
dt_pred = decision_tree.predict(X_test)
nb_pred = naive_bayes.predict(X_test_scaled)

# evaluate model
log_acc = accuracy_score(y_test, log_pred)
log_prec = precision_score(y_test, log_pred, zero_division=0)
log_rec = recall_score(y_test, log_pred, zero_division=0)
log_f1 = f1_score(y_test, log_pred, zero_division=0)
log_mse = mean_squared_error(y_test, log_pred)

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, zero_division=0)
dt_rec = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)
dt_mse = mean_squared_error(y_test, dt_pred)

nb_acc = accuracy_score(y_test, nb_pred)
nb_prec = precision_score(y_test, nb_pred, zero_division=0)
nb_rec = recall_score(y_test, nb_pred, zero_division=0)
nb_f1 = f1_score(y_test, nb_pred, zero_division=0)
nb_mse = mean_squared_error(y_test, nb_pred)

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
log_cv = cross_val_score(logistic, X_scaled, y, cv=kf).mean()
dt_cv = cross_val_score(decision_tree, X, y, cv=kf).mean()
nb_cv = cross_val_score(naive_bayes, X_scaled, y, cv=kf).mean()

# print results
print("Logistic Regression Metrics:")
print(f"Accuracy: {log_acc:.4f}")
print(f"Precision: {log_prec:.4f}")
print(f"Recall: {log_rec:.4f}")
print(f"F1 Score: {log_f1:.4f}")
print(f"Mean Squared Error: {log_mse:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_pred))
print(f"K-Fold Accuracy: {log_cv:.4f}\n")

print("Decision Tree Metrics:")
print(f"Accuracy: {dt_acc:.4f}")
print(f"Precision: {dt_prec:.4f}")
print(f"Recall: {dt_rec:.4f}")
print(f"F1 Score: {dt_f1:.4f}")
print(f"Mean Squared Error: {dt_mse:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print(f"K-Fold Accuracy: {dt_cv:.4f}\n")

print("Naive Bayes Metrics:")
print(f"Accuracy: {nb_acc:.4f}")
print(f"Precision: {nb_prec:.4f}")
print(f"Recall: {nb_rec:.4f}")
print(f"F1 Score: {nb_f1:.4f}")
print(f"Mean Squared Error: {nb_mse:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))
print(f"K-Fold Accuracy: {nb_cv:.4f}\n")

# result table
print("| Model | Accuracy | Precision | Recall |")
print("|-------|-----------|-----------|--------|")
print(f"| Logistic Regression | {log_acc:.2f} | {log_prec:.2f} | {log_rec:.2f} |")
print(f"| Decision Tree | {dt_acc:.2f} | {dt_prec:.2f} | {dt_rec:.2f} |")
print(f"| Naive Bayes | {nb_acc:.2f} | {nb_prec:.2f} | {nb_rec:.2f} |")
print()

# best model
best_acc = max(log_acc, dt_acc, nb_acc)
if best_acc == log_acc:
    print("Best model based on accuracy: Logistic Regression")
elif best_acc == dt_acc:
    print("Best model based on accuracy: Decision Tree")
else:
    print("Best model based on accuracy: Naive Bayes")

# explanation
print("\nNote: High accuracy may be due to clear separation between healthy and failure data in the dataset.")
