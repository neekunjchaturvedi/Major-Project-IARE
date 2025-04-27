import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./heart.csv')

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


real_world_sample = pd.DataFrame([{
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3,
    'slope': 0, 'ca': 0, 'thal': 1
}])

real_world_sample = scaler.transform(real_world_sample)
real_world_prediction = rf_clf.predict(real_world_sample)

result = "The person has heart disease" if real_world_prediction[0] == 1 else "The person does not have heart disease"
print(result)