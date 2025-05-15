import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import MultiLabelBinarizer

# --- Load and clean data ---
df = pd.read_csv("data/tmdb_5000_movies.csv")

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
df['success'] = df['revenue'] > df['budget']

# --- Parse genres and one-hot encode ---
df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notnull(x) else [])
top_genres = df['genres'].explode().value_counts().head(10).index.tolist()
df['genres_filtered'] = df['genres'].apply(lambda g: [x for x in g if x in top_genres])

mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres_filtered']), columns=mlb.classes_)
df = pd.concat([df.reset_index(drop=True), genre_dummies], axis=1)

# --- Prepare features ---
features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count'] + top_genres
df = df.dropna(subset=features)
X = df[features]
y = df['success']
expected_features = X.columns.tolist()

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --- Feature importance plot ---
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Flop', 'Hit'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid(True)
plt.legend()
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_proba))

# --- Custom movie prediction ---
custom_input = {f: 0 for f in expected_features}
custom_input.update({
    'budget': 80000000,
    'popularity': 40,
    'runtime': 120,
    'vote_average': 7.2,
    'vote_count': 3000,
    'Action': 1,
    'Drama': 1,
    'Thriller': 1
})

custom_df = pd.DataFrame([custom_input])
prediction = model.predict(custom_df)[0]
print("\nCustom Movie Prediction: ", "✅ HIT!" if prediction else "❌ FLOP!")
