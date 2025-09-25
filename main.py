import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# 1️⃣ Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

# 2️⃣ Feature engineering (if any)
df['age_chol'] = df['age'] * df['cholesterol']
df['thalach_age'] = df['thalassemia'] / (df['age'] + 1)

# 3️⃣ Define features and target
target_col = "heart_disease"
X = df.drop(columns=[target_col])
y = df[target_col]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Define models
rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=10,
                            min_samples_leaf=2, max_features="sqrt", random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
lr = LogisticRegression(max_iter=500, solver="liblinear")

# 7️⃣ Stacking/Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft',
    n_jobs=-1
)

# 8️⃣ Train ensemble
voting_clf.fit(X_train_scaled, y_train)

# 9️⃣ Save model and scaler
joblib.dump(voting_clf, "HeartDisease_Ensemble.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training complete! Model and scaler saved.")
