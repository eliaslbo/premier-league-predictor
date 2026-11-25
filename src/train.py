import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from features import load_data, build_features, MODEL_DIR, RESULTS_DIR

SPLIT_DATE = pd.Timestamp('2024-08-01')

df = load_data()
print(f"Loaded {len(df)} matches from {df['Season'].nunique()} seasons")

print("Building features (takes a minute)...")
features = build_features(df)

is_test = df['Date'] >= SPLIT_DATE
X_train = features[~is_test]
y_train = df[~is_test]['FTR']
X_test = features[is_test]
y_test = df[is_test]['FTR']
print(f"Training matches: {len(X_train)}, test matches: {len(X_test)}")

# min_samples_leaf stops the trees from memorising single matches,
# which gives smoother and more realistic probabilities
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
rf.fit(X_train, y_train)

print(f"Test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.pkl'))

# Save test predictions together with the real odds so backtest.py can use them
probs = rf.predict_proba(X_test)
test = df[is_test][['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']].reset_index(drop=True)
for i, result in enumerate(rf.classes_):
    test['model_prob_' + result] = probs[:, i]
test['book_prob_H'] = X_test['prob_home'].values
test['book_prob_D'] = X_test['prob_draw'].values
test['book_prob_A'] = X_test['prob_away'].values

os.makedirs(RESULTS_DIR, exist_ok=True)
test.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)
print("Saved model and test predictions")
