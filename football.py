import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data from external CSV
df = pd.read_csv(r'D:\Vishnu\cc\matches.csv')  # <-- Make sure this file exists in your directory

# Features used for prediction
features = ['HomeShots', 'AwayShots', 'HomePossession', 'AwayPossession']
X = df[features]

# Use vectorized mapping to get the winning team — NO if statements
result_map = {
    "HomeWin": df['HomeTeam'],
    "AwayWin": df['AwayTeam'],
    "Draw": "Draw"
}
df['Winner'] = df['Result'].map(result_map)

# Encode the winner label
le = LabelEncoder()
y = le.fit_transform(df['Winner'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Predict a new match ---
# You can change this or loop over many predictions
new_match = pd.DataFrame({
    'HomeShots': [13],
    'AwayShots': [11],
    'HomePossession': [52],
    'AwayPossession': [48]
})

# Make prediction
prediction = model.predict(new_match)
predicted_team = le.inverse_transform(prediction)

print(f"🏆 Predicted Winner: {predicted_team[0]}")
