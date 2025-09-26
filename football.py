import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r'D:\Vishnu\cc\matches.csv') 


features = ['HomeShots', 'AwayShots', 'HomePossession', 'AwayPossession']
X = df[features]


result_map = {
    "HomeWin": df['HomeTeam'],
    "AwayWin": df['AwayTeam'],
    "Draw": "Draw"
}
df['Winner'] = df['Result'].map(result_map)


le = LabelEncoder()
y = le.fit_transform(df['Winner'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


new_match = pd.DataFrame({
    'HomeShots': [13],
    'AwayShots': [11],
    'HomePossession': [52],
    'AwayPossession': [48]
})

prediction = model.predict(new_match)
predicted_team = le.inverse_transform(prediction)

print(f"🏆 Predicted Winner: {predicted_team[0]}")
