import os
import argparse
import joblib
import pandas as pd

from features import load_data, match_features, MODEL_DIR

parser = argparse.ArgumentParser(description='Predict a Premier League match and look for a value bet')
parser.add_argument('home_team')
parser.add_argument('away_team')
parser.add_argument('odds_h', type=float, help='decimal odds for home win')
parser.add_argument('odds_d', type=float, help='decimal odds for draw')
parser.add_argument('odds_a', type=float, help='decimal odds for away win')
parser.add_argument('--date', default=None, help='match date, YYYY-MM-DD (default: today)')
parser.add_argument('--season', default='2025-26')
parser.add_argument('--bankroll', type=float, default=1000)
args = parser.parse_args()

rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))
df = load_data()

if args.date:
    date = pd.Timestamp(args.date)
else:
    date = pd.Timestamp.today()

teams = set(df['HomeTeam'])
for team in [args.home_team, args.away_team]:
    if team not in teams:
        print(f"Unknown team '{team}'. Use one of: {', '.join(sorted(teams))}")
        exit()

features = match_features(df, args.home_team, args.away_team, date, args.season,
                          args.odds_h, args.odds_d, args.odds_a)
X = pd.DataFrame([features])

probs = dict(zip(rf.classes_, rf.predict_proba(X)[0]))
odds = {'H': args.odds_h, 'D': args.odds_d, 'A': args.odds_a}
names = {'H': 'Home win', 'D': 'Draw', 'A': 'Away win'}
bookie = {'H': features['prob_home'], 'D': features['prob_draw'], 'A': features['prob_away']}

print(f"\n{args.home_team} vs {args.away_team}")
print(f"{'':10} {'Model':>7} {'Bookie':>7} {'Value':>7}")
for result in ['H', 'D', 'A']:
    value = probs[result] * odds[result] - 1
    print(f"{names[result]:10} {probs[result]:>7.1%} {bookie[result]:>7.1%} {value:>7.3f}")
    if result == 'H' or value > best_value:
        best = result
        best_value = value

if best_value > 0.05:
    # Quarter Kelly, never more than 10% of the bankroll
    kelly = (probs[best] * odds[best] - 1) / (odds[best] - 1)
    stake = min(args.bankroll * kelly * 0.25, args.bankroll * 0.1)
    print(f"\nValue bet: {names[best]} @ {odds[best]:.2f}, suggested stake {stake:.2f}")
else:
    print("\nNo value bet found")
