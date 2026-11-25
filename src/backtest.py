import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score

from features import RESULTS_DIR

STAKE = 10
VALUE_THRESHOLD = 0.05

test = pd.read_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'))
outcomes = ['A', 'D', 'H']
odds_columns = {'H': 'B365H', 'D': 'B365D', 'A': 'B365A'}

model_probs = test[['model_prob_A', 'model_prob_D', 'model_prob_H']].values
book_probs = test[['book_prob_A', 'book_prob_D', 'book_prob_H']].values
actual = test['FTR'].values

# 1. How good are the probabilities compared to the bookmaker?
model_pred = [outcomes[i] for i in model_probs.argmax(axis=1)]
book_pred = [outcomes[i] for i in book_probs.argmax(axis=1)]

print("=== Model vs bookmaker on the test set ===")
print(f"Matches:            {len(test)}")
print(f"Accuracy model:     {accuracy_score(actual, model_pred):.3f}")
print(f"Accuracy favourite: {accuracy_score(actual, book_pred):.3f}")
print(f"Log loss model:     {log_loss(actual, model_probs, labels=outcomes):.4f}")
print(f"Log loss bookmaker: {log_loss(actual, book_probs, labels=outcomes):.4f}  (lower is better)")


# 2. Betting simulation with the REAL odds (including the bookmaker margin)
def run_backtest(test, threshold):
    bets = []
    for _, row in test.iterrows():
        best_value = -1
        for result in outcomes:
            value = row['model_prob_' + result] * row[odds_columns[result]] - 1
            if value > best_value:
                best_value = value
                bet_on = result
                odds = row[odds_columns[result]]

        if best_value < threshold:
            continue

        if bet_on == row['FTR']:
            profit = STAKE * (odds - 1)
        else:
            profit = -STAKE
        bets.append({'Date': row['Date'], 'bet_on': bet_on, 'odds': odds, 'value': best_value, 'profit': profit})
    return pd.DataFrame(bets)


print("\n=== Value betting with real Bet365 odds ===")
print(f"{'Threshold':>10} {'Bets':>6} {'Win rate':>9} {'ROI':>7}")
for threshold in [0.0, 0.05, 0.10, 0.15, 0.20]:
    bets = run_backtest(test, threshold)
    if len(bets) == 0:
        continue
    win_rate = (bets['profit'] > 0).mean()
    roi = bets['profit'].sum() / (len(bets) * STAKE)
    print(f"{threshold:>10.2f} {len(bets):>6} {win_rate:>9.1%} {roi:>7.1%}")

bets = run_backtest(test, VALUE_THRESHOLD)
bets['cumulative_profit'] = bets['profit'].cumsum()

# Rough check of how much of the result could be luck
roi = bets['profit'].sum() / (len(bets) * STAKE)
roi_std = bets['profit'].std() / STAKE / np.sqrt(len(bets))
print(f"\nROI at threshold {VALUE_THRESHOLD}: {roi:.1%} +/- {1.96 * roi_std:.1%} (95% interval)")


# 3. Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(bets['cumulative_profit'].values)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title(f'Cumulative profit, £{STAKE} per bet, real odds')
axes[0].set_xlabel('Bet number')
axes[0].set_ylabel('Profit (£)')
axes[0].grid(True)

# Calibration: when the model says 40%, does it happen 40% of the time?
bins = np.linspace(0, 1, 11)
for probs, label in [(model_probs, 'Model'), (book_probs, 'Bookmaker')]:
    predicted = probs.flatten()
    happened = np.array([[actual[i] == r for r in outcomes] for i in range(len(actual))]).flatten()
    x, y = [], []
    for i in range(len(bins) - 1):
        in_bin = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if in_bin.sum() >= 20:
            x.append(predicted[in_bin].mean())
            y.append(happened[in_bin].mean())
    axes[1].plot(x, y, marker='o', label=label)

axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect')
axes[1].set_title('Calibration')
axes[1].set_xlabel('Predicted probability')
axes[1].set_ylabel('Actual frequency')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'backtest.png'), dpi=120)
print(f"\nPlot saved to results/backtest.png")
plt.show()
