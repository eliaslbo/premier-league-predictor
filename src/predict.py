import pandas as pd
import joblib
import numpy as np
import os

# Load models
rf = joblib.load('../models/random_forest.pkl')
le = joblib.load('../models/label_encoder.pkl')

# Load data
data_folder = '../data/'
all_seasons = []
for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        df_season = pd.read_csv(data_folder + file)
        df_season['Season'] = file.replace('.csv', '')
        all_seasons.append(df_season)

df = pd.concat(all_seasons, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)
df = df[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']]

def get_team_form_v2(df, team, date, n=5):
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].tail(n)
    
    home_matches = df[
        (df['HomeTeam'] == team) &
        (df['Date'] < date)
    ].tail(n)
    
    away_matches = df[
        (df['AwayTeam'] == team) &
        (df['Date'] < date)
    ].tail(n)
    
    def calc_stats(matches, team):
        wins, draws, losses = 0, 0, 0
        goals_scored, goals_conceded = 0, 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                goals_scored += match['FTHG']
                goals_conceded += match['FTAG']
                if match['FTR'] == 'H': wins += 1
                elif match['FTR'] == 'D': draws += 1
                else: losses += 1
            else:
                goals_scored += match['FTAG']
                goals_conceded += match['FTHG']
                if match['FTR'] == 'A': wins += 1
                elif match['FTR'] == 'D': draws += 1
                else: losses += 1
        return wins, draws, losses, goals_scored, goals_conceded
    
    w, d, l, gs, gc = calc_stats(team_matches, team)
    hw, hd, hl, hgs, hgc = calc_stats(home_matches, team)
    aw, ad, al, ags, agc = calc_stats(away_matches, team)
    
    return {
        'wins': w, 'draws': d, 'losses': l,
        'goals_scored': gs, 'goals_conceded': gc,
        'home_wins': hw, 'home_goals_scored': hgs, 'home_goals_conceded': hgc,
        'away_wins': aw, 'away_goals_scored': ags, 'away_goals_conceded': agc
    }

def get_team_points_this_season(df, team, date, season):
    home_matches = df[
        (df['HomeTeam'] == team) & 
        (df['Date'] < date) & 
        (df['Season'] == season)
    ]
    away_matches = df[
        (df['AwayTeam'] == team) & 
        (df['Date'] < date) & 
        (df['Season'] == season)
    ]
    
    points = 0
    goal_diff = 0
    
    for _, match in home_matches.iterrows():
        if match['FTR'] == 'H': points += 3
        elif match['FTR'] == 'D': points += 1
        goal_diff += match['FTHG'] - match['FTAG']
    
    for _, match in away_matches.iterrows():
        if match['FTR'] == 'A': points += 3
        elif match['FTR'] == 'D': points += 1
        goal_diff += match['FTAG'] - match['FTHG']
    
    return points, goal_diff

def predict_match(home_team, away_team, date, season, odds_h, odds_d, odds_a, bankroll, kelly_fraction=0.25):
    # Get features
    home_form = get_team_form_v2(df, home_team, date)
    away_form = get_team_form_v2(df, away_team, date)
    home_points, home_gd = get_team_points_this_season(df, home_team, date, season)
    away_points, away_gd = get_team_points_this_season(df, away_team, date, season)
    
    prob_h = 1 / odds_h
    prob_d = 1 / odds_d
    prob_a = 1 / odds_a
    total = prob_h + prob_d + prob_a
    
    features = {
        'home_wins': home_form['wins'],
        'home_draws': home_form['draws'],
        'home_losses': home_form['losses'],
        'home_goals_scored': home_form['goals_scored'],
        'home_goals_conceded': home_form['goals_conceded'],
        'home_home_wins': home_form['home_wins'],
        'home_home_goals_scored': home_form['home_goals_scored'],
        'home_home_goals_conceded': home_form['home_goals_conceded'],
        'away_wins': away_form['wins'],
        'away_draws': away_form['draws'],
        'away_losses': away_form['losses'],
        'away_goals_scored': away_form['goals_scored'],
        'away_goals_conceded': away_form['goals_conceded'],
        'away_away_wins': away_form['away_wins'],
        'away_away_goals_scored': away_form['away_goals_scored'],
        'away_away_goals_conceded': away_form['away_goals_conceded'],
        'home_points': home_points,
        'home_goal_diff': home_gd,
        'away_points': away_points,
        'away_goal_diff': away_gd,
        'prob_home': prob_h / total,
        'prob_draw': prob_d / total,
        'prob_away': prob_a / total,
    }
    
    X = pd.DataFrame([features])
    
    # Get probabilities
    probs = rf.predict_proba(X)[0]
    prob_a_model, prob_d_model, prob_h_model = probs[0], probs[1], probs[2]
    
    # Calculate value
    value_h = prob_h_model * odds_h - 1
    value_d = prob_d_model * odds_d - 1
    value_a = prob_a_model * odds_a - 1
    
    # Find best value bet
    best_value = max(value_h, value_d, value_a)
    if best_value == value_h:
        bet_on = 'Home Win'
        bet_odds = odds_h
        bet_prob = prob_h_model
    elif best_value == value_d:
        bet_on = 'Draw'
        bet_odds = odds_d
        bet_prob = prob_d_model
    else:
        bet_on = 'Away Win'
        bet_odds = odds_a
        bet_prob = prob_a_model
    
    # Kelly stake
    kelly = (bet_prob * bet_odds - 1) / (bet_odds - 1)
    kelly = max(0, kelly)
    stake = bankroll * kelly * kelly_fraction
    stake = min(stake, bankroll * 0.1)
    
    print(f"\n{'='*40}")
    print(f"{home_team} vs {away_team}")
    print(f"{'='*40}")
    print(f"Model probabilities:")
    print(f"  Home win: {prob_h_model:.1%}")
    print(f"  Draw:     {prob_d_model:.1%}")
    print(f"  Away win: {prob_a_model:.1%}")
    print(f"\nValue:")
    print(f"  Home: {value_h:.3f}")
    print(f"  Draw: {value_d:.3f}")
    print(f"  Away: {value_a:.3f}")
    
    if best_value > 0.05:
        print(f"\n✅ VALUE BET: {bet_on} @ {bet_odds:.2f}")
        print(f"   Suggested stake: £{stake:.2f}")
    else:
        print(f"\n❌ No value bet found")
    
    return best_value, bet_on, stake


if __name__ == "__main__":
    predict_match(
        home_team='Arsenal',
        away_team='Chelsea',
        date=pd.Timestamp('2025-09-01'),
        season='2025-26',
        odds_h=2.1,
        odds_d=3.5,
        odds_a=3.8,
        bankroll=1000
    )