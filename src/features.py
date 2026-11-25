import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

COLUMNS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']


def load_data():
    all_seasons = []
    for file in sorted(os.listdir(DATA_DIR)):
        if file.endswith('.csv'):
            df_season = pd.read_csv(os.path.join(DATA_DIR, file), usecols=COLUMNS)
            df_season['Season'] = file.replace('.csv', '')
            all_seasons.append(df_season)

    df = pd.concat(all_seasons, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df


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


def get_team_form(df, team, date, n=5):
    before = df[df['Date'] < date]
    team_matches = before[(before['HomeTeam'] == team) | (before['AwayTeam'] == team)].tail(n)
    home_matches = before[before['HomeTeam'] == team].tail(n)
    away_matches = before[before['AwayTeam'] == team].tail(n)

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
    season_matches = df[(df['Season'] == season) & (df['Date'] < date)]
    team_matches = season_matches[(season_matches['HomeTeam'] == team) | (season_matches['AwayTeam'] == team)]

    w, d, l, gs, gc = calc_stats(team_matches, team)
    points = 3 * w + d
    goal_diff = gs - gc
    return points, goal_diff


def match_features(df, home_team, away_team, date, season, odds_h, odds_d, odds_a):
    home_form = get_team_form(df, home_team, date)
    away_form = get_team_form(df, away_team, date)
    home_points, home_gd = get_team_points_this_season(df, home_team, date, season)
    away_points, away_gd = get_team_points_this_season(df, away_team, date, season)

    # Remove the bookmaker margin so the probabilities sum to 1
    prob_h = 1 / odds_h
    prob_d = 1 / odds_d
    prob_a = 1 / odds_a
    total = prob_h + prob_d + prob_a

    return {
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



def build_features(df):
    rows = []
    for _, row in df.iterrows():
        features = match_features(df, row['HomeTeam'], row['AwayTeam'], row['Date'], row['Season'],
                                  row['B365H'], row['B365D'], row['B365A'])
        rows.append(features)
    return pd.DataFrame(rows)
