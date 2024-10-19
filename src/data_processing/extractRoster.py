import pandas as pd


def extract_players(game_nhl: dict) -> pd.DataFrame:
    """
    Extract player data from the NHL game
    :param game_nhl: Dictionary containing the data of the NHL game
    :return: A Pandas DataFrame containing player data
    """
    # Extract player data
    df_players = pd.DataFrame(game_nhl['rosterSpots'])[['playerId', 'firstName', 'lastName']]

    # Keep the default name for each player (first and last name)
    df_players['firstName'] = df_players['firstName'].apply(lambda x: x['default'])
    df_players['lastName'] = df_players['lastName'].apply(lambda x: x['default'])
    return df_players


def extract_teams(game_nhl: dict) -> pd.DataFrame:
    """
     Extract team data from the NHL game
     :param game_nhl: Dictionary containing the data of the NHL game
     :return: A Pandas DataFrame containing team data
    """
    home_team = {'teamId': game_nhl['homeTeam']['id'], 'teamName': game_nhl['homeTeam']['name']['default'],
                 'teamSide': 'home'}
    away_team = {'teamId': game_nhl['awayTeam']['id'], 'teamName': game_nhl['awayTeam']['name']['default'],
                 'teamSide': 'away'}
    return pd.DataFrame([home_team, away_team])
