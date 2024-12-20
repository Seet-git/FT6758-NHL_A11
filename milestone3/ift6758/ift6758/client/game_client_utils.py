import pandas as pd
import numpy as np


def determine_goal_advantage(df: pd.DataFrame) -> pd.Series:
    """
    Determine if the event team is in an advantage, disadvantage, or neutral situation
    :param df: DataFrame containing 'situationCode' and 'teamSide' columns
    :return: Series containing 'Advantage', 'Disadvantage', or 'Neutral'
    """

    return df.apply(lambda x: "Advantage" if (int(x['situationCode'][1]) > int(x['situationCode'][2]) and x[
        'teamSide'] == 'away') or (int(x['situationCode'][2]) > int(x['situationCode'][1]) and x[
        'teamSide'] == 'home')
    else "Disadvantage" if (int(x['situationCode'][1]) < int(x['situationCode'][2]) and x['teamSide'] == 'away') or
                           (int(x['situationCode'][2]) < int(x['situationCode'][1]) and x['teamSide'] == 'home')
    else 'Neutral', axis=1)

def calculate_empty_goal_net(df: pd.DataFrame) -> pd.Series:
    """
    Determine if the goal net is empty
    :param df: DataFrame containing 'situationCode' and 'teamSide' columns
    :return: Series containing boolean values
    """

    return df.apply(lambda x: x['situationCode'][3] if x['teamSide'] == 'away' else x['situationCode'][0] if len(x['situationCode']) == 4 else 0
                    , axis=1).map(
        {'0': True, '1': False})

def process_previous_event(df: pd.DataFrame):
    # Décale le dataframe pour récupérer l'élément précédent
    df_copy = df.copy().shift(1)
    # Get previous event
    df['previousEventType'] = df_copy['typeDescKey']

    # Get previous time
    df['timeSinceLastEvent'] = df['gameSeconds'].diff()
    df['timeSinceLastEvent'] = df.apply(lambda x: 0
    if pd.isnull(x['timeSinceLastEvent']) else abs(x['timeSinceLastEvent']), axis=1) #Abs to prevent some error or negative time

    # Get previous coordinates
    details = df_copy['details'].apply(pd.Series)
    df["previousXCoord"] = details['xCoord']
    df["previousYCoord"] = details['yCoord']

    return df

def process_event_details(df: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Process event details and merge player information.
    :param df: DataFrame containing 'details' column
    :param df_players: DataFrame containing player data
    :return: DataFrame containing 'iceCoord', 'shootingPlayer', 'goaliePlayer', 'zoneShoot', and 'shotType' columns
    """

    # Split 'details' into 'iceCoord', 'shootingPlayerId', 'scoringPlayerId', and 'goalieInNetId'
    df_details = pd.DataFrame(df['details'].tolist())

    # Merge 'shooting' and 'scoring' player, to keep only one column
    df_details['shootingPlayerId'] = df_details['shootingPlayerId'].fillna(0) + df_details['scoringPlayerId'].fillna(0)

    # Fill missing 'goalieInNetId' values with 0
    df_details['goalieInNetId'] = df_details['goalieInNetId'].fillna(0)

    # Convert 'shootingPlayerId' and 'goalieInNetId' as integer
    df_details['shootingPlayerId'] = df_details['shootingPlayerId'].astype(int)
    df_details['goalieInNetId'] = df_details['goalieInNetId'].astype('Int64')  # Int64: handling NaN values

    # Add the shooter names by merging IDs
    df_details = pd.merge(df_players, df_details, left_on='playerId', right_on='shootingPlayerId', how='right').drop(
        columns=['playerId'])

    # Keep only full name
    df_details['shootingPlayer'] = df_details['firstName'] + ' ' + df_details['lastName']
    df_details.drop(['firstName', 'lastName'], axis=1, inplace=True)

    # Add the goalie names by merging IDs
    df_details = pd.merge(df_players, df_details, left_on='playerId', right_on='goalieInNetId', how='right').drop(
        columns=['playerId'])

    # Keep only full name
    df_details['goaliePlayer'] = df_details['firstName'] + ' ' + df_details['lastName']
    df_details.drop(['firstName', 'lastName'], axis=1, inplace=True)

    return df_details

def process_period_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose periodDescriptor and return processed dataframe.
    :param df: DataFrame containing 'periodDescriptor' column
    :return: DataFrame containing 'periodType', 'number', 'maxRegulationPeriods', and 'currentPeriod' columns
    """

    # Split 'periodDescriptor' into 'periodType', 'number', and 'maxRegulationPeriods'
    df_period = pd.DataFrame(df['periodDescriptor'].tolist())

    # Convert 'number' and 'maxRegulationPeriods' columns as strings
    df_period[['number', 'maxRegulationPeriods']] = df_period[['number', 'maxRegulationPeriods']].astype(str)

    # Add period
    df_period['numberPeriod'] = df_period['number']

    return df_period

def minutes_to_seconds(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Convert 'minutes:seconds' to total seconds
    :param df: DataFrame containing 'minutes' and 'seconds' columns
    :param column: Column to convert
    :return: Series containing total seconds
    """

    # Split columns into 'minutes' and 'seconds' and 'number of period' as integer
    df['minutes'] = df[column].str.split(':').str[0].astype(int)
    df['seconds'] = df[column].str.split(':').str[1].astype(int)
    df['numberPeriod'] = df['numberPeriod'].astype(int)

    # Total in seconds
    df[column] = df['minutes'] * 60 + df['seconds'] + 20 * 60 * (df['numberPeriod'] - 1)

    # Drop columns
    df.drop(['minutes', 'seconds'], axis=1, inplace=True)

    return df[column]

def convert_game_to_dataframe(game_nhl: dict) -> pd.DataFrame:
    """
    Convert NHL game event data into a clean dataframe
    :param game_nhl: Dictionary containing the data of the NHL game
    :return: A Pandas DataFrame containing filtered data
    """
    # Extract play-by-play data
    df_pbp = pd.DataFrame(game_nhl['plays'])

    # Extract player and team data
    df_players = extract_players(game_nhl)
    df_teams = extract_teams(game_nhl)

    # Create a new dataframe for the event data
    clean_df = pd.DataFrame(df_pbp[['periodDescriptor', 'timeInPeriod', 'situationCode',
                                    'typeDescKey', 'details']])

    # PERIOD DESCRIPTOR
    # Create a dataframe to decompose the period descriptor fields
    df_period = process_period_data(clean_df)
    clean_df.drop('periodDescriptor', axis=1, inplace=True)

    # Add 'gameID', 'periodType' and 'currentPeriod' columns to the new dataframe
    clean_df.insert(0, 'idGame', game_nhl['id'])
    clean_df.insert(1, 'periodType', df_period['periodType'])
    clean_df.insert(3, 'numberPeriod', df_period['numberPeriod'])

    # TIME IN PERIOD
    # Convert time in the period to seconds
    clean_df['gameSeconds'] = minutes_to_seconds(clean_df, 'timeInPeriod')
    clean_df.drop('timeInPeriod', axis=1, inplace=True)

    clean_df = process_previous_event(clean_df)

    # DETAILS
    # Filter to keep only events of type 'shot-on-goal' or 'goal'
    clean_df = clean_df[
        (clean_df['typeDescKey'] == 'shot-on-goal') | (clean_df['typeDescKey'] == 'goal')].reset_index(
        drop=True)

    # Process event details and merge player information
    df_details = process_event_details(clean_df, df_players)
    clean_df.drop('details', axis=1, inplace=True)

    # Add team data by merging IDs
    df_details = pd.merge(df_teams, df_details, left_on='teamId', right_on='eventOwnerTeamId', how='right')

    # Add the extracted data to the new dataframe
    clean_df['xCoord'] = df_details['xCoord']
    clean_df['yCoord'] = df_details['yCoord']
    clean_df['zoneShoot'] = df_details['zoneCode']
    clean_df['shootingPlayer'] = df_details['shootingPlayer']
    clean_df['goaliePlayer'] = df_details['goaliePlayer']
    clean_df['shotType'] = df_details['shotType']
    clean_df.insert(5, 'eventOwnerTeam', df_details['teamName'])
    clean_df['teamSide'] = df_details['teamSide']

    # Calculate emptyGoalNet and goal advantage
    clean_df['emptyGoalNet'] = calculate_empty_goal_net(clean_df).astype(int)
    clean_df['isGoalAdvantage'] = determine_goal_advantage(clean_df)

    clean_df['isGoal'] = clean_df['typeDescKey'].apply(lambda x: 1 if x == 'goal' else 0)

    # Add shot distance
    clean_df = additional_features(clean_df)

    # Drop situation code
    clean_df.drop('situationCode', axis=1, inplace=True)
    return clean_df

def angle_between_vectors(v1: np.array, v2: np.array) -> float:
    """
    Calculate the angle in degrees between two 2D vectors.
    :param v1: np.array Vector 1
    :param v2: np.array Vector 2
    :return: float Angle in degrees
    """
    # Dot product
    dot_product = np.dot(v1, v2)

    # norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # cosine of the angle
    cos_angle = dot_product / (norm_v1 * norm_v2)

    # Clip the value to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Angle in radians
    angle_radians = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def get_coordinates(row: pd.Series, home_team_initial_side: str) -> list:
    """
    Cette fonction calcule les nouvelles coordonnées d'un tir en fonction de la période et du côté de la patinoire
    sur lequel l'équipe commence. Elle prend en compte les rotations des coordonnées en fonction du changement de côté
    de l'équipe après chaque période.

    :param row: Une ligne d'un DataFrame contenant les données de tir, y compris les coordonnées et la période.
    :param side: Le côté initial de l'équipe (gauche ou droite).

    :return: tuple: Les nouvelles coordonnées après ajustement basé sur le côté et la période.
    """

    initial_side = None

    if(row['teamSide'] == 'home'):
        initial_side = home_team_initial_side
    else:
        if home_team_initial_side == 'left':
            initial_side = 'right'
        else:
            initial_side = 'left'


    new_coords = [(0, 0), (0,0)]
    current_side = initial_side

    # en saison réguliere, on change de cote seulement si on n'est pas en prolongation
    if str(row['idGame'])[4:6] == '02' and row['numberPeriod'] <= 3:
        if row['numberPeriod'] % 2 == 0:
            # on change le camp
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'
    else:
        if row['numberPeriod'] % 2 == 0:
            # on change le camp
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'

    if current_side == 'left':
        # je fais une rotation de 90 degre dans le sens inverse des aiguilles d'une montre
        new_coords = [(-row['yCoord'], row['xCoord']), (-row['previousYCoord'], row['previousXCoord'])]
    else:
        # je fais une rotation de 90 degre dans le sens des aiguilles d'une montre
        new_coords = [(row['yCoord'], -row['xCoord']), (row['previousYCoord'], -row['previousXCoord'])]

    return new_coords

def additional_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features to the cleaned DataFrame.
    :param clean_df: Pd.DataFrame Cleaned DataFrame
    :return: pd.Dataframe with additional features
    """

    # List to store the updated coordinates as tuples
    coords_list = []  # List to store the updated coordinates as tuples
    coords_last_event_list = []

    # Finding on which side the team is starting the game
    first_home_team_offensive_event = clean_df[(clean_df['zoneShoot'] == 'O') & (clean_df['teamSide'] == 'home')].iloc[
        0]
    home_team_initial_side = 'right' if first_home_team_offensive_event['xCoord'] < 0 else 'left'

    for _, row in clean_df.iterrows():
        # Calculer les nouvelles coordonnées selon le côté et la période
        new_coords, new_last_event_coord = get_coordinates(row, home_team_initial_side)
        coords_list.append(new_coords)  # Stocker les coordonnées ajustées
        coords_last_event_list.append(new_last_event_coord)

    # Ajouter les nouvelles coordonnées à la DataFrame
    clean_df['adjustedCoord'] = coords_list
    clean_df['adjustedLastEventCoord'] = coords_last_event_list

    # ADDITIONAL FEATURES

    # Add shot distance

    # Define the Euclidian distance function
    dist_euclidian = lambda x1, x2: np.round(np.linalg.norm(np.array(x1) - np.array(x2)), decimals=1)

    # Add shot distance based on the ice coordinates
    clean_df['shotDistance'] = clean_df.apply(lambda x: dist_euclidian(x['adjustedCoord'], np.array([0, 89])), axis=1)

    # Add distance from the last event
    clean_df['distanceFromLastEvent'] = clean_df.apply(
        lambda x: dist_euclidian(x1=(x['xCoord'], x['yCoord'])
                                 , x2=(x['previousXCoord'], x['previousYCoord']))
        if not pd.isnull(x['previousXCoord']) else None, axis=1)

    # Add rebound information
    clean_df['rebound'] = clean_df.apply(lambda x:
                                         True if x['previousEventType'] == 'shot-on-goal' else False, axis=1
                                         )

    # Add speed
    clean_df['speedFromLastEvent'] = clean_df.apply(lambda x:
                                       x['distanceFromLastEvent'] / x['timeSinceLastEvent']
                                       if x['timeSinceLastEvent'] != 0 else 0
                                       , axis=1)

    # Add a shot angle based on the ice coordinates
    # x['adjustedCoord']-np.array([0,89]) calcule les coordonnées du vecteur qui commence aux filets et s'arrête à l'emplacement du tirs
    # np.array([0, -89]) est le vecteur qui commence dans les filets et s'arrête au centre du stade/de la patinoire
    clean_df['shotAngle'] = clean_df.apply(
        lambda x: angle_between_vectors(x['adjustedCoord'] - np.array([0, 89]), np.array([0, -89])), axis=1)

    # Previous shot angle
    clean_df['reboundAngleShot'] = clean_df.apply(
        lambda x: angle_between_vectors(x['adjustedLastEventCoord'] - np.array([0, 89]), np.array([0, -89]) + x['shotAngle']
        if x['rebound'] else 0), axis=1)

    # Drop the adjusted coordinates
    clean_df.drop(columns=['adjustedCoord'], inplace=True)
    clean_df.drop(columns=['adjustedLastEventCoord'], inplace=True)

    # Add time before the last shot to observe the offensive pressure

    # Sort the dataframe by the period to calculate the time since the last shot
    clean_df['offensivePressureTime'] = clean_df.groupby('eventOwnerTeam')['gameSeconds'].diff()

    # Convert the time to minutes and seconds
    clean_df['offensivePressureTime'] = clean_df.apply(lambda x: 0
    if pd.isnull(x['offensivePressureTime']) else x['offensivePressureTime'], axis=1)

    return clean_df

def extract_teams(game_nhl: dict) -> pd.DataFrame:
    """
     Extract team data from the NHL game
     :param game_nhl: Dictionary containing the data of the NHL game
     :return: A Pandas DataFrame containing team data
    """
    home_team = {'teamId': game_nhl['homeTeam']['id'], 'teamName': game_nhl['homeTeam']['commonName']['default'],

                 'teamSide': 'home'}
    away_team = {'teamId': game_nhl['awayTeam']['id'], 'teamName': game_nhl['awayTeam']['commonName']['default'],
                 'teamSide': 'away'}
    return pd.DataFrame([home_team, away_team])

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

