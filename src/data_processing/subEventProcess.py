import pandas as pd


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

    # Add 'currentPeriod' column
    df_period['currentPeriod'] = df_period['number'] + '/' + df_period['maxRegulationPeriods']

    df_period['numberPeriod'] = df_period['number']

    return df_period


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

def process_previous_event(df: pd.DataFrame):
    # Décale le dataframe pour récupérer l'élément précédent
    df_copy = df.copy().shift(1)
    # Get previous event
    df['previousEventType'] = df_copy['typeDescKey']

    # Get previous time
    df['timeSinceLastEvent'] = df['Game Seconds'].diff()
    df['timeSinceLastEvent'] = df.apply(lambda x: 0
    if pd.isnull(x['timeSinceLastEvent']) else abs(x['timeSinceLastEvent']), axis=1) #Abs to prevent some error or negative time

    # Get previous coordinates
    details = df_copy['details'].apply(pd.Series)
    df["previousXCoord"] = details['xCoord']
    df["previousYCoord"] = details['yCoord']

    return df

