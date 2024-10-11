import pandas as pd


def process_period_data(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose periodDescriptor and return processed dataframe."""

    # Split 'periodDescriptor' into 'periodType', 'number', and 'maxRegulationPeriods'
    df_period = pd.DataFrame(df['periodDescriptor'].tolist())

    # Convert 'number' and 'maxRegulationPeriods' columns as strings
    df_period[['number', 'maxRegulationPeriods']] = df_period[['number', 'maxRegulationPeriods']].astype(str)

    # Add 'currentPeriod' column
    df_period['currentPeriod'] = df_period['number'] + '/' + df_period['maxRegulationPeriods']

    return df_period


def process_event_details(df: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
    """Process event details and merge with player information."""
    df_details = pd.DataFrame(df['details'].tolist())

    # Combine x and y coordinates into a tuple
    df_details['iceCoord'] = df_details[['xCoord', 'yCoord']].apply(tuple, axis=1)

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

    # Add the goalies names by merging IDs
    df_details = pd.merge(df_players, df_details, left_on='playerId', right_on='goalieInNetId', how='right').drop(
        columns=['playerId'])

    # Keep only full name
    df_details['goaliePlayer'] = df_details['firstName'] + ' ' + df_details['lastName']
    df_details.drop(['firstName', 'lastName'], axis=1, inplace=True)

    return df_details
