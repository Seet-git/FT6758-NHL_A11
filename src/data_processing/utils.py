import pandas as pd


def calculate_empty_goal_net(df: pd.DataFrame) -> pd.Series:
    """
    Determine if the goal net is empty
    :param df: DataFrame containing 'situationCode' and 'teamSide' columns
    :return: Series containing boolean values
    """

    return df.apply(lambda x: x['situationCode'][3] if x['teamSide'] == 'away' else x['situationCode'][0] if len(x['situationCode']) == 4 else 0
                    , axis=1).map(
        {'0': True, '1': False})


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
