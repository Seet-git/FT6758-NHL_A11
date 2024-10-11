from src.data_processing.additionalFeatures import *
from src.data_processing.extractRoster import *
from src.data_processing.subEventProcess import *
from src.data_processing.utils import *
from src.fetch.NHLData import NHLData


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
    clean_df = pd.DataFrame(df_pbp[['periodDescriptor', 'timeInPeriod', 'situationCode', "homeTeamDefendingSide",
                                    'typeDescKey', 'details']])

    # PERIOD DESCRIPTOR
    # Create a dataframe to decompose the period descriptor fields
    df_period = process_period_data(clean_df)
    clean_df.drop('periodDescriptor', axis=1, inplace=True)

    # Add 'gameID', 'periodType' and 'currentPeriod' columns to the new dataframe
    clean_df.insert(0, 'idGame', game_nhl['id'])
    clean_df.insert(1, 'periodType', df_period['periodType'])
    clean_df.insert(2, 'currentPeriod', df_period['currentPeriod'])

    # TIME IN PERIOD
    # Parse the game's start time to UTC
    start_time = pd.to_datetime(game_nhl['startTimeUTC'])

    # Convert time in the period to seconds
    time_series = start_time + pd.to_timedelta(minutes_to_seconds(clean_df, 'timeInPeriod'), unit='s')

    # add it to the game start time
    clean_df['timeInPeriod'] = time_series

    # Filter to keep only events of type 'shot-on-goal' or 'goal'
    clean_df = clean_df[(clean_df['typeDescKey'] == 'shot-on-goal') | (clean_df['typeDescKey'] == 'goal')].reset_index(
        drop=True)

    # DETAILS
    # Process event details and merge player information
    df_details = process_event_details(clean_df, df_players)
    clean_df.drop('details', axis=1, inplace=True)

    # Add team data by merging IDs
    df_details = pd.merge(df_teams, df_details, left_on='teamId', right_on='eventOwnerTeamId', how='right')

    # Add the extracted data to the new dataframe
    clean_df['iceCoord'] = df_details['iceCoord']
    clean_df['shootingPlayer'] = df_details['shootingPlayer']
    clean_df['goaliePlayer'] = df_details['goaliePlayer']
    clean_df['shotType'] = df_details['shotType']
    clean_df.insert(5, 'eventOwnerTeam', df_details['teamName'])
    clean_df['teamSide'] = df_details['teamSide']

    # Calculate emptyGoalNet and goal advantage
    clean_df['emptyGoalNet'] = calculate_empty_goal_net(clean_df)
    clean_df['isGoalAdvantage'] = determine_goal_advantage(clean_df)

    # Add shot distance
    clean_df = additional_features(clean_df)

    # Drop situation code
    clean_df.drop('situationCode', axis=1, inplace=True)

    return clean_df


def clean_data(raw_data: NHLData) -> tuple:
    """
    Clean the raw data and return a dictionary with all the games
    :param raw_data: NHLData object containing the raw data
    :return: A tuple containing two dictionaries (regular season and playoff) with all the games cleaned
    """

    # REGULAR SEASON
    regular_season = {}  # Initialize the return dictionary

    # Loop through all the years in the regular season data
    for year in raw_data.regular_season.keys():
        yearly_data = raw_data.regular_season[year]

        # Add all the cleaned dataframes (shots per game) for a year to the dictionary
        regular_season[year] = [convert_game_to_dataframe(game) for game in yearly_data]

    # PLAYOFF
    playoff = {}  # Initialize the return dictionary

    # Loop through all the years in the playoff data
    for year in raw_data.playoffs.keys():
        yearly_data = raw_data.playoffs[year]

        # Add all the cleaned dataframes (shots per game) for a year to the dictionary
        playoff[year] = [convert_game_to_dataframe(game) for game in yearly_data]

    return regular_season, playoff
