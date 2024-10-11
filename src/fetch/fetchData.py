from src.fetch.NHLData import NHLData


def get_data_from(first_year: int, last_year: int) -> NHLData:
    """
    Retrieves game data for each season within the specified range of years, processes the events of each game
    (focusing on shots), and organizes the information into a dictionary. The dictionary's keys are the years,
    and the corresponding values are lists of DataFrames, where each DataFrame represents all the shots taken
    in a particular game during that year.

    Parameters:
    - first (int): The first year (inclusive) from which to retrieve data.
    - last (int): The last year (inclusive) from which to retrieve data.

    Returns:
    - dict: A dictionary where the keys are years, and the values are lists of DataFrames. Each DataFrame corresponds
    to a single game in the specified year and contains information about the shots taken during that game.

    """

    raw_data = NHLData()  # Initialize the data object

    # Loop over all years
    for year in range(first_year, last_year + 1):
        # Generate the data from API or local if available
        raw_data.fetch_regular_season(year)
        raw_data.fetch_playoffs(year)

    return raw_data
