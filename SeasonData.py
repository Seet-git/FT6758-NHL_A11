import requests
import json
from os.path import join, isfile, exists
from os import makedirs
from enum import Enum


# Différents NHL game types
class GameType(Enum):
    PRESEASON = '01'
    REGULAR_SEASON = '02'
    PLAYOFF = '03'
    ALL_STAR = '04'


class NHLData:

    # Constructor
    def __init__(self):
        """
        Initialize the NHLData object.
        """

        self.playoff = {}
        self.regular_season = {}

    def _fetch_data(self, game_type: GameType, season: str):
        """
        Fetches play-by-play game data for a season and stores it locally.
        The function continues fetching data until a non-200 response is received from the API.
        The fetched data is saved to local JSON files, and if the file exists locally, it will be loaded from there.

        Attributes:
            game_type (GameType): The game type (Regular season or playoff)
            self.games_list (list): A list to store game data for each game in the season.
        """

        path_directory = ""  # Path to store data

        # Directory based on the game type
        if game_type == GameType.REGULAR_SEASON:
            path_directory = "data/regular_season"
        elif game_type == GameType.PLAYOFF:
            path_directory = "data/playoff"
        else:
            raise NotImplementedError("Unsupported game type.")

        # Ensure directory exists to store the JSON files locally
        if not exists(path_directory):
            makedirs(path_directory)

        game = "0001"  # Initialize the first game number as a zero-padded string
        games_list = []  # List to store all game data for the season
        nb_data = 0  # Number of successful data imports

        # Continuously fetch data until a non-200 response is received
        while True:

            # Construct the local file path based on season and game number
            local_file = join(path_directory, f"{season}_{game}.json")

            # API URL to fetch play-by-play data for the given game and season
            url = f"https://api-web.nhle.com/v1/gamecenter/{season}02{game}/play-by-play"

            # Check if the data file exists locally
            if not isfile(local_file):

                # Fetch data from the API
                response = requests.get(url)

                # Check if the API request was successful (status code 200)
                if response.status_code == 200:

                    # Parse the response JSON data
                    data = response.json()

                    # Write the fetched data to the local file in JSON format
                    with open(local_file, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)  # Save with proper formatting
                        print(f"Data was successfully imported: {local_file}")
                    nb_data += 1  # Increment the count of fetched data.

                # If the API request fails (non-200 response), break the loop
                else:
                    print(f"Data imported: {nb_data}")
                    break

            # Load the game data from the local file (either newly fetched or pre-existing)
            with open(local_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                games_list.append(data)  # Append the game data to the games_list

            # Increment the game number, keeping it zero-padded to 4 digits (e.g., 0002, 0003, etc.)
            game = f"{int(game) + 1:04d}"

        # Store the list of game data in the appropriate class attribute for later use
        if game_type == GameType.REGULAR_SEASON:
            self.regular_season[season] = games_list
        elif game_type == GameType.PLAYOFF:
            self.playoff[season] = games_list

    def fetch_regular_season(self, year: int):
        """
        Fetch regular season data for a specific year.
        :param year: The season year
        """
        NHLData._fetch_data(self, game_type=GameType.REGULAR_SEASON, season=str(year))

    def fetch_playoff(self, year: int):
        """
        Fetch playoff season data for a specific year.
        :param year: The season year
        """
        NHLData._fetch_data(self, game_type=GameType.PLAYOFF, season=str(year))