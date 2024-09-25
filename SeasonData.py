import requests
import json
from os.path import join, isfile, exists
from os import makedirs

class NHLData:

    # Constructor
    def __init__(self):
        """
        Initialize the NHLData object.
        """

        self.playoffs = {}
        self.regular_season = {}

    def fetch_regular_season(self, year: str):
        """
        Fetch regular season data for a specific year.
        :param year: The season year
        """

        """
        Fetches play-by-play regular season game data for a season and stores it locally.
        The function continues fetching data until a non-200 response is received from the API.
        The fetched data is saved to local JSON files, and if the file exists locally, it will be loaded from there.
        The loaded data is added to dictionary self.regular_season, where the key is the year/season of the regular season.
        """

        # Path directory to save files
        path_directory = f"data/regular_season/{year}"

        # Ensure directory exists to store the JSON files locally
        if not exists(path_directory):
            makedirs(path_directory)

        game = "0001"  # Initialize the first game number as a zero-padded string
        games_list = []  # List to store all game data for the season
        nb_data = 0  # Number of successful data imports

        # Continuously fetch data until a non-200 response is received
        while True:

            # Construct the local file path based on season and game number
            local_file = join(path_directory, f"{year}_{game}.json")

            # API URL to fetch play-by-play data for the given game and season
            url = f"https://api-web.nhle.com/v1/gamecenter/{year}02{game}/play-by-play"

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
                    
                # If the API request fails (non-200 response), break the loop
                else:
                    break
            else:
                # Load the game data from the local file
                with open(local_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
            nb_data += 1  # Increment the count of fetched data. 
            games_list.append(data)  # Append the game data to the games_list
            game = f"{int(game) + 1:04d}" # Increment the game number, keeping it zero-padded to 4 digits (e.g., 0002, 0003, etc.)
            
        # Store the list of game data in the appropriate class attribute for later use
        self.regular_season[year] = games_list
        print(f"Data imported: {nb_data}")
        

    def fetch_playoffs(self, year: str):
        """
        Fetch playoffs data for a specific year.
        :param year: The season year
        """

        """
        Fetches play-by-play playoffs game data for a season and stores it locally.
        The function continues fetching data until the final playoff round (4).
        The fetched data is saved to local JSON files, and if the file exists locally, it will be loaded from there.
        The loaded data is added to dictionary self.playoffs, where the key is the year/season of the playoffs. 
        """
        
        # Path directory to save files
        path_directory = f"data/playoffs/{year}"
        
        # Ensure directory exists to store the JSON files locally
        if not exists(path_directory):
            makedirs(path_directory)
        
        game = "0111"  # Initialize the first playoff game as number as a zero-padded string
        games_list = []  # List to store all game data for the season
        nb_data = 0  # Number of successful data imports
        
        # Goes through all the games, rounds and matchups in the playoffs
        while int(game[1]) < 5:
            
            # Construct the local file path based on season and game number
            local_file = join(path_directory, f"{year}_{game}.json")
            
            # API URL to fetch play-by-play data for the given game and season
            url = f"https://api-web.nhle.com/v1/gamecenter/{year}03{game}/play-by-play"
            
            # Verify if the data file exists locally
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
                
                # If the API request failed, update the last 4 digits of the game id
                else:
                    game = self.__generate_playoff_id(game) # Generate a new game_id
                    continue # skip to the next loop
            
            # Load the game data from the local file (either newly fetched or pre-existing)
            with open(local_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                games_list.append(data)  # Append the game data to the games_list
                nb_data += 1  # Increment the count of fetched data.
            
            # Increment the game number, keeping it zero-padded to 4 digits (e.g., 0002, 0003, etc.)
            game = f"{int(game) + 1:04d}"
            
        # Store the list of game data in the appropriate class attribute for later use
        self.playoffs[year] = games_list
        print(f"Data imported: {nb_data}")
            
    def __generate_playoff_id(self, playoff_id : str) -> str:
            
        # Convert the game_id into individual components
        prefix = playoff_id[0]  # Keep the first digit (always '0')
        round_digit = int(playoff_id[1])  # 2nd digit gives the round
        matchup_digit = int(playoff_id[2])  # 3rd digit gives the matchup
        game_digit = int(playoff_id[3])  # 4th digit gives the game number in the series
        
        # Reset game digit to one
        game_digit = 1
        
        # Conditions to update the matchup digit
        if (round_digit == 1 and matchup_digit < 8) or (round_digit == 2 and matchup_digit < 4) or (round_digit == 3 and matchup_digit < 2):
            matchup_digit += 1
        # Increment the round digit and reset the matchup to one
        else:
            round_digit += 1 
            matchup_digit = 1

        # Build the new game_id as a string with the format '0XXX'
        new_playoff_id = f"{prefix}{round_digit}{matchup_digit}{game_digit}"

        return new_playoff_id