import requests
import json
from os.path import join, isfile, exists
from os import makedirs

class RegularSeasonData:
    
    # Constructor
    def __init__(self, season, auto_fetch=False):
        
        """
        Initialize the RegularSeasonData object.

        Args:
            season (int): The season year.
            auto_fetch (bool): If True, automatically fetch the data on initialization.
        """
        
        self.season = season
        self.games_list = []
        
        if auto_fetch:
            self.fetchData
    
    def fetchData(self):
        
        """
        Fetches play-by-play game data for a season and stores it locally. 
        The function continues fetching data until a non-200 response is received from the API.
        The fetched data is saved to local JSON files, and if the file exists locally, it will be loaded from there.

        Attributes:
            self.games_list (list): A list to store game data for each game in the season.
        """

        # Ensure the "data" directory exists to store the JSON files locally
        if not exists("data"):
            makedirs("data")

        game = "0001"  # Initialize the first game number as a zero-padded string
        games_list = []  # List to store all game data for the season

        # Continuously fetch data until a non-200 response is received
        while True:

            # Construct the local file path based on season and game number
            local_file = join("data", f"{self.season}_{game}.json")

            # API URL to fetch play-by-play data for the given game and season
            url = f"https://api-web.nhle.com/v1/gamecenter/{self.season}02{game}/play-by-play"

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

            # Load the game data from the local file (either newly fetched or pre-existing)
            with open(local_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                games_list.append(data)  # Append the game data to the games_list

            # Increment the game number, keeping it zero-padded to 4 digits (e.g., 0002, 0003, etc.)
            game = f"{int(game) + 1:04d}"

        # Store the list of game data in the class attribute for later use
        self.games_list = games_list