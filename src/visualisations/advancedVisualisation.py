import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series


def get_coordinates(row: Series, side: str) -> tuple:
    previous_period = '1/3'
    new_coords = (0, 0)
    coords = row['iceCoord']  # Get the tuple from the DataFrame
    currentPeriod = row['currentPeriod']

    # If the team is on the left side of the ice
    if side == 'left':

        # If the period has not change
        if currentPeriod == previous_period:
            new_coords = (coords[1], -coords[0])  # Rotate coordinates 90 degrees

        # If the period has changed
        else:

            previous_period = currentPeriod  # Update the previous period

            # If its not overtime during the regular season, update team side and rotate accordingly
            if not (previous_period == '4/3' and str(row['idGame'])[4:6] == '02'):
                side = 'right'  # Switch the team side
                new_coords = (coords[1], -coords[0])  # Rotate coordinates -90 degrees

    # If the team is on the left side of the ice
    else:

        # If the period has not change
        if currentPeriod == previous_period:
            new_coords = (-coords[1], coords[0])  # Rotate coordinates -90 degrees

        # If the period has changed
        else:

            previous_period = currentPeriod  # Update the previous period

            # If its not overtime during the regular season, update team side and rotate accordingly
            if not (previous_period == '4/3' and str(row['idGame'])[4:6] == '02'):
                side = 'left'  # Switch the team side
                new_coords = (coords[1], -coords[0])  # Rotate coordinates 90 degrees
    return new_coords


def get_team_shoot(data: dict, year: int, team_name: str) -> list:
    # Récupérer le nom de chaque équipe de l'année
    teams_regular_season = pd.concat([df['eventOwnerTeam'] for df in data[year]])
    unique_teams = teams_regular_season.unique()

    # Dictionary to store the shots per team
    team_shots_dfs = {}
    dfs_list = data[year]  # Simplify

    # Loop over each team and filter rows accordingly
    for team in unique_teams:
        # Creates a DataFrame with only the shots made by a specific team
        team_df = pd.concat([df[df['eventOwnerTeam'] == team] for df in dfs_list])

        # Add the new DataFrame to the team_shots_dfs dictionary
        team_shots_dfs[team] = team_df

    # Testing on a specific DataFrame
    df = team_shots_dfs[team_name].copy()

    coords_list = []  # List to store the updated coordinates as tuples

    # Finding on which side the team is starting the game
    first_offensive_zone_event = df[df['zoneShoot'] == 'O'].iloc[0]
    side = 'left' if first_offensive_zone_event['iceCoord'][0] < 0 else 'right'

    # Modify the coordinates to fit the offensive zone
    for _, row in df.iterrows():
        # Append the new coordinates to the coords_list
        coords_list.append(get_coordinates(row, side))

    # Extract the new coordinates
    x_coords = [coord[0] for coord in coords_list]
    y_coords = [coord[1] for coord in coords_list]

    return x_coords, y_coords


def plot_team_shots(data: dict, year: int, team_name: str):
    x_coords, y_coords = get_team_shoot(data, year, team_name)

    # Load the offensive zone
    zone_img = cv2.imread('images/zone_offensive.png')

    # Create a plot
    fig, ax = plt.subplots()

    # Display the offensive zone image
    extent = [-42.5, 42.5, 0, 100]
    ax.imshow(zone_img, extent=extent)

    # Set axis limits
    ax.set_xlim(-42.5, 42.5)
    ax.set_ylim(0, 100)

    # Plot the filtered shots
    ax.scatter(x_coords, y_coords, color='orange', s=5, label='Shots')

    # Add labels and title (optional)
    ax.set_title('Team Shots in Offensive Zone')
    ax.legend()

    # Show the plot
    plt.show()
