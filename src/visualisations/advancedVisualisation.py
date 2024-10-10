import cv2
import matplotlib.pyplot as plt
import pandas as pd


def get_team_shoot(data: dict, year: int, team_name: str):
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

    # Modify the coordinates to fit the offensive zone
    for index, row in df.iterrows():
        coords = row['iceCoord']  # Get the tuple from the DataFrame

        if row['teamSide'] == 'home':
            # Rotate -90 degrees if home team is defending left side, else rotate 90 degrees
            new_coords = (-coords[1], coords[0]) if row['homeTeamDefendingSide'] == 'left' else (coords[1], -coords[0])

        else:
            # Rotate 90 degrees if home team is defending left side, else rotate -90 degrees
            new_coords = (coords[1], -coords[0]) if row['homeTeamDefendingSide'] == 'left' else (-coords[1], coords[0])

        # Append the new coordinates to the coords_list
        coords_list.append(new_coords)

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
