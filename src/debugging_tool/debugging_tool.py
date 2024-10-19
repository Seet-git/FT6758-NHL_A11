import pprint

import ipywidgets
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src.fetch.NHLData import NHLData


def plot_nhl_data(nhl_data_provider: NHLData, game_type, season):
    if game_type == 'regular':
        games_data = nhl_data_provider.regular_season[season]
    else:
        games_data = nhl_data_provider.playoffs[season]

    game_count = len(games_data)
    ipywidgets.interact(plot_game, game_number=(1, game_count, 1), games_data=ipywidgets.fixed(games_data))


def plot_game(game_number, games_data):
    game_data = games_data[game_number - 1]

    print(game_data['startTimeUTC'])
    print(
        f"Game ID: {game_number}; {game_data['homeTeam']['abbrev']} (home) vs {game_data['awayTeam']['abbrev']} (away)")

    col1 = ['', 'Teams', 'Goals', 'SoG']
    col2 = ["Home", f"{game_data['homeTeam']['abbrev']}", f"{game_data['homeTeam']['score']}",
            f"{game_data['homeTeam']['sog']}"]
    col3 = ["Away", f"{game_data['awayTeam']['abbrev']}", f"{game_data['awayTeam']['score']}",
            f"{game_data['awayTeam']['sog']}"]
    print('')
    for c1, c2, c3 in zip(col1, col2, col3):
        print(f'{c1:<18} {c2:<18} {c3:<18}')

    event_count = len(game_data['plays'])

    ipywidgets.interact(plot_game_event, event_number=(1, event_count, 1), game_data=ipywidgets.fixed(game_data))


def plot_game_event(game_data, event_number):
    event_data = game_data['plays'][event_number - 1]
    print("infos de l'événement")

    img = mpimg.imread('images/patinoire.png')
    fig, ax = plt.subplots()

    # Afficher l'image dans le fond
    ax.imshow(img, extent=[-100, 100, -42.5, 42.5], origin='lower')

    # Positionner les axes x et y aux bords (gauche pour y et bas pour x)
    ax.spines['left'].set_position(('axes', 0))  # Garder l'axe y à gauche
    ax.spines['bottom'].set_position(('axes', 0))  # Garder l'axe x en bas

    # Masquer les axes du haut et de droite
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    if 'details' in event_data and 'xCoord' in event_data['details'] and 'yCoord' in event_data['details']:
        ax.scatter(event_data['details']['xCoord'], event_data['details']['yCoord'], color="blue", s=100, zorder=5)

    y_min, y_max = plt.ylim()

    home_team_defending_side = get_home_team_side(game_data, event_data)

    if home_team_defending_side == 'right':
        home_team_position_x = 40
        away_team_position_x = -60
    else:
        home_team_position_x = -60
        away_team_position_x = 40

    plt.text(home_team_position_x, y_max, game_data['homeTeam']['abbrev'], fontsize=12, verticalalignment='bottom')
    plt.text(away_team_position_x, y_max, game_data['awayTeam']['abbrev'], fontsize=12, verticalalignment='bottom')

    plt.show()

    # on affiche les données brutes de l'événement
    pprint.pprint(event_data)


def get_home_team_initial_side(game_data):
    first_offensive_event = None
    home_team_id = game_data['homeTeam']['id']

    for event_data in game_data['plays']:
        if ('details' in event_data
                and 'zoneCode' in event_data['details']
                and event_data['details']['zoneCode'] == 'O'):
            first_offensive_event = event_data
            break

    # l'attaque se fait du côté gauche
    if first_offensive_event['details']['xCoord'] < 0:
        # le camp de l'équipe initiant l'attaque est du côté droit
        if first_offensive_event['details']['eventOwnerTeamId'] == home_team_id:
            home_team_side = 'right'
        else:
            home_team_side = 'left'
    else:
        # le camp de l'équipe initiant l'attaque est du côté gauche
        if first_offensive_event['details']['eventOwnerTeamId'] == home_team_id:
            home_team_side = 'left'
        else:
            home_team_side = 'right'

    return home_team_side


def get_home_team_side(game_data, event_data):
    home_team_initial_side = get_home_team_initial_side(game_data)

    # match de saison régulière
    if game_data['gameType'] == 2:
        # si pas en prolongation, on change de côté selon la période
        if event_data['periodDescriptor']['number'] <= 3:
            # on change de côté quand la période est paire
            if event_data['periodDescriptor']['number'] % 2 == 0:
                if home_team_initial_side == 'left':
                    home_team_initial_side = 'right'
                else:
                    home_team_initial_side = 'left'
    else:
        # on change de camp quand la période est paire
        if event_data['periodDescriptor']['number'] % 2 == 0:
            if home_team_initial_side == 'left':
                home_team_initial_side = 'right'
            else:
                home_team_initial_side = 'left'

    return home_team_initial_side
