import pandas as pd

from src.visualisations.advancedVisualisations.globals import team_shots_coords


def get_coordinates(row: pd.Series, home_team_initial_side: str) -> list:
    """
    Cette fonction calcule les nouvelles coordonnées d'un tir en fonction de la période et du côté de la patinoire
    sur lequel l'équipe commence. Elle prend en compte les rotations des coordonnées en fonction du changement de côté
    de l'équipe après chaque période.

    :param row: Une ligne d'un DataFrame contenant les données de tir, y compris les coordonnées et la période.
    :param side: Le côté initial de l'équipe (gauche ou droite).

    :return: tuple: Les nouvelles coordonnées après ajustement basé sur le côté et la période.
    """

    initial_side = None

    if(row['teamSide'] == 'home'):
        initial_side = home_team_initial_side
    else:
        if home_team_initial_side == 'left':
            initial_side = 'right'
        else:
            initial_side = 'left'


    new_coords = [(0, 0), (0,0)]
    current_side = initial_side

    # en saison réguliere, on change de cote seulement si on n'est pas en prolongation
    if str(row['idGame'])[4:6] == '02' and row['numberPeriod'] <= 3:
        if row['numberPeriod'] % 2 == 0:
            # on change le camp
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'
    else:
        if row['numberPeriod'] % 2 == 0:
            # on change le camp
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'

    if current_side == 'left':
        # je fais une rotation de 90 degre dans le sens inverse des aiguilles d'une montre
        new_coords = [(-row['yCoord'], row['xCoord']), (-row['previousYCoord'], row['previousXCoord'])]
    else:
        # je fais une rotation de 90 degre dans le sens des aiguilles d'une montre
        new_coords = [(row['yCoord'], -row['xCoord']), (row['previousYCoord'], -row['previousXCoord'])]

    return new_coords


def get_team_shots(regular_season: dict = None, playoff:dict = None, year: int = 2020) -> dict:
    """
    Cette fonction récupère les coordonnées des tirs effectués par chaque équipe pour une année donnée.
    Elle ajuste les coordonnées en fonction du côté de la patinoire avec laquelle l'équipe commence le match et les stocke
    dans un dictionnaire par équipe.

    :param regular_season: Dictionnaire contenant les données en saison régulière des tirs pour chaque année.
    :param playoff: Dictionnaire contenant les données en playoff des tirs pour chaque année.
    :param year: Année pour laquelle récupérer les tirs.

    :return: dict : Dictionnaire où chaque clé est une équipe et la valeur est un tuple de listes des coordonnées x et y des tirs.

    """

    # Vérifier si les données pour cette année ont déjà été calculées dans la variable globale
    if year in team_shots_coords:
        return team_shots_coords[year]  # Retourner les données déjà calculées

    # Dictionnaire pour stocker les tirs par équipe pour l'année donnée
    year_shots_coords = {}
    dfs_combined = regular_season[year]
    if playoff is not None:
        dfs_combined += playoff[year]

    # Récupérer le nom de chaque équipe
    df_teams = pd.concat([df['eventOwnerTeam'] for df in dfs_combined])
    unique_teams = df_teams.unique()

    # Boucle sur chaque équipe
    for team in unique_teams:

        # Crée un DataFrame avec uniquement les tirs effectués par une équipe spécifique
        team_df = pd.concat([df[df['eventOwnerTeam'] == team] for df in dfs_combined])

        # Liste pour stocker les coordonnées mises à jour sous forme de tuples
        coords_list = []

        # Trouver de quel côté de la patinoire l'équipe commence le match
        first_offensive_zone_event = team_df[team_df['zoneShoot'] == 'O'].iloc[0]
        side = 'right' if first_offensive_zone_event['iceCoord'][0] < 0 else 'left'

        # Modifier les coordonnées pour correspondre à la zone offensive
        for _, row in team_df.iterrows():
            ice_coord = row['iceCoord']  # Get the tuple
            if ice_coord is not None and all(pd.notnull(coord) for coord in ice_coord):
                coords_list.append(get_coordinates(row, side))

        # Extraire les nouvelles coordonnées
        x_coords = [coord[0] for coord in coords_list]
        y_coords = [coord[1] for coord in coords_list]

        # Ajouter les coordonnées x et y sous forme de tuple au dictionnaire pour cette équipe
        year_shots_coords[team] = (x_coords, y_coords)

    # Stocker les données pour l'année dans le dictionnaire global
    team_shots_coords[year] = year_shots_coords

    return year_shots_coords



