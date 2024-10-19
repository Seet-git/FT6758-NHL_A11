import base64

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
from sklearn.neighbors import KernelDensity

from src.visualisations.advancedVisualisations.utils import *


def avg_shots_per_game_per_location(years: list, total_games_per_year: list) -> dict:
    """ L’Avalanche avait terminé dernier de la ligue cette saison.
    La carte de tirs pourrait montrer une répartition des tirs peu concentrée, indiquant une difficulté à maintenir une pression offensive soutenue dans des zones de tir clés.
    Cette fonction calcule le nombre moyen de tirs par emplacement pour chaque année.
    Elle parcourt les coordonnées des tirs de chaque équipe et agrège le nombre total de tirs
    et la moyenne des tirs par emplacement. Le résultat est retourné sous forme de dictionnaire
    où chaque clé est une année, et la valeur est un DataFrame correspondant à cette année.

    :param years: Liste des années pour lesquelles calculer les tirs.
    :param total_games_per_year: Liste contenant le nombre total de matchs pour chaque année.

    :return: dict: Dictionnaire où chaque clé est une année et la valeur est un DataFrame des tirs agrégés par emplacement.
    """

    # Dictionnaire pour stocker les DataFrames pour chaque année
    year_shot_dict = {}

    # Boucle sur chaque année et calcule le DataFrame pour cette année
    for year, total_games in zip(years, total_games_per_year):

        # Dictionnaire pour stocker les tirs agrégés par emplacement
        shot_location_counts = {}
        teams_data = team_shots_coords[year]

        # Boucle sur chaque équipe
        for team in teams_data:

            # Listes des coordonnées
            x_coords = teams_data[team][0]
            y_coords = teams_data[team][1]

            # Boucle sur chaque emplacement de tir pour l'équipe
            for x, y in zip(x_coords, y_coords):

                # Si les coordonnées n'existent pas encore dans notre dictionnaire
                if (x, y) not in shot_location_counts:
                    # Ajouter les coordonnées comme clé, assigner 1 comme valeur et calculer la moyenne des tirs par match
                    shot_location_counts[(x, y)] = {"total_shots": 1, "avg_shots_per_game": 1 / total_games}
                else:
                    # Incrémenter le nombre de tirs pour un emplacement et mettre à jour la moyenne
                    shot_location_counts[(x, y)]["total_shots"] += 1
                    shot_location_counts[(x, y)]["avg_shots_per_game"] += 1 / total_games

                    # Convertir en DataFrame pour l'année actuelle
        shot_location_df = pd.DataFrame(
            [(x, y, data["total_shots"], data["avg_shots_per_game"]) for (x, y), data in shot_location_counts.items()],
            columns=['x', 'y', 'Total Shots', 'Average Shots per Game']
        )

        # Ajouter le DataFrame au dictionnaire avec l'année comme clé
        year_shot_dict[year] = shot_location_df

    return year_shot_dict


def avg_shots_per_game_per_team(year: int, games_per_team=82) -> dict:
    """
    Cette fonction calcule le nombre moyen de tirs par équipe pour une année donnée,
    en fonction du nombre de matchs joués par équipe.
    Elle retourne un dictionnaire avec lequel chaque clé est une équipe et la valeur est le nombre moyen de tirs par match.

    :param year: L'année pour laquelle calculer les tirs moyens par équipe.
    :param games_per_team: Le nombre de matchs joués par équipe (par défaut 82).

    :return: dict : Dictionnaire où chaque clé est une équipe et la valeur est le nombre moyen de tirs par match.
    """

    team_shots_per_game = {}
    teams_data = team_shots_coords[year]

    # Boucle sur chaque équipe
    for team in teams_data:
        # Compter le nombre total de tirs pour l'équipe
        total_shots = len(teams_data[team][0])

        # Calculer le nombre moyen de tirs par match
        avg_shots_per_game = total_shots / games_per_team

        # Stocker le résultat dans le dictionnaire
        team_shots_per_game[team] = avg_shots_per_game

    return team_shots_per_game


def plot_team_shots(data: dict, year: int, team_name: str):
    """
    Cette fonction trace les tirs d'une équipe spécifique dans la zone offensive pour une année donnée.
    Les coordonnées des tirs sont extraites des données, et une image de la zone offensive est affichée
    en arrière-plan. Les tirs sont ensuite superposés sous forme de points.

    :param data: Dictionnaire contenant les données des tirs par année et par équipe.
    :param year: Année pour laquelle les tirs doivent être tracés.
    :param team_name: Nom de l'équipe dont les tirs doivent être tracés.

    :return: Rien : Affiche le graphique des tirs dans la zone offensive.
    """

    # Récupérer les coordonnées à partir du dictionnaire global
    x_coords, y_coords = get_team_shots(data, year=year)[team_name]

    # Charger l'image de la zone offensive
    zone_img = cv2.imread('images/zone_offensive.png')

    # Créer un graphique
    fig, ax = plt.subplots()

    # Afficher l'image de la zone offensive
    extent = [-42.5, 42.5, 0, 100]  # Dimensions x et y de la zone offensive
    ax.imshow(zone_img, extent=extent)

    # Définir les limites des axes
    ax.set_xlim(-42.5, 42.5)
    ax.set_ylim(0, 100)

    # Tracer les tirs filtrés
    ax.scatter(x_coords, y_coords, color='orange', s=5, label='Tirs')

    # Ajouter un titre et une légende (facultatif)
    ax.set_title(f'Tirs de {team_name} dans la zone offensive ({year})')
    ax.legend()

    # Afficher le graphique
    plt.show()


def heatmap(shot_data: pd.DataFrame, bins: int):
    """
    Cette fonction génère une heatmap des tirs en zone offensive,
    en filtrant les coordonnées valides et en les agrégeant dans un histogramme 2D.
    Une image de la zone offensive est affichée en arrière-plan, et la densité des tirs
    est superposée sous forme de heatmap.

    :param shot_data: DataFrame contenant les coordonnées des tirs et la moyenne de tirs par match par emplacement.
    :param bins: Le nombre de divisions dans chaque direction (x et y) pour l'histogramme 2D.

    :return: Rien : Affiche la heatmap des tirs.
    """

    # Filtrer les données avec lesquelles y < 0
    valid_data = shot_data[(shot_data['y'] > 0) & ((shot_data['x'] != 0) | (shot_data['y'] != 0))]
    valid_data = valid_data.dropna(subset=['x', 'y', 'Average Shots per Game'])

    # Extraire les données valides après filtrage
    x_coords = valid_data['x']
    y_coords = valid_data['y']
    avg_shots_per_game = valid_data['Average Shots per Game']

    # Charger l'image de la zone offensive
    zone_img = cv2.imread('images/zone_offensive.png')

    # Créer un histogramme 2D
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins, weights=avg_shots_per_game)

    # Créer un graphique
    fig, ax = plt.subplots()

    # Afficher l'image de la zone offensive
    extent = [-42.5, 42.5, 0, 100]
    ax.imshow(zone_img, extent=extent, aspect='auto')

    # Créer la carte de chaleur
    im = ax.imshow(
        heatmap.T, extent=extent, origin='lower',
        cmap='Reds', alpha=0.7
    )

    # Ajouter une barre de couleur pour indiquer l'intensité des tirs
    cbar = plt.colorbar(im)
    cbar.set_label('Moyenne de tirs par partie (60 minutes)')

    # Ajouter un titre (facultatif)
    ax.set_title("Zone de tirs de la ligue en zone offensive (2018)")

    # Afficher le graphique
    plt.show()


def smooth_heatmap(shot_data: pd.DataFrame, bandwidth=1.0, grid_size=100):
    """
    Cette fonction génère une carte de densité des tirs lissée à l'aide d'une estimation par noyau gaussien (Kernel Density Estimation).
    Elle filtre les données invalides et calcule une estimation de densité pour afficher une heatmap des tirs dans la zone offensive.

    :param shot_data: DataFrame contenant les coordonnées des tirs et la moyenne de tirs par partie par emplacement.
    :param bandwidth: Paramètre de lissage pour le noyau gaussien (par défaut 1.0).
    :param grid_size: Taille de la grille pour l'évaluation de la densité (par défaut 100).

    :return: Rien : Affiche la carte de densité lissée des tirs.
    """

    # Filtrer les données erronées (coordonnées 0, 0) et les données avec lesquelles y < 0
    valid_data = shot_data[(shot_data['y'] > 0) & ((shot_data['x'] != 0) | (shot_data['y'] != 0))]
    valid_data = valid_data.dropna(subset=['x', 'y', 'Average Shots per Game'])

    # Extraire les données valides
    x_coords = valid_data['x'].values
    y_coords = valid_data['y'].values
    avg_shots_per_game = valid_data['Average Shots per Game'].values

    # Calculer le nombre total de tirs
    total_shots = np.sum(avg_shots_per_game)

    # Créer une grille sur la patinoire
    x_min, x_max = -42.5, 42.5
    y_min, y_max = 0, 100
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    # Empiler les coordonnées de la grille pour l'évaluation
    xy_sample = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Combiner les données pour l'estimation par noyau (pondération par la moyenne de tirs par partie)
    shot_coords = np.vstack([x_coords, y_coords]).T

    # Appliquer l'estimation de densité par noyau avec un noyau gaussien
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(shot_coords, sample_weight=avg_shots_per_game)

    # Évaluer et ajuster la densité sur la grille
    z_density = np.exp(kde.score_samples(xy_sample)).reshape(x_grid.shape) * total_shots

    # Charger l'image de la zone offensive
    zone_img = cv2.imread('images/zone_offensive.png')

    # Tracer la carte de densité lissée
    fig, ax = plt.subplots()

    # Afficher l'image
    extent = [x_min, x_max, y_min, y_max]
    ax.imshow(zone_img, extent=extent, aspect='auto')

    # Propriétés de la carte
    kde_plot = ax.imshow(
        z_density, extent=extent, origin='lower', cmap='coolwarm', alpha=0.6
    )

    # Affichage de la barre de couleurs et du titre
    cbar = plt.colorbar(kde_plot)
    cbar.set_label('Nombre estimé de tirs par partie par pied carré')
    ax.set_title(f'Carte de densité des tirs lissée pour la ligue')
    plt.show()


def interactive_smooth_heatmap(year: int, bandwidth=1.0, grid_size=150):
    """
    Cette fonction génère une carte de densité des tirs lissée de manière interactive pour chaque équipe
    d'une année donnée. Chaque équipe a une heatmap basée sur une estimation par noyau (KDE),
    et l'utilisateur peut interagir avec le menu déroulant pour changer d'équipe et voir la carte correspondante.

    :param year: Année pour laquelle générer les cartes de densité.
    :param bandwidth: Paramètre de lissage pour l'estimation par noyau (par défaut 1.0).
    :param grid_size: Taille de la grille pour générer la carte (par défaut 150).

    :return: Rien : Affiche une carte interactive de densité des tirs lissée pour chaque équipe.
   """

    # Variables
    teams_data = team_shots_coords[year]
    team_names = list(teams_data.keys())

    # Créer la grille sur la patinoire
    x_min, x_max = -42.5, 42.5
    y_min, y_max = 0, 100
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )
    xy_sample = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Encoder l'image en base 64 pour plotly
    zone_img = cv2.imread('images/zone_offensive.png')
    _, buffer = cv2.imencode('.png', zone_img)
    zone_img_base64 = base64.b64encode(buffer).decode('utf-8')

    fig = go.Figure()

    # Itérer à travers chaque équipe pour générer les heatmaps
    for team in team_names:
        # Extraire les coordonnées des tirs
        x_coords, y_coords = teams_data[team]
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        # Filtrer les données pour exclure les coordonnées (0, 0) et hors de la zone offensive (y < 0)
        valid_indices = (y_coords > 0) & ((x_coords != 0) | (y_coords != 0))
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]

        # Processus nécessaire pour KDE
        total_shots = len(x_coords)
        shot_coords = np.vstack([x_coords, y_coords]).T
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(shot_coords)

        # Densité de la grille multipliée par le nombre total de tirs pour avoir le nombre de tirs estimés (et non la probabilité)
        z_density = np.exp(kde.score_samples(xy_sample)).reshape(x_grid.shape) * total_shots

        # Propriétés de la carte
        heatmap_trace = go.Heatmap(
            z=z_density,
            x=np.linspace(x_min, x_max, grid_size),
            y=np.linspace(y_min, y_max, grid_size),
            colorscale='RdBu_r',
            zmin=0,
            zmax=np.max(z_density),
            opacity=0.4,
            visible=False,
            name=team
        )
        fig.add_trace(heatmap_trace)

    # Rendre la première carte visible
    fig.data[0].visible = True

    # Menu dropdown
    dropdown_buttons = [
        {
            'label': team,
            'method': 'update',
            'args': [{'visible': [team == trace.name for trace in fig.data]},
                     {'title': f"Carte de densité des tirs lissée pour {team} ({year})"}]
        }
        for team in team_names
    ]

    # Configurer le layout et les propriétés de l'image de background (patinoire)
    fig.update_layout(

        title=f"Carte de densité des tirs lissée interactive ({year})",

        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True
            }
        ],

        xaxis=dict(title='', range=[x_min, x_max]),
        yaxis=dict(title='', range=[y_min, y_max]),

        images=[dict(
            source='data:image/png;base64,' + zone_img_base64,
            xref="x",
            yref="y",
            x=x_min,
            y=y_max,
            sizex=x_max - x_min,
            sizey=y_max - y_min,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],

        coloraxis_colorbar=dict(
            title="Tirs estimés par pied carré"
        ),

        # Modifier manuellement la taille de l'image
        autosize=False,
        width=700,
        height=750
    )

    fig.show()


def mean_shots_game_ligue(df, column1, column2, eventId):
    nb = df.groupby([column1, column2])[eventId].nunique()
    mean_nb = round(nb.groupby(level=0).mean(), 2)
    mean_nb_df = mean_nb.reset_index()  # Convertir l'index en colonne
    mean_nb_df.columns = [column1, 'Taux tir moyen pour la Ligue']  # Nommer les colonnes
    return mean_nb_df


def mean_shots_game_team(df, year, column1, column2, event_id, df_ligue):
    df_year = df[df['Year'] == str(year)]
    grouped_team_game = df_year.groupby([column1, column2])
    nb_shots_team_game = grouped_team_game[event_id].nunique()
    mean_nb_shots_year_team_game = nb_shots_team_game.groupby(level=0).mean()
    mean_nb_shots_year_team_game = pd.DataFrame(mean_nb_shots_year_team_game)
    mean_nb_shots_year_team_game.columns = [f'# Tir moyen par match en {year}']
    mean_ligue_year_team = df_ligue['Taux tir moyen pour la Ligue'].values[year - 2016] / 2
    mean_nb_shots_year_team_game['Diff. # tir moyen / La ligue'] = mean_nb_shots_year_team_game - mean_ligue_year_team
    mean_nb_shots_year_team_game['% Diff. # tir moyen / La ligue'] = (
            mean_nb_shots_year_team_game['Diff. # tir moyen / La ligue'] * 100.0 / mean_ligue_year_team).apply(
        lambda x: str(round(x, 5)) + '%')
    mean_nb_shots_year_team_game.index.name = 'Equipe'
    mean_nb_shots_year_team_game.sort_values(by=f'# Tir moyen par match en {year}', ascending=False, inplace=True)
    mean_nb_shots_year_team_game.plot(kind='bar')
    return mean_nb_shots_year_team_game


def estimation_kde_noyau_gaussien(year, mean_df):
    sns.kdeplot(data=mean_df,
                x=f'# Tir moyen par match en {year}',
                multiple="stack", warn_singular=False
                )
    plt.title(f"Estimation de la densité des tirs par heure en {year}")
    plt.xlabel("Taux de tir moyen par heure")
    plt.ylabel("Densité")
    plt.show()
