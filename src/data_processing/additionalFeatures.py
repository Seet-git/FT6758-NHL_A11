import numpy as np
import pandas as pd

from src.visualisations.advancedVisualisations.advancedVisualisation import get_coordinates


def angle_between_vectors(v1: np.array, v2: np.array) -> float:
    """
    Calculate the angle in degrees between two 2D vectors.
    :param v1: np.array Vector 1
    :param v2: np.array Vector 2
    :return: float Angle in degrees
    """
    # Dot product
    dot_product = np.dot(v1, v2)

    # norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # cosine of the angle
    cos_angle = dot_product / (norm_v1 * norm_v2)

    # Clip the value to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Angle in radians
    angle_radians = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def additional_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features to the cleaned DataFrame.
    :param clean_df: Pd.DataFrame Cleaned DataFrame
    :return: pd.Dataframe with additional features
    """

    # List to store the updated coordinates as tuples
    coords_list = []  # List to store the updated coordinates as tuples

    # Finding on which side the team is starting the game
    first_home_team_offensive_event = clean_df[(clean_df['zoneShoot'] == 'O') & (clean_df['teamSide'] == 'home')].iloc[
        0]
    home_team_initial_side = 'right' if first_home_team_offensive_event['xCoord'] < 0 else 'left'

    for _, row in clean_df.iterrows():
        # Calculer les nouvelles coordonnées selon le côté et la période
        new_coords = get_coordinates(row, home_team_initial_side)
        coords_list.append(new_coords)  # Stocker les coordonnées ajustées

    # Ajouter les nouvelles coordonnées à la DataFrame
    clean_df['adjustedCoord'] = coords_list

    # ADDITIONAL FEATURES

    # Add shot distance

    # Define the Euclidian distance function
    dist_euclidian = lambda x1, x2: np.round(np.linalg.norm(np.array(x1 - x2)), decimals=1)

    # Add shot distance based on the ice coordinates
    clean_df['shotDistance'] = clean_df.apply(lambda x: dist_euclidian(x['adjustedCoord'], np.array([0, 89])), axis=1)

    # Add distance from the last event
    clean_df['distanceFromLastEvent'] = clean_df.apply(
            lambda x: dist_euclidian(np.array([x['previousXCoord'], x["previousYCoord"]]), np.array([x['xCoord'], x['yCoord']]))
            if not pd.isnull(x['previousXCoord']) else None, axis=1)

    # Add rebound information
    clean_df['rebound'] = clean_df.apply(lambda x:
       True if x['previousEvent'] == 'shot-on-goal' else False, axis=1
    )

    # Add a shot angle based on the ice coordinates
    # x['adjustedCoord']-np.array([0,89]) calcule les coordonnées du vecteur qui commence aux filets et s'arrête à l'emplacement du tirs
    # np.array([0, -89]) est le vecteur qui commence dans les filets et s'arrête au centre du stade/de la patinoire
    clean_df['shotAngle'] = clean_df.apply(
        lambda x: angle_between_vectors(x['adjustedCoord'] - np.array([0, 89]), np.array([0, -89])), axis=1)

    clean_df.drop(columns=['adjustedCoord'], inplace=True)  # Drop the adjusted coordinates

    # Add time before the last shot to observe the offensive pressure

    # Sort the dataframe by the period to calculate the time since the last shot
    clean_df['timeSinceLastShot'] = clean_df.groupby('eventOwnerTeam')['Game Seconds'].diff()

    # Convert the time to minutes and seconds
    clean_df['timeSinceLastShot'] = clean_df.apply(lambda x: 0
    if pd.isnull(x['timeSinceLastShot']) else x['timeSinceLastShot'], axis=1)

    # Add

    return clean_df
