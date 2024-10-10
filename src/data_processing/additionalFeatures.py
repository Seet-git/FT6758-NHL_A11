import numpy as np
import pandas as pd


def angle_between_vectors(v1: np.array, v2: np.array) -> float:
    """
    Calculate the angle in degrees between two 2D vectors.
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
    """Add additional features to the dataframe"""

    # Add shot distance

    # Define the Euclidian distance function
    dist_euclidian = lambda x1, x2: np.round(np.linalg.norm(np.array(x1 - x2)), decimals=1)

    # Add shot distance based on the ice coordinates
    clean_df['shotDistance'] = clean_df.apply(
        lambda x: dist_euclidian(x['iceCoord'], np.array([-89, 0])) if x['iceCoord'][0] <= 0
        else dist_euclidian(x['iceCoord'], np.array([89, 0])), axis=1)

    # Add shot angle based on the ice coordinates
    clean_df['shotAngle'] = clean_df.apply(
        lambda x: angle_between_vectors(x['iceCoord'], np.array([-89, 0])) if x['iceCoord'][0] <= 0
        else angle_between_vectors(x['iceCoord'], np.array([89, 0])), axis=1)

    return clean_df
