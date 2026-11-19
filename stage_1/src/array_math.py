# small vector helpers used throughout the pipeline: 2D vectors on the map, and the conversion from a 3D direction vector to a pixel position on equirectangular panorama

# NumPy 2.x: import from numpy directly (numpy.core.umath is deprecated)
from typing import Tuple

import numpy as np

# Helper constants for readability
X, Y = 0, 1


# 2D helpers
def get_vector(from_2d, to_2d):
    return to_2d[X] - from_2d[X], to_2d[Y] - from_2d[Y]


def get_vector_length(vector_2d):
    return np.sqrt(vector_2d[X] ** 2 + vector_2d[Y] ** 2)


def get_midpoint(vertices):
    return (
        (vertices[0][X] + vertices[1][X]) / 2,
        (vertices[0][Y] + vertices[1][Y]) / 2,
    )


def get_viewing_angle(
    segment: Tuple[Tuple[float, float], Tuple[float, float]],
    observer: Tuple[float, float],
) -> float:
    # Get segment vector
    segment_vector = get_vector(segment[0], segment[1])

    # Get midpoint of the facade and the distance from this point to the observer
    midpoint = get_midpoint(segment)
    to_midpoint = get_vector(observer, midpoint)

    # Calculate viewing angle from observer to facade
    if get_vector_length(segment_vector) == 0 or get_vector_length(to_midpoint) == 0:
        return 0
    else:
        dot = (
            segment_vector[X] * to_midpoint[X] + segment_vector[Y] * to_midpoint[Y]
        )  # dot product
        det = (
            segment_vector[X] * to_midpoint[Y] - segment_vector[Y] * to_midpoint[X]
        )  # determinant
        return np.arctan2(det, dot)


# 3D helpers
def get_vector_3d(to_point, from_point):
    """
    Calculate a 3D vector from from_point to to_point,
    also works on numpy 2D arrays of points

    :param to_point: 3d tuple of the point the vector will point to
    :param from_point: 3d tuple of the point the vector will point from
    :return: 3D tuple of the vector
    """

    return (
        to_point[0] - from_point[0],
        to_point[1] - from_point[1],
        to_point[2] - from_point[2],
    )


def cartesian2cylindrical(vector, source_width, source_height, r_is_1=True):
    """
    Calculates the location on a equirectangular plane of a vector,
    Will work with numpy's 2D arrays of points

    :param vector: 3d tuple of the point the vector
    :param source_width: width and
    :param source_height: height of the equirectangular image
    :param r_is_1: boolean, denoting if vectors are normalized to 1 (default=True)
    :return: 3D tuple of the coordinates of the vector on the equirectangular plane
    """

    middle = source_width / 2

    x = vector[0]
    y = vector[1]
    z = vector[2]

    # Vectors are defined by (r, theta, phi) and is given in Cartesian coordinates by:
    r = 1 if r_is_1 else np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    x1 = np.mod(middle + middle * phi / np.pi, source_width - 1)
    y1 = source_height * theta / np.pi

    return x1, y1
