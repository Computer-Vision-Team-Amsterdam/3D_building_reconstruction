"""
Previous work by the City of Amsterdam:
https://github.com/Amsterdam/panorama-textures/
"""

from typing import Any, Dict, Optional, Tuple

from numpy import float64, full, int32, linspace, meshgrid, pi
from numpy.typing import NDArray
from PIL import Image, ImageFile

from stage_1.src.array_image import get_as_rgb_array, sample_rgb_array_image_as_array
from stage_1.src.array_math import (
    cartesian2cylindrical,
    get_vector,
    get_vector_3d,
    get_vector_length,
    get_viewing_angle,
)

# helper constants for readability
X, Y = 0, 1


# Funtion that projects each facade grid point onto the panorama and samples the colours. Facades that are not forward facing, or are viewed at too "tilted" angle, become a grey placeholder, unless `force=True`
def project_facade(
    facade_plane: Dict[str, Any],
    observer: Tuple[float, float, float],
    source_file: ImageFile.ImageFile,
    texture_height: int,
    texture_resolution: int,
    force: bool = True,
) -> Optional[NDArray]:
    nx = int(round(facade_plane["length"] * texture_resolution))
    if nx > 0:
        # Gets the distances to facades with observer as middlepoint
        if not ((0.7 * pi > facade_plane["viewing_angle"] > 0.3 * pi) or force):
            facade_texture = full((texture_height, nx, 3), 128, dtype=int32)
        else:
            # 3D tuple of the vector
            vector_x, vector_y, vector_z = get_vector_3d(
                (
                    facade_plane["x-mesh"],
                    facade_plane["y-mesh"],
                    facade_plane["z-mesh"],
                ),
                observer,
            )
            # 3D tuple of the coordinates of the vector on the panoramic image (equirectangular plane)
            image_x, image_y = cartesian2cylindrical(
                (vector_y, vector_x, vector_z),
                source_width=source_file.width,
                source_height=source_file.height,
                r_is_1=False,
            )
            # Load the source panoramic image, return numpy image array (an array of three color channels)
            source_rgb_array = get_as_rgb_array(source_file)
            # The sampled target image as a scipy rgb array representation
            facade_texture = sample_rgb_array_image_as_array(
                (image_x, image_y), source_rgb_array
            )
    else:
        facade_texture = None
    return facade_texture


# builds a 3D grid of points on the facade: `x`/`y` run along the bottom edge between the two visible points, `z` runs from the image height down to 0. Also computes the viewing angle, and a facade counts as forward facing when that angle is positive
def create_facade_plane(
    facade_height: float,
    texture_height: int,
    texture_resolution: int,
    observer: Tuple[float, float, float],
    facade: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Dict[str, Any]:
    # Create dict elements for later
    facade_plane = {
        "from": facade[0],
        "to": facade[1],
        "vector": get_vector(facade[0], facade[1]),
    }
    # Get the width of the facade
    facade_plane["length"] = get_vector_length(facade_plane["vector"])

    # Calculate viewing angle from observer to facade
    facade_plane["viewing_angle"] = get_viewing_angle(
        facade, (observer[0], observer[1])
    )

    # Forward_facing is true when we have a viewing angle
    facade_plane["forward_facing"] = facade_plane["viewing_angle"] > 0

    # numpy: Create sequences of evenly spaced values within a defined interval
    nx = int(round(facade_plane["length"] * texture_resolution))
    x = linspace(facade_plane["from"][X], facade_plane["to"][X], nx, dtype=float64)
    y = linspace(facade_plane["from"][Y], facade_plane["to"][Y], nx, dtype=float64)
    z = linspace(facade_height, 0, texture_height, dtype=float64)

    # numpy: Return coordinate matrices from coordinate vectors
    gevel_x, _ = meshgrid(x, z)
    gevel_y, gevel_z = meshgrid(y, z)
    facade_plane["x-mesh"] = gevel_x
    facade_plane["y-mesh"] = gevel_y
    facade_plane["z-mesh"] = gevel_z

    return facade_plane


# The entry point used by `run.py`. takes the two visible facade points, the resolution in pixels per metre, the image height in metres, the camera height, the source panorama and the observer's RD position, and returns a PIL image.
def export_facade_to_texture(
    facade: Tuple[Tuple[float, float], Tuple[float, float]],
    texture_resolution: int,
    facade_height: float,
    camera_height: float,
    source_image: ImageFile.ImageFile,
    pano_location: Tuple[float, float],
) -> Image.Image:
    observer = (pano_location[0], pano_location[1], camera_height)

    texture_height = int(round(facade_height * texture_resolution))
    facade_plane = create_facade_plane(
        facade_height, texture_height, texture_resolution, observer, facade
    )
    facade_texture = project_facade(
        facade_plane,
        observer,
        source_image,
        texture_height,
        texture_resolution,
        force=True,
    )

    return Image.fromarray(facade_texture.astype("uint8"), "RGB")
