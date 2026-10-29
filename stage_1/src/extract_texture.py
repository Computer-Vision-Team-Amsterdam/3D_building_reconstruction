"""
Previous work by the City of Amsterdam:
https://github.com/Amsterdam/panorama-textures/
"""

from math import atan2, pi
from PIL import Image
from numpy import linspace, float64, meshgrid, full, int32
from src.array_math import get_vector, cartesian2cylindrical, vector, vector_length, get_midpoint
from src.array_image import get_as_rgb_array, sample_rgb_array_image_as_array

# helper constants for readability
X, Y = 0, 1

# We use the "panorama_4000" images, with the following dimensions
SOURCE_WIDTH = 4000 # pixels
PANO_HEIGHT = 2000 # pixels


# Funtion that projects each facade grid point onto the panorama and samples the colours. Facades that are not forward facing, or are viewed at too "tilted" angle, become a grey placeholder, unless `force=True`
def project_facade(facade, observer, source_file, img_height, resolution, force=False):
    nx = int(round(facade["length"] * resolution))
    if nx > 0:
        # Gets the distances to facades with observer as middlepoint
        if not facade["forward_facing"] or (not 0.7 * pi > facade["viewing_angle"] > 0.3 * pi and not force):
            imarray = full((img_height, nx, 3), 128, dtype=int32)
        else:
            # 3D tuple of the vector
            vector_x, vector_y, vector_z = get_vector((facade["x-mesh"], facade["y-mesh"], facade["z-mesh"]), observer)
            # 3D tuple of the coordinates of the vector on the panoramic image (equirectangular plane)
            image_x, image_y = cartesian2cylindrical((vector_y, vector_x, vector_z), source_width=SOURCE_WIDTH,
                                                     source_height=PANO_HEIGHT, r_is_1=False)
            # Load the source panoramic image, return numpy image array (an array of three color channels)
            source_rgb_array = get_as_rgb_array(source_file)
            # The sampled target image as a scipy rgb array representation
            imarray = sample_rgb_array_image_as_array((image_x, image_y), source_rgb_array)
    else:
        imarray = None
    return imarray


# builds a 3D grid of points on the facade: `x`/`y` run along the bottom edge between the two visible points, `z` runs from the image height down to 0. Also computes the viewing angle, and a facade counts as forward facing when that angle is positive
def create_plane(hoogte, img_height, resolution, observer, vertices):
    # Create dict elements for later
    line = {
        "from": vertices[0],
        "to": vertices[1],
        "vector": vector(vertices[0], vertices[1])
    }
    # Get the width of the facade
    line["length"] = vector_length(line["vector"])

    # Get midpoint of the facade and the distance from this point to the observer
    midpoint = get_midpoint(vertices)
    to_midpoint = vector(observer, midpoint)

    # Calculate viewing angle from observer to facade
    if line["length"] == 0 or vector_length(to_midpoint) == 0:
        line["viewing_angle"] = 0
    else:
        dot = line["vector"][X]*to_midpoint[X] + line["vector"][Y]*to_midpoint[Y]      # dot product
        det = line["vector"][X]*to_midpoint[Y] - line["vector"][Y]*to_midpoint[X]      # determinant
        line["viewing_angle"] = atan2(det, dot)

    # Forward_facing is true when we have a viewing angle
    line["forward_facing"] = line["viewing_angle"] > 0

    # numpy: Create sequences of evenly spaced values within a defined interval
    nx = int(round(line["length"] * resolution))
    x = linspace(line["from"][X], line["to"][X], nx, dtype=float64)
    y = linspace(line["from"][Y], line["to"][Y], nx, dtype=float64)
    z = linspace(hoogte, 0, img_height, dtype=float64)

    # numpy: Return coordinate matrices from coordinate vectors
    gevel_x, _ = meshgrid(x, z)
    gevel_y, gevel_z = meshgrid(y, z)
    line["x-mesh"] = gevel_x
    line["y-mesh"] = gevel_y
    line["z-mesh"] = gevel_z

    return line


# The entry point used by `run.py`. takes the two visible facade points, the resolution in pixels per metre, the image height in metres, the camera height, the source panorama and the observer's RD position, and returns a PIL image.
def export_line_to_texture(visible_points, resolution, image_height_irl, camera_height, source_image, pano_rd):
    observer = (pano_rd[0], pano_rd[1], camera_height)

    img_height = int(round(image_height_irl * resolution))
    plane = create_plane(image_height_irl, img_height, resolution, observer, visible_points)
    facade = project_facade(plane, observer, source_image, img_height, resolution, force=True)

    return Image.fromarray(facade.astype("uint8"), "RGB")
