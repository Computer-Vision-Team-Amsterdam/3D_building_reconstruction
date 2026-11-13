"""
The first stage of the pipeline determines — within a captured panoramic image
sequence — the texture region for each building (pand_id) to be reconstructed.

There are three input sources used in this stage: BAG building footprint data,
panoramic images, and GNSS/INS sensor values. The initial GNSS/INS sensor
specifies the camera’s geospatial location and orientation information that
correspond to each of the panoramic images.

NOTE: Change the variables BBOX_REGION and MISSION_YEAR to your own settings!
"""

import shapely.geometry as sg

# External Python files
from stage_1.facade_extraction import FacadeExtractor

# API options
BBOX_REGION = [122300.00, 486475.48, 122400.00, 486628.78]
MISSION_YEAR = 2025  # API tag
SEARCH_RADIUS = 10  # Building search radius from an initial panoramic image location

# Texture options
TEXTURE_RESOLUTION = 30
CAMERA_HEIGHT = 2  # Height of the camera mounted on the car

# Validation options before rectification (strict settings)
MIN_DISTANCE = 1.7  # Min distance camera to building
MAX_DISTANCE = 9
MIN_BUILDING_AREA = 15
MAX_BUILDING_AREA = 400
MIN_FACADE_LENGTH = 1.0

IMAGES_OUTPUT = "texture_output/"
OUTPUT_NAME = "output_stage1"


def save_results(df):
    """Save results of extracted facade textures"""
    df_output = df[
        ["pand_id", "visible_point_one", "visible_point_two", "texture_filename"]
    ]
    compression_opts = dict(method="zip", archive_name=OUTPUT_NAME + ".csv")
    df_output.to_csv(OUTPUT_NAME + ".zip", index=False, compression=compression_opts)


def main():
    """
    Identify, rectify and extract the texture region of buildings from a
    panoramic image sequence.
    """

    target_area = sg.box(*BBOX_REGION)

    facade_extractor = FacadeExtractor(
        target_area=target_area,
        panorama_year=MISSION_YEAR,
        texture_output_folder=IMAGES_OUTPUT,
        camera_height=CAMERA_HEIGHT,
        texture_resolution=TEXTURE_RESOLUTION,
        min_building_area=MIN_BUILDING_AREA,
        max_building_area=MAX_BUILDING_AREA,
        min_building_distance=MIN_DISTANCE,
        max_building_distance=MAX_DISTANCE,
        min_facade_length=MIN_FACADE_LENGTH,
        building_search_radius=SEARCH_RADIUS,
    )

    facade_df = facade_extractor.run()

    save_results(facade_df)


if __name__ == "__main__":
    main()
