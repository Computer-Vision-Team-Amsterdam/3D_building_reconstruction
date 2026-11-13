import os
import time
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import shapely.geometry as sg

from stage_1.src import api_request, array_math, extract_texture, visible_view


class FacadeExtractor:

    def __init__(
        self,
        target_area: sg.Polygon,
        panorama_year: int,
        texture_output_folder: str,
        camera_height: float = 2.0,
        texture_resolution: int = 30,
        min_building_area: float = 15.0,
        max_building_area: float = 400.0,
        min_building_distance: float = 1.7,
        max_building_distance: float = 9.0,
        min_facade_length: float = 1.0,
        building_search_radius: float = 10.0,
        building_simplifaction_tolerance: float = 0.1,
    ):
        self.target_area = target_area
        self.panorama_year = panorama_year
        self.texture_output_folder = texture_output_folder
        self.camera_height = camera_height
        self.texture_resolution = texture_resolution
        self.min_building_area = min_building_area
        self.max_building_area = max_building_area
        self.min_building_distance = min_building_distance
        self.max_building_distance = max_building_distance
        self.min_facade_length = min_facade_length
        self.building_search_radius = building_search_radius
        self.building_simplifaction_tolerance = building_simplifaction_tolerance
        self.state = 0

        if not os.path.exists(self.texture_output_folder):
            os.makedirs(self.texture_output_folder)
        else:
            print(
                f"Warning: 'texture_output_folder' {self.texture_output_folder} already exists."
            )

    def run(self) -> pd.DataFrame:
        print("--- Start ---")

        # Start timer
        start_time = time.time()

        print("Step 1: Gather metadata for buildings and panoramas in target area.")
        self.gather_data()

        print("Step 2: Identify visible building edges in range of a panorama images.")
        facade_df = self.get_visible_building_edges()

        print("Step 3: Drop duplicate facade textures.")
        # Drop duplicate facade rows, keep the ones with shortest distance.
        facade_df = (
            facade_df.sort_values("distance", ascending=True)
            .drop_duplicates(subset=["visible_point_one", "visible_point_two"])
            .sort_index()
            .reset_index(drop=True)
        )

        print("Step 4: Extract facade textures from panoramic images.")
        self.extract_facade_texture(facade_df)

        print(f"--- {(time.time() - start_time):.2f} seconds ---")

        return facade_df

    def gather_data(self) -> None:
        print("Gathering building metadata in target area...")
        self.buildings_gdf = api_request.get_buildings_in_area(
            target_area=self.target_area
        )
        print(f"{len(self.buildings_gdf)} buildings found")

        print("Gathering panorama metadata in target area...")
        self.panos_gdf = api_request.get_panos_in_area(
            target_area=self.target_area, mission_year=self.panorama_year
        )
        print(f"{len(self.buildings_gdf)} panoramas found")

        self.state = 1

    def _get_admissible_buildings(self, observer: sg.Point) -> gpd.GeoDataFrame:
        """
        Validate the GPS location of a panoramic image and the size of a building.
        """
        buildings_in_range = self.buildings_gdf[
            self.buildings_gdf.distance(observer) <= self.building_search_radius
        ].copy()

        admissible_buildings = buildings_in_range[
            (buildings_in_range.area >= self.min_building_area)
            & (buildings_in_range.area <= self.max_building_area)
        ]

        admissible_buildings = admissible_buildings[
            (
                admissible_buildings.exterior.distance(observer)
                >= self.min_building_distance
            )
            & ~admissible_buildings.intersects(observer)
        ]

        admissible_buildings["geometry"] = admissible_buildings.simplify(
            tolerance=self.building_simplifaction_tolerance, preserve_topology=True
        )

        return admissible_buildings

    def _get_facade_coordinates(
        self,
        buildings: gpd.GeoDataFrame,
        visibility_polygon: sg.Polygon,
        pano_id: str,
        observer: sg.Point,
        image_url: str,
        facade_data: List[Dict[str, Any]],
    ) -> None:
        """Get the two most distant visible points of a building."""
        for building in buildings.itertuples():
            # Get all the points from this polygon that are in the visibility polygon
            visible_segments = visible_view.segments_in_visibility_polygon(
                building.geometry, visibility_polygon, observer
            )

            # If there are fewer than two visible points for a building we skip (no facade)
            if len(visible_segments) == 0:
                continue

            texture_height_irl = float(building.building_height) - float(
                building.ground_elevation
            )

            for segment in visible_segments:
                new_data: Dict[str, Any] = dict()

                # Calculate the distance between two points
                vector_facade = array_math.vector(segment[0], segment[1])
                facade_length = array_math.vector_length(vector_facade)

                # Validation 1. The length of a "normal" facade is at least more than 2 meters
                if facade_length < self.min_facade_length:
                    # Go to next iteration of for loop, check for other buildings in same panorama
                    continue

                # Distance observer to middlepoint
                midpoint = array_math.get_midpoint(segment)
                vector_distance = array_math.vector(midpoint, (observer.x, observer.y))
                new_data["distance"] = round(
                    array_math.vector_length(vector_distance), 6
                )

                # Validation 2. Errors are found in the rectification process when the distance is above 9
                if new_data["distance"] > self.max_building_distance:
                    # Go to next iteration of for loop, check for other buildings in same panorama
                    continue

                # Fill dict after all the validation checks
                new_data["pand_id"] = str(building.bag_id)
                new_data["visible_point_one"] = tuple(segment[0])  # TODO why a tuple
                new_data["visible_point_two"] = tuple(segment[1])
                new_data["pano_id"] = pano_id
                new_data["pano_image_url"] = image_url
                new_data["observer"] = (observer.x, observer.y)
                new_data["texture_height_irl"] = texture_height_irl
                # The distance value makes the filename unique
                new_data["texture_filename"] = (
                    str(building.bag_id) + "_" + str(new_data["distance"])
                )

                facade_data.append(new_data)

    def get_visible_building_edges(self) -> Optional[pd.DataFrame]:
        """
        Estimation of buildings in range and identify visible building edges in 2D
        map.
        """
        if self.state < 1:
            print("ERROR: you must run .gather_data() before this step.")
            return None

        facade_data: List[Dict[str, Any]] = list()

        for pano in self.panos_gdf.itertuples():
            pano_id = pano.pano_id
            image_url = pano.image_url
            observer = pano.geometry

            admissible_buildings = self._get_admissible_buildings(observer=observer)

            if len(admissible_buildings) == 0:
                # print(
                #     "No admissible building polygons in range. Go to the next panorama"
                # )
                continue

            # Get the Visibility Polygon for observer
            visibility_polygon = visible_view.get_visibility_polygon(
                observer=observer, polygons=admissible_buildings.geometry.to_list()
            )

            # Get the visible facades
            self._get_facade_coordinates(
                buildings=admissible_buildings,
                visibility_polygon=visibility_polygon,
                pano_id=pano_id,
                observer=observer,
                image_url=image_url,
                facade_data=facade_data,
            )

        self.state = 2

        return pd.DataFrame(facade_data)

    def extract_facade_texture(self, df: pd.DataFrame):
        """
        Extract facade textures from panoramic images.
        """
        if self.state < 2:
            print("ERROR: you must run .get_visible_building_edges() before this step.")
            return None

        for group_name, df_group in df.groupby(by="pano_image_url"):
            # Get the panoramic image
            source_image = api_request.download_pano_image(group_name)
            if source_image is not None:
                # Iterate over the rows that are inside a group
                for _, row in df_group.iterrows():
                    # Create the facade texture
                    image_file = extract_texture.export_line_to_texture(
                        (row["visible_point_one"], row["visible_point_two"]),
                        self.texture_resolution,
                        row["texture_height_irl"],
                        self.camera_height,
                        source_image,
                        row["observer"],
                    )
                    # Save the facade texture
                    filename = row["texture_filename"]
                    image_file.save(
                        f"{self.texture_output_folder}{filename}.jpeg", "jpeg"
                    )
        self.state = 3
