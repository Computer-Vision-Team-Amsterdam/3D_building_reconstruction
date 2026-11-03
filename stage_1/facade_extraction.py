import pandas as pd
import shapely.geometry

from stage_1.src import (
    api_request,
    array_math,
    extract_texture,
    furthest_pair,
    visible_view,
)


class FacadeExtractor:

    def __init__(
        self,
        texture_output_folder: str,
        camera_height: float = 2.0,
        image_height_irl: float = 30.0,
        texture_resolution: int = 30,
        min_building_area: float = 15.0,
        max_building_area: float = 400.0,
        min_building_distance: float = 1.7,
        max_building_distance: float = 9.0,
        building_search_radius: float = 10.0,
    ):
        self.texture_output_folder = texture_output_folder
        self.camera_height = camera_height
        self.image_height_irl = image_height_irl
        self.texture_resolution = texture_resolution
        self.min_building_area = min_building_area
        self.max_building_area = max_building_area
        self.min_building_distance = min_building_distance
        self.max_building_distance = max_building_distance
        self.building_search_radius = building_search_radius

    def get_admissible_data(self, observer, building_polygons):
        """
        Validate the GPS location of a panoramic image and the size of a building.
        """
        observer_point = shapely.geometry.Point(observer)

        admissible_building_polygons = {}
        for k, v in list(building_polygons.items()):
            building_poly = shapely.geometry.Polygon(v)

            # Check if point q is too close to a building
            if (
                building_poly.exterior.distance(observer_point)
                < self.min_building_distance
            ):
                print("GNSS/INS measurement error found! Too close to building.")
                return {}

            # Check if point q is inside a building (GNSS/INS error)
            if observer_point.intersects(building_poly):
                print("GNSS/INS measurement error found! Point inside building.")
                return {}

            # Check for building area sizes that we ignore (Too large: Bijenkorf, too small: mini snackbar)
            if self.min_building_area < building_poly.area < self.max_building_area:
                admissible_building_polygons[k] = v

        return admissible_building_polygons

    def get_facade_coordinates(
        self, building_polygons, visibility_polygon, observer, image_url, facade_data
    ):
        """Get the two most distant visible points of a building."""
        for pand_id in building_polygons.keys():
            new_data = {}

            # Get all the points from this polygon that are in the visibility polygon
            visible_points = visible_view.point_in_visibility_polygon(
                building_polygons[pand_id], visibility_polygon
            )

            # Check if there are two or more visible points for a building
            if len(visible_points) >= 2:
                # Get from a list of items the furthest pair where we keep the (winding) order of the list
                visible_points_left_right = furthest_pair.get_furthest_pair(
                    visible_points, observer
                )

                # Two validation checks

                # Calculate the distance between two points
                vector_facade = array_math.vector(
                    visible_points_left_right[0], visible_points_left_right[1]
                )
                facade_length = array_math.vector_length(vector_facade)

                # Validation 1. The length of a "normal" facade is at least more than 2 meters
                if facade_length < 2:
                    # Go to next iteration of for loop, check for other buildings in same panorama
                    continue

                # Distance observer to middlepoint
                midpoint = array_math.get_midpoint(visible_points_left_right)
                vector_distance = array_math.vector(midpoint, observer)
                new_data["distance"] = round(
                    array_math.vector_length(vector_distance), 6
                )

                # Validation 2. Errors are found in the rectification process when the distance is above 9
                if new_data["distance"] > self.max_building_distance:
                    # Go to next iteration of for loop, check for other buildings in same panorama
                    continue

                # Fill dict after all the validation checks
                new_data["pand_id"] = str(pand_id)
                new_data["visible_point_one"] = tuple(
                    visible_points_left_right[0]
                )  # TODO why a tuple
                new_data["visible_point_two"] = tuple(visible_points_left_right[1])
                new_data["pano_image_url"] = image_url
                new_data["observer"] = observer
                # The distance value makes the filename unique
                new_data["texture_filename"] = (
                    str(pand_id) + "_" + str(new_data["distance"])
                )

                facade_data.append(new_data)

    def get_visible_building_edges(self, pano_ids, facade_data):
        """
        Estimation of buildings in range and identify visible building edges in 2D
        map.
        """
        for pano_id in pano_ids:
            # Get the panoramic image url and the location of the "observer"
            image_url, observer, observer_wgs = api_request.get_pano_image(pano_id)

            # Get BAG "pand" information. Search radius in meters from point "near"
            pand_json = api_request.get_buildings_in_range(
                observer_wgs, self.building_search_radius
            )

            # Get a dict of the buildings (footprint data) in range
            building_polygons = api_request.get_building_polygons(pand_json)

            # Validation checks on the GPS location and building polygons
            building_polygons = self.get_admissible_data(observer, building_polygons)

            if not building_polygons:
                # print("No admissible building polygons in range. Go to the next panorama")
                continue

            # Generate the outlines of the arrangement
            arr = visible_view.generate_arrangement(
                observer, building_polygons.values()
            )

            # Get the Visibility Polygon for point q
            visibility_polygon = visible_view.get_visibility_polygon(arr, observer)

            # Get two most distant visible points of a builing
            self.get_facade_coordinates(
                building_polygons, visibility_polygon, observer, image_url, facade_data
            )

    def extract_facade_texture(self, df: pd.DataFrame):
        """
        Extract facade textures from panoramic images.
        """
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
                        self.image_height_irl,
                        self.camera_height,
                        source_image,
                        row["observer"],
                    )
                    # Save the facade texture
                    filename = row["texture_filename"]
                    image_file.save(
                        f"{self.texture_output_folder}{filename}.jpeg", "jpeg"
                    )
