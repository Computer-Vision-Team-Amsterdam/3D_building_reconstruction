"""
Amsterdam API description: https://api.data.amsterdam.nl/api/
"""

import json
from io import BytesIO
from typing import Any, Tuple

import requests
from PIL import Image
from pyproj import Transformer

API_TIMEOUT = 10
RETRY_ATTEMPTS = 5

# WGS84 (EPSG:4326, lon/lat) to Rijksdriehoek (EPSG:28992, metres); always_xy keeps the order (lon, lat) / (x, y)
wgs84_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def get_request(url: str, load_json: bool = True) -> Any:
    attempt = 1
    while attempt <= RETRY_ATTEMPTS:
        try:
            with requests.get(url, timeout=API_TIMEOUT) as response:
                if load_json:
                    return json.loads(response.content)
                else:
                    return response.content
        except requests.exceptions.ReadTimeout as e:
            if attempt < RETRY_ATTEMPTS:
                attempt += 1
                print(
                    f"{e} for {url}, trying again (attempt {attempt}/{RETRY_ATTEMPTS}).."
                )
                continue
            else:
                raise e


def get_pano_id(mission_year, bbox):
    """
    Get panoramic image id's for a user defined bounding box region
    """
    pano_url = f"https://api.data.amsterdam.nl/panorama/panoramas/?tags=mission-{mission_year}&bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&srid=28992"

    pano_data_all = get_request(pano_url)

    pano_data = pano_data_all["_embedded"]["panoramas"]

    pano_id = []
    for item in pano_data:
        pano_id.append(item["pano_id"])

    # Check for next page with data
    next_page = pano_data_all["_links"]["next"]["href"]

    # Exit the while loop if there is no next page
    while next_page:
        pano_data_all = get_request(next_page)

        pano_data = pano_data_all["_embedded"]["panoramas"]

        # Append the panorama id's to the list
        for item in pano_data:
            pano_id.append(item["pano_id"])

        # Check for next page
        next_page = pano_data_all["_links"]["next"]["href"]

    return pano_id


def get_pano_image(pano_id):
    """
    Get a panoramic image and its initial location coordinates and convert it
    from WGS84 (EPSG:4326) to Rijksdriehoek (EPSG:28992)
    """
    pano_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pano_id}/"
    pano_data = get_request(pano_url)

    geom = pano_data["geometry"]["coordinates"]

    # WGS84 to Rijksdriehoek (RD) conversion
    lon, lat = geom[0], geom[1]
    x_rd, y_rd = wgs84_to_rd.transform(lon, lat)

    # Get image_url from API
    image_url = pano_data["_links"]["equirectangular_medium"]["href"]

    return (
        str(image_url),
        (x_rd, y_rd),
        (lon, lat),
    )


def get_buildings_in_range(observer, radius):
    """
    For a panoramic image, get the buildings that are within a user defined
    range.
    """
    bag_url = f"https://api.data.amsterdam.nl/v1/bag/v1/panden?geometrie[within]=POINT({observer[0]} {observer[1]}),{radius}"
    return get_request(bag_url)["_embedded"]["panden"]


def get_building_polygons(pand_json):
    """Get the polygon of a building"""
    building_polygons = {}

    # Loop over json content
    for item in pand_json:
        pand_id = item["identificatie"]
        pand_polygon = item["geometrie"]["coordinates"][0]

        # A polygon should have at least 3 vertices to be a valid polygon
        if len(pand_polygon) > 2 and pand_id:
            # Verify counter-clockwise winding order
            # TODO (Removed this because it happens rarely and slows the process down)

            building_polygons[pand_id] = pand_polygon

    return building_polygons


def get_height_for_id(bag_id: str) -> Tuple[float, float]:
    "Query the 3DBAG api for the given ID and return ground elevation and building height in meters."

    full_bag_id_str = f"NL.IMBAG.Pand.{bag_id}"
    url = f"https://api.3dbag.nl/collections/pand/items/{full_bag_id_str}"

    json_data = get_request(url=url, load_json=True)

    ground_elevation = json_data["feature"]["CityObjects"][full_bag_id_str][
        "attributes"
    ]["b3_h_maaiveld"]
    building_height = json_data["feature"]["CityObjects"][full_bag_id_str][
        "attributes"
    ]["b3_h_dak_max"]
    return (ground_elevation, building_height)


def download_pano_image(pano_url):
    """Download a panoramic image from the API"""
    response_content = get_request(pano_url, load_json=False)
    return Image.open(BytesIO(response_content))
