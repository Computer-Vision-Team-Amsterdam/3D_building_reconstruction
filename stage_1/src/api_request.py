"""
Amsterdam API description: https://api.data.amsterdam.nl/api/
3DBAG API description: https://api.3dbag.nl/api.html
"""

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import requests
import shapely
from PIL import Image, ImageFile
from pyproj import Transformer

API_TIMEOUT = 10
RETRY_ATTEMPTS = 5

WGS84_EPSG = 4326
RD_EPSG = 28992

rd_to_wgs84 = Transformer.from_crs(
    crs_from=RD_EPSG, crs_to=WGS84_EPSG, always_xy=True
).transform


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


def get_buildings_in_area(target_area: shapely.Geometry) -> gpd.GeoDataFrame:
    """
    Query the BAG api for all buildings that intersect with the given target shape. Note: target shape is expected to be in Amersfoort / RD New format.
    """
    bag_gdf = get_bag_buildings_in_area(target_area=target_area)
    bag3d_df = get_3dbag_buildings_in_area(target_area=target_area)
    return bag_gdf.join(
        other=bag3d_df.set_index("bag_id"),
        on="bag_id",
        how="left",
    )


def get_bag_buildings_in_area(target_area: shapely.Geometry) -> gpd.GeoDataFrame:
    date_str = datetime.today().strftime("%Y-%m-%d")  # TODO base this on MISSION_YEAR?
    target_area_wgs84 = shapely.transform(target_area, rd_to_wgs84, interleaved=False)
    url = f"https://api.data.amsterdam.nl/v1/bag/v1/panden?geometrie%5Bintersects%5D={target_area_wgs84.wkt}&geldigOp={date_str}"

    dfs = []

    json_data = get_request(url=url, load_json=True)

    dfs.append(_parse_bag_response(json_data=json_data))
    next_page_url = _get_next_page_ams(links=json_data["_links"])

    while next_page_url is not None:
        json_data = get_request(url=next_page_url, load_json=True)
        dfs.append(_parse_bag_response(json_data=json_data))
        next_page_url = _get_next_page_ams(links=json_data["_links"])

    return pd.concat(dfs, ignore_index=True)


def get_3dbag_buildings_in_area(target_area: shapely.Geometry) -> pd.DataFrame:
    minx, miny, maxx, maxy = target_area.bounds
    url = f"https://api.3dbag.nl/collections/pand/items?bbox={minx},{miny},{maxx},{maxy}&limit=100"

    dfs = []

    json_data = get_request(url=url, load_json=True)

    dfs.append(_parse_3dbag_response(json_data=json_data))
    next_page_url = _get_next_page_3dbag(links=json_data["links"])

    while next_page_url is not None:
        json_data = get_request(url=next_page_url, load_json=True)
        dfs.append(_parse_3dbag_response(json_data=json_data))
        next_page_url = _get_next_page_3dbag(links=json_data["links"])

    return pd.concat(dfs, ignore_index=True)


def get_panos_in_area(
    target_area: shapely.Geometry, mission_year: int, image_quality: str = "medium"
) -> gpd.GeoDataFrame:
    """
    Get panoramic image id's for a user defined bounding box region
    """
    minx, miny, maxx, maxy = target_area.bounds
    url = f"https://api.data.amsterdam.nl/panorama/panoramas/?tags=mission-{mission_year}&bbox={minx},{miny},{maxx},{maxy}&srid=28992"

    dfs = []

    json_data = get_request(url=url, load_json=True)

    dfs.append(
        _parse_panorama_response(json_data=json_data, image_quality=image_quality)
    )
    next_page_url = _get_next_page_ams(links=json_data["_links"])

    while next_page_url is not None:
        json_data = get_request(url=next_page_url, load_json=True)
        dfs.append(
            _parse_panorama_response(json_data=json_data, image_quality=image_quality)
        )
        next_page_url = _get_next_page_ams(links=json_data["_links"])

    pano_gdf = pd.concat(dfs, ignore_index=True)

    return pano_gdf[pano_gdf.intersects(target_area)]


def download_pano_image(pano_url: str) -> ImageFile:
    """Download a panoramic image from the API"""
    response_content = get_request(pano_url, load_json=False)
    return Image.open(BytesIO(response_content))


def _parse_bag_response(json_data: Dict) -> gpd.GeoDataFrame:
    data: Dict[str, List] = {
        "bag_id": [],
        "geometry": [],
    }

    for pand in json_data["_embedded"]["panden"]:
        data["bag_id"].append(pand["identificatie"])
        data["geometry"].append(shapely.from_geojson(json.dumps(pand["geometrie"])))

    return gpd.GeoDataFrame(data=data, crs=RD_EPSG)


def _parse_3dbag_response(json_data: Dict) -> pd.DataFrame:
    data: Dict[str, List] = {
        "bag_id": [],
        "ground_elevation": [],
        "building_height": [],
    }
    for pand in json_data["features"]:
        id = pand["id"]
        data["bag_id"].append(id.strip("NL.IMBAG.Pand."))
        data["ground_elevation"].append(
            pand["CityObjects"][id]["attributes"]["b3_h_maaiveld"]
        )
        data["building_height"].append(
            pand["CityObjects"][id]["attributes"]["b3_h_dak_max"]
        )
    return pd.DataFrame(data=data)


def _parse_panorama_response(
    json_data: Dict, image_quality: str = "medium"
) -> gpd.GeoDataFrame:
    data: Dict[str, List] = {
        "pano_id": [],
        "image_url": [],
        "timestamp": [],
        "geometry": [],
    }
    for pano in json_data["_embedded"]["panoramas"]:
        data["pano_id"].append(pano["pano_id"])
        data["image_url"].append(
            pano["_links"][f"equirectangular_{image_quality}"]["href"]
        )
        data["timestamp"].append(pd.to_datetime(pano["timestamp"]))
        data["geometry"].append(shapely.from_geojson(json.dumps(pano["geometry"])))

    return gpd.GeoDataFrame(data=data, crs=WGS84_EPSG).to_crs(RD_EPSG)


def _get_next_page_3dbag(links: List[Dict[str, str]]) -> Optional[str]:
    for link in links:
        if link["rel"] == "next":
            return link["href"]
    return None


def _get_next_page_ams(links: Dict[str, Dict[str, Optional[str]]]) -> Optional[str]:
    if "next" in links.keys():
        return links["next"]["href"]
    else:
        return None
