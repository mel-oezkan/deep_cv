""" Code from: https://www.kaggle.com/code/sandhiwangiyana/sn6-splitting-image-tiles"""

from os.path import exists
import pathlib
from posixpath import split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy

# geospatial frameworks
import rasterio as rs
from rasterio.plot import show  # imshow for raster
import geopandas as gpd
from shapely.geometry import Polygon, box  # for geometry processing
from src2.data import ROOT_DIR, TEST_ROOT_DIR


def get_filepath(image_id, mode='PS-RGB'):
    return f'{ROOT_DIR}/{mode}/SN6_Train_AOI_11_Rotterdam_{mode}_{image_id}.tif'


def get_raster(image_id, mode='PS-RGB'):
    return rs.open(get_filepath(image_id, mode))


def create_geometry(filepath, image_ids):
    """Combines all geojson info of the spacenet 6 datasets 
    and writes them into one large file
    """

    geometry = []
    for image_id in image_ids:
        # read the raster
        ex_raster = rs.open(get_filepath(image_id, 'PS-RGB'))

        # grab its boundaries and convert to box coordinates
        geometry.append(box(*ex_raster.bounds))

    # create geodataframe
    d = {'image_id': image_ids, 'geometry': geometry}
    gdf = gpd.GeoDataFrame(d, crs='epsg:32631')

    # saving to geojson file
    gdf.to_file(filepath, driver='GeoJSON')
    print(f'{filepath} saved successfully!')

# get total bounds of gdf


def generate_AOI(split, gdf):
    lbox, bbox, rbox, tbox = gdf.total_bounds

    # horizontal stripes divides top-bot
    unit = (tbox-bbox)/split
    geometry = []
    for i in range(split):
        # i starts at 0, so u_bbox=bbox, then adds unit for each iter
        u_bbox = bbox+(unit*i)
        u_tbox = u_bbox+unit
        stripe = Polygon([
            (lbox, u_tbox),
            (rbox, u_tbox),
            (rbox, u_bbox),
            (lbox, u_bbox)])
        geometry.append(stripe)

    # create geodataframe
    df = gpd.GeoDataFrame({'geometry': geometry}, crs='epsg:32631')
    return df


def filter_tile(aoi_df, gdf):
    # overlay
    aoi_overlay = gpd.overlay(gdf, aoi_df, how='intersection')

    # count percentage remaining area
    aoi_overlay['Area'] = aoi_overlay.area
    max_area = aoi_overlay.Area.max()
    aoi_overlay['Per_Area'] = aoi_overlay['Area'].apply(
        lambda x: x/max_area*100)

    # grab tiles that are more than half in the AOI
    aoi_overlay_filt = aoi_overlay[aoi_overlay.Per_Area > 50.1]
    return aoi_overlay_filt.image_id.values


def split_tiles(geojson_name='tile_positions.geojson', splits=10) -> list:

    if not exists(f'{ROOT_DIR}/SummaryData/{geojson_name}'):
        print('\t\t\t Creating new global geojson csv')
        # grab unique image_id from the annotation csv
        building_csv_path = TEST_ROOT_DIR.joinpath(
            'SummaryData', 'SN6_Train_AOI_11_Rotterdam_Buildings.csv')

        df = pd.read_csv(
            str(building_csv_path))

        image_ids = df.ImageId.unique()
        create_geometry(f'{ROOT_DIR}/SummaryData/{geojson_name}', image_ids)

    # load geodataframe containing positional information for every tile
    gdf = gpd.read_file(f'{ROOT_DIR}/SummaryData/{geojson_name}')
    AOI_stripes_gdf = generate_AOI(splits, gdf)
    filtered_tiles = []

    # iterate through all rows
    for _, rows in AOI_stripes_gdf.iterrows():
        aoi_df = gpd.GeoDataFrame({'geometry': rows}, crs='epsg:32631')
        filtered_tiles.append(filter_tile(aoi_df, gdf))

    return filtered_tiles


def recombine_splits(even_splits: list, out_splits: dict) -> dict:
    """

    :param even_splits: list of tile group inidces
    :param out_split: distribution of each split


    :returns result: where keys contains the split type and 
        values are the indices used for that split
    :rtype: dict
    """

    print(len(even_splits))
    print([len(x) for x in even_splits])

    global min_conf, d_min
    min_conf = {}

    d_min = np.inf
    split_sizes = [len(split) for split in even_splits]
    total_sum = sum(split_sizes)

    def recursive_loop(split_dict: dict, arr: list, idx=0, pos_key=0, splits={}) -> None:
        """

        :param split_dict: 
        :param arr: combination of different tile groups

        """
        global d_min, min_conf

        if pos_key == len(split_dict.keys()) - 1:
            splits[list(split_dict.keys())[pos_key]] = arr[idx+1:]
            d = 0
            for key in out_splits.keys():
                d += abs(total_sum * out_splits[key] - sum(splits[key]))
            if d < d_min:
                min_conf = copy.deepcopy(splits)
                d_min = d
        else:
            for i in range(idx+1, len(arr) - (len(split_dict.keys()) - pos_key - 1)):
                arr_subset = arr[idx: i]
                splits[list(split_dict.keys())[pos_key]] = arr_subset
                recursive_loop(split_dict, arr, i, pos_key+1, splits)

    # iterates over all possible permuations
    for l, perm in enumerate(itertools.permutations(split_sizes, len(split_sizes))):
        recursive_loop(out_splits, perm)

    result = {}
    for key in min_conf.keys():
        result[key] = []
        for length in min_conf[key]:
            result[key].append(even_splits[split_sizes.index(length)])
        result[key] = np.concatenate(result[key], axis=0)

    return result


def simple_dist(cuts):
    print(itertools.chain(*cuts[:8]))
    tile_dist = {
        'train': list(itertools.chain(*cuts[:8])),
        'val': cuts[8],
        'test': cuts[9]
    }

    return tile_dist
