from tokenize import String
from venv import create
import pandas as pd
from shapely.geometry import Polygon
import rasterio.features
import numpy as np
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt


def load_summary(path='./datasets/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv') -> pd.DataFrame:
    """Load the summary csv containing polygon information, image_id.
    
    :param path: path to the summary csv documents,
        defaults to [...]/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv
    :type path: string, optional
    :return: dataframe containing summary data
    :rtype: pd.DataFrame
    """
    summary_data = pd.read_csv(path, delimiter=',')
    return summary_data


def wkt_string_to_polygon(polygon_string: str) -> Polygon:
    """Convert WKT-String to shapely.geometry.Polygon
    
    :param polygon_string: string describing a polgons shape in wkt format
        taken from the summary-csv column 'PolygonWKT_Pix'
    :type polygon_string: string
    :return: Polygon object
    :rtype: shapely.geometry.Polygon
    """
    # cut off beginning and end
    shortened = polygon_string.split('((')[1]
    shortened = shortened[:-2]
    # split into polygon descriptors
    s1 = [e.replace('(', '').split(',') for e in shortened.split(')') if len(e)]
    float_tuples = [tuple(tuple(map(float, e.split())) for e in s2 if len(e)) for s2 in s1]
    # get shell and hole descriptor of final shape
    shell = float_tuples[0]
    holes = None if len(float_tuples)==1 else float_tuples[1:]
    polygon = Polygon(shell=shell, holes=holes)
    return polygon


def create_mask(polygons: list, shape=(900, 900), edges=False, edge_thickness=1):
    """Create mask from list of polygons.
    
    :param polygons: list of polygons belonging to one tile
    :type polygons: list of shapely.geometry.Polygons
    :param shape: target shape of mask
    :type shape: tuple of two ints
    :param edges: add house edges to mask if true, defaults to False
    :type edges: boolean, optional
    :param edges_thickness: thickness of building edge, defaults to 1 pixel
    :type edges_thickness: integer, optional
    :return: mask consisting of 0's for background, 1's for house footprint and
        2's for house edges
    :rtype: numpy.ndarray
    """
    mask = rasterio.features.rasterize(polygons, out_shape=shape)
    if edges:
        eroded_img = mask
        # itertively erode the house footprint
        for _ in range(edge_thickness):
            eroded_img = morph.binary_erosion(eroded_img)
        # compare eroded footprints to original footprints
        boundary = mask ^ eroded_img
        # set mask value at boundaries to 2
        mask[boundary==1] = 2
    return mask


def mask_from_id(image_id: str, summary: pd.DataFrame,
                 shape=(900, 900), edges=True, edge_thickness=5):
    """Create the mask correcponding to the ImageId
    
    :param image_id: the id of the image to be masked
    :type image_id: string
    :param summary: summary containing image id and corresponding polygon descriptors
    :type summary: pd.DataFrame
    :param shape: target shape of mask
    :type shape: tuple of two ints
    :param edges: add house edges to mask if true, defaults to False
    :type edges: boolean, optional
    :param edges_thickness: thickness of building edge, defaults to 1 pixel
    :type edges_thickness: integer, optional
    :return: mask of the Image
    :rtype: np.ndarray"""
    image_loc = summary['ImageId'] == image_id
    wkt_strings = summary['PolygonWKT_Pix'][image_loc]
    wkt_strings = wkt_strings[wkt_strings != "POLYGON EMPTY"]
    if len(wkt_strings) == 0:
        return np.zeros((900, 900))
    polygons = map(wkt_string_to_polygon, wkt_strings)
    mask = create_mask(list(polygons), shape, edges, edge_thickness)
    return mask