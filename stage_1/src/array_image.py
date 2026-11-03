from numpy import squeeze, dsplit, asarray, dstack, meshgrid, linspace
from scipy.ndimage import map_coordinates

def get_as_rgb_array(image_file):
    """
    Gets the raw image prepared for calculations.

    :param image_file: loaded image file
    :return: numpy image array, an array of three color channels
    """

    # read image as numpy rgb image array
    panorama_array_image = asarray(image_file, dtype="int32")
    # split image in the 3 RGB channels
    return squeeze(dsplit(panorama_array_image, 3))

def sample_rgb_array_image_as_array(coordinates, rgb_array):
    """
    Resampling of the source image

    :param coordinates: meshgrid of numpy arrays where each target coordinate is mapped to a coordinate set
    of the source
    :param rgb_array: the source image as a numpy rgb array representation
    :return: the sampled target image as a scipy rgb array representation
    """
    x = coordinates[0]
    y = coordinates[1]

    # Resample each channel of the source image, "per channel"
    r = map_coordinates(rgb_array[0], [y, x], order=1)
    g = map_coordinates(rgb_array[1], [y, x], order=1)
    b = map_coordinates(rgb_array[2], [y, x], order=1)

    # Merge channels
    return dstack((r, g, b))
