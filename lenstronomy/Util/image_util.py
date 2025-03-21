__author__ = "sibirrer"

import numpy as np
from scipy import ndimage
import copy
import lenstronomy.Util.util as util

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def add_layer2image(grid2d, x_pos, y_pos, kernel, order=1):
    """Adds a kernel on the grid2d image at position x_pos, y_pos with an interpolated
    subgrid pixel shift of order=order.

    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :param order: interpolation order for sub-pixel shift of the kernel to be added
    :return: image with added layer, cut to original size
    """

    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    kernel_shifted = ndimage.shift(kernel, shift=[-shift_y, -shift_x], order=order)
    return add_layer2image_int(grid2d, x_int, y_int, kernel_shifted)


@export
def add_layer2image_int(grid2d, x_pos, y_pos, kernel):
    """Adds a kernel on the grid2d image at position x_pos, y_pos at integer positions
    of pixel.

    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :return: image with added layer
    """

    k_rows, k_cols = np.shape(kernel)
    if k_rows % 2 == 0 or k_cols % 2 == 0:
        raise ValueError("kernel dimensions must be odd")

    num_rows, num_cols = np.shape(grid2d)
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))

    kernel_y_radius = int((k_rows - 1) / 2)
    kernel_x_radius = int((k_cols - 1) / 2)

    min_row = np.maximum(0, y_int - kernel_y_radius)
    min_col = np.maximum(0, x_int - kernel_x_radius)
    max_row = np.minimum(num_rows, y_int + kernel_y_radius + 1)
    max_col = np.minimum(num_cols, x_int + kernel_x_radius + 1)

    min_k_row = np.maximum(0, -y_int + kernel_y_radius)
    min_k_col = np.maximum(0, -x_int + kernel_x_radius)
    max_k_row = np.minimum(k_rows, -y_int + kernel_y_radius + num_rows)
    max_k_col = np.minimum(k_cols, -x_int + kernel_x_radius + num_cols)

    # The conditions check if the (x_pos, y_pos) is so far outside the grid that
    # the kernel no longer overlaps with the image
    if (
        min_row >= max_row
        or min_col >= max_col
        or min_k_row >= max_k_row
        or min_k_col >= max_k_col
        or (max_row - min_row != max_k_row - min_k_row)
        or (max_col - min_col != max_k_col - min_k_col)
    ):
        return grid2d
    kernel_re_sized = kernel[min_k_row:max_k_row, min_k_col:max_k_col]
    new = grid2d.copy()
    new[min_row:max_row, min_col:max_col] += kernel_re_sized
    return new


@export
def add_background(image, sigma_bkd):
    """Generates background noise to image. To generate a noisy image with background
    noise, generate image_noisy = image + add_background(image, sigma_bkd)

    :param image: pixel values of image
    :param sigma_bkd: background noise (sigma)
    :return: a realisation of Gaussian noise of the same size as image
    """
    nx, ny = np.shape(image)
    background = np.random.randn(nx, ny) * sigma_bkd
    return background


@export
def add_poisson(image, exp_time):
    """Generates a poison (or Gaussian) distributed noise with mean given by surface
    brightness. To generate a noisy image with Poisson noise, perform image_noisy =
    image + add_poisson(image, exp_time)

    :param image: pixel values (photon counts per unit exposure time)
    :param exp_time: exposure time
    :return: Poisson noise realization of input image
    """

    # Gaussian approximation for Poisson distribution, normalized to exposure time
    sigma = np.sqrt(np.abs(image) / exp_time)
    nx, ny = np.shape(image)
    poisson = np.random.randn(nx, ny) * sigma
    return poisson


@export
def rotateImage(img, angle):
    """Querries scipy.ndimage.rotate routine :param img: image to be rotated :param
    angle: angle to be rotated (radian) :return: rotated image."""
    imgR = ndimage.rotate(img, angle, reshape=False)
    return imgR


@export
def shift_image(img, shift):
    """Queries scipy.ndimage.shift routine.

    :param img: image to be shifted
    :param shift: sequence containing x and y shift in pixels
    :return: shifted image
    """
    img_s = ndimage.shift(img, shift)
    return img_s


@export
def re_size_array(x_in, y_in, input_values, x_out, y_out):
    """Resizes 2d array (i.e. image) to new coordinates. So far only works with square
    output aligned with coordinate axis.

    :param x_in:
    :param y_in:
    :param input_values:
    :param x_out:
    :param y_out:
    :return:
    """
    from scipy.interpolate import RectBivariateSpline

    func = RectBivariateSpline(x_in, y_in, z=input_values, kx=1, ky=1, s=0)
    return func(x_out, y_out)


@export
def symmetry_average(image, symmetry):
    """Symmetry averaged image.

    :param image:
    :param symmetry:
    :return:
    """
    img_sym = np.zeros_like(image)
    angle = 360.0 / symmetry
    for i in range(symmetry):
        img_sym += rotateImage(image, angle * i)
    img_sym /= symmetry
    return img_sym


@export
def findOverlap(x_mins, y_mins, min_distance):
    """Finds overlapping solutions, deletes multiples and deletes non-solutions and if
    it is not a solution, deleted as well."""
    n = len(x_mins)
    idex = []
    for i in range(n):
        if i == 0:
            pass
        else:
            for j in range(0, i):
                if (
                    abs(x_mins[i] - x_mins[j]) < min_distance
                    and abs(y_mins[i] - y_mins[j]) < min_distance
                ):
                    idex.append(i)
                    break
    x_mins = np.delete(x_mins, idex, axis=0)
    y_mins = np.delete(y_mins, idex, axis=0)
    return x_mins, y_mins


@export
def coordInImage(x_coord, y_coord, num_pix, deltapix):
    """
    checks whether image positions are within the pixel image in units of arcsec
    if not: remove it

    :returns: image positions within the pixel image
    """
    idex = []
    min_ = -deltapix * num_pix / 2
    max_ = deltapix * num_pix / 2
    for i in range(len(x_coord)):  # sum over image positions
        if (
            x_coord[i] < min_
            or x_coord[i] > max_
            or y_coord[i] < min_
            or y_coord[i] > max_
        ):
            idex.append(i)
    x_coord = np.delete(x_coord, idex, axis=0)
    y_coord = np.delete(y_coord, idex, axis=0)
    return x_coord, y_coord


@export
def re_size(image, factor=1):
    """Re-sizes image with nx x ny to nx/factor x ny/factor.

    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError("scaling factor in re-sizing %s < 1" % factor)
    elif factor == 1:
        return image
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx / f) == nx / f and int(ny / f) == ny / f:
        small = image.reshape([int(nx / f), f, int(ny / f), f]).mean(3).mean(1)
        return small
    else:
        raise ValueError(
            "scaling with factor %s is not possible with grid size %s, %s" % (f, nx, ny)
        )


@export
def rebin_image(bin_size, image, wht_map, sigma_bkg, ra_coords, dec_coords, idex_mask):
    """Re-bins pixels, updates cutout image, wht_map, sigma_bkg, coordinates, PSF.

    :param bin_size: number of pixels (per axis) to merge
    :return:
    """
    numPix = int(len(image) / bin_size)
    numPix_precut = numPix * bin_size
    factor = int(len(image) / numPix)
    if not numPix * bin_size == len(image):
        image_precut = image[0:numPix_precut, 0:numPix_precut]
    else:
        image_precut = image
    image_resized = re_size(image_precut, factor)
    image_resized *= bin_size**2
    wht_map_resized = re_size(wht_map[0:numPix_precut, 0:numPix_precut], factor)
    sigma_bkg_resized = bin_size * sigma_bkg
    ra_coords_resized = re_size(ra_coords[0:numPix_precut, 0:numPix_precut], factor)
    dec_coords_resized = re_size(dec_coords[0:numPix_precut, 0:numPix_precut], factor)
    idex_mask_resized = re_size(idex_mask[0:numPix_precut, 0:numPix_precut], factor)
    idex_mask_resized[idex_mask_resized > 0] = 1
    return (
        image_resized,
        wht_map_resized,
        sigma_bkg_resized,
        ra_coords_resized,
        dec_coords_resized,
        idex_mask_resized,
    )


@export
def rebin_coord_transform(factor, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix):
    """Adopt coordinate system and transformation between angular and pixel coordinates
    of a re-binned image."""
    factor = int(factor)
    Mcoord2pix_resized = Mcoord2pix / factor
    Mpix2coord_resized = Mpix2coord * factor
    x_at_radec_0_resized = (x_at_radec_0 + 0.5) / factor - 0.5
    y_at_radec_0_resized = (y_at_radec_0 + 0.5) / factor - 0.5
    ra_at_xy_0_resized, dec_at_xy_0_resized = util.map_coord2pix(
        -x_at_radec_0_resized, -y_at_radec_0_resized, 0, 0, Mpix2coord_resized
    )
    return (
        ra_at_xy_0_resized,
        dec_at_xy_0_resized,
        x_at_radec_0_resized,
        y_at_radec_0_resized,
        Mpix2coord_resized,
        Mcoord2pix_resized,
    )


@export
def stack_images(image_list, wht_list, sigma_list):
    """Stacks images and saves new image as a fits file.

    :return:
    """
    image_stacked = np.zeros_like(image_list[0])
    wht_stacked = np.zeros_like(image_stacked)
    sigma_stacked = 0.0
    for i in range(len(image_list)):
        image_stacked += image_list[i] * wht_list[i]
        sigma_stacked += sigma_list[i] ** 2 * np.median(wht_list[i])
        wht_stacked += wht_list[i]
    image_stacked /= wht_stacked
    sigma_stacked /= np.median(wht_stacked)
    wht_stacked /= len(wht_list)
    return image_stacked, wht_stacked, np.sqrt(sigma_stacked)


@export
def cut_edges(image, num_pix):
    """Cuts out the edges of a 2d image and returns re-sized image to numPix center is
    well defined for odd pixel sizes.

    :param image: 2d numpy array
    :param num_pix: square size of cut out image
    :return: cutout image with size numPix
    """
    nx, ny = image.shape
    if nx < num_pix or ny < num_pix:
        raise ValueError(
            "image can not be resized, in routine cut_edges with image shape (%s %s) "
            "and desired new shape (%s %s)" % (nx, ny, num_pix, num_pix)
        )
    if (nx % 2 == 0 and ny % 2 == 1) or (nx % 2 == 1 and ny % 2 == 0):
        raise ValueError(
            "image with odd and even axis (%s %s) not supported for re-sizing"
            % (nx, ny)
        )
    if (nx % 2 == 0 and num_pix % 2 == 1) or (nx % 2 == 1 and num_pix % 2 == 0):
        raise ValueError(
            "image can only be re-sized from even to even or odd to odd number."
        )

    x_min = int((nx - num_pix) / 2)
    y_min = int((ny - num_pix) / 2)
    x_max = nx - x_min
    y_max = ny - y_min
    resized = image[x_min:x_max, y_min:y_max]
    return copy.deepcopy(resized)


@export
def radial_profile(data, center):
    """Computes radial profile.

    :param data: 2d numpy array
    :param center: center [x, y] from which pixel to compute the radial profile
    :return: radial profile (in units pixel)
    """
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


@export
def gradient_map(image):
    """Computes gradients of images with the sobel transform.

    :param image: 2d numpy array
    :return: array of same size as input, with gradients between neighboring pixels
    """
    from skimage import filters

    return filters.sobel(image)
