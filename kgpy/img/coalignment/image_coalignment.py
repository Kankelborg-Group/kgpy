import pathlib
import pickle
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import typing as typ
import copy

# from kso.kso_iii.science.instruments.esis import level_3
import scipy.signal
import scipy.ndimage
import scipy.optimize

from dataclasses import dataclass


@dataclass
class TransformCube:
    '''
    For a given cube of images this class with contain a similar cube of ImageTransform Objects for transforming
    an entire cube of images.
    '''

    transform_cube: typ.List[typ.List['ImageTransform']]

    def to_pickle(self, path: pathlib.Path):
        file = open(str(path), 'wb')
        pickle.dump(self, file)
        file.close()

        return

    @staticmethod
    def from_pickle(path: pathlib.Path) -> 'TransformCube':
        file = open(str(path), 'rb')
        obj = pickle.load(file)
        file.close()

        return obj


@dataclass
class ImageTransform:
    '''
    Class tracking transformations done to an image during co-alignment for repeating later or for use in a forward model
    '''

    transform: np.ndarray
    origin: np.ndarray
    transform_func: typ.Callable
    initial_crop: typ.Tuple[slice, ...]
    initial_pad: typ.Tuple[typ.Tuple[int, int], ...]
    post_transform_size: tuple
    post_transform_translation: np.ndarray
    post_transform_crop: tuple

    # If TransformCube being a list is annoying then this might be a better approach.

    # def __getitem__(self ,indices: typ.Tuple[int,...]) -> 'ImageTransform':
    #     transform = self.transform[indices, ...]
    #     origin = self.origin[indices, ...]
    #     initial_crop = self.initial_crop[indices, ...]
    #     initial_pad = self.initial_pad[indices, ...]
    #     post_transform_size = self.post_transform_size[indices, ...]
    #     post_transform_translation = self.post_transform_translation[indices, ...]
    #     post_transform_crop = self.post_transform_crop[indices, ...]
    #
    #     return ImageTransform(transform,origin,self.transform_func,initial_crop,initial_pad,post_transform_size,
    #                           post_transform_translation,post_transform_crop)

    def to_pickle(self, path: pathlib.Path):

        file = open(str(path), 'wb')
        pickle.dump(self, file)
        file.close()

        return

    @staticmethod
    def from_pickle(path: pathlib.Path) -> 'ImageTransform':

        file = open(str(path), 'rb')
        obj = pickle.load(file)
        file.close()

        return obj

    def transform_image(self, img) -> np.ndarray:
        '''Take in image and perform all transformation operations stored in the ImageTransform object. '''

        img = self.img_pre_process(img)

        if self.transform_func == modified_affine:
            img = self.transform_func(img, self.transform, self.origin)
        else:
            transformed_coord = self.transform_func(img, self.transform, self.origin)
            img = scipy.ndimage.map_coordinates(img, transformed_coord)
            img = img.T

        img = self.img_post_process(img)

        return img

    def img_post_process(self, img):
        big_img = np.empty(self.post_transform_size)
        big_img[0:img.shape[0], 0:img.shape[1]] = img
        big_img = np.roll(big_img, self.post_transform_translation, (0, 1))
        img = big_img[self.post_transform_crop]
        return img

    def img_pre_process(self, img, reverse=None):
        img = img[self.initial_crop]
        img = np.pad(img, self.initial_pad)
        return img

    def transform_coordinates(self, coords) -> np.ndarray:
        '''
        If you only want the transformed coordinates and not the transformed images to avoid extra interpolation.
        '''

        coords = self.coord_pre_process(coords)

        dummy_img = np.empty((1, 1))
        if self.transform_func == modified_affine:
            quad_transform = modified_affine_to_quadratic(self.transform, self.origin)
            coords = quadratic_transform(dummy_img, quad_transform, self.origin, old_coord=coords)
        else:
            coords = self.transform_func(dummy_img, self.transform, self.origin, old_coord=coords)

        coords = self.coord_post_process(coords)

        return coords

    def coord_post_process(self, coords, reverse=None):
        post_crop = np.array([self.post_transform_crop[1].start, self.post_transform_crop[0].start], dtype=float)
        post_crop[np.isnan(post_crop)] = 0
        if reverse:
            coords = coords - np.array(self.post_transform_translation[::-1]).reshape(2, 1, 1) + post_crop.reshape(2, 1,
                                                                                                                   1)
        else:
            coords = coords + np.array(self.post_transform_translation[::-1]).reshape(2, 1, 1) - post_crop.reshape(2, 1,
                                                                                                                   1)
        return coords

    def coord_pre_process(self, coords, reverse=None):
        crop = np.array([self.initial_crop[1].start, self.initial_crop[0].start], dtype=float)
        crop[np.isnan(crop)] = 0
        pad = self.initial_pad
        pad = np.array([pad[1][0], pad[0][0]])
        if reverse:
            coords = coords + crop.reshape(2, 1, 1) - pad.reshape(2, 1, 1)
        else:
            coords = coords - crop.reshape(2, 1, 1) + pad.reshape(2, 1, 1)
        return coords

    def invert_quadratic_transform(self, img) -> np.ndarray:
        '''
        Given a transformation object and the originally transformed image the transformation step can be inverted
        using this routine.

        '''

        img = self.img_pre_process(img)
        coords = get_img_coords(img)
        coords = coords - self.origin.reshape(2, 1,
                                              1)  # origin shift included in quadratic transform was influencing the inversion

        prime_coords = quadratic_transform(img, self.transform, self.origin)
        prime_coords = prime_coords - self.origin.reshape(2, 1,
                                                          1)  # origin shift included in quadratic transform was influencing the inversion

        prime_coord_power = np.array(
            [np.ones_like(prime_coords[0, :]), prime_coords[0, :], prime_coords[1, :],
             prime_coords[0, :] * prime_coords[1, :], np.square(prime_coords[0, :]),
             np.square(prime_coords[1, :])])
        cp_shp = prime_coord_power.shape
        prime_coord_power = prime_coord_power.reshape(cp_shp[0], cp_shp[1] * cp_shp[2])
        coords = coords.reshape(2, cp_shp[1] * cp_shp[2])

        transform = np.linalg.lstsq(prime_coord_power.T, coords.T)

        return transform[0].T.reshape(12)


def normalized_cc(im1, im2, mode='same'):
    if im1.std() != 0:
        im1 = (im1 - im1.mean()) / im1.std()
    if im2.std() != 0:
        im2 = (im2 - im2.mean()) / im2.std()
    cc = scipy.signal.correlate(im1, im2, mode=mode)
    if mode == 'full':
        cc /= cc.size / 2
    if mode == 'same':
        cc /= cc.size
    return cc


def test_alignment_quality(transform: np.ndarray, im1: np.ndarray, im2: np.ndarray,
                           transformation_object: 'ImageTransform') -> float:
    """
    Apply a chosen transform to Image 1 and cross-correlate with Image 2 to check coalignment
    Designed as a merit function for scipy.optimize routines

    If origin isn't specified it is assumed to be (0,0).  If moving_origin is set to True, origin = transform[-2:]
    so that it can be controlled by scipy.optimize routines.
    """
    # temp_transform_object = copy.deepcopy(transformation_object)

    transformation_object.transform = transform
    im1 = transformation_object.transform_image(im1)

    cc = normalized_cc(im1, im2)

    # fit_quality = np.max(cc)
    center = np.array(cc.shape) // 2
    fit_quality = cc[center[0], center[1]]

    # print(fit_quality)
    return -fit_quality


def alignment_quality(transform: np.ndarray, im1: np.ndarray, im2: np.ndarray, transform_func, origin=np.array([0, 0]),
                      moving_origin=False, kwargs={}) -> float:
    """
    Apply a chosen transform to Image 1 and cross-correlate with Image 2 to check coalignment
    Designed as a merit function for scipy.optimize routines

    If origin isn't specified it is assumed to be (0,0).  If moving_origin is set to True, origin = transform[-2:]
    so that it can be controlled by scipy.optimize routines.
    """

    if moving_origin == True:
        origin = transform[-2:]
        transform = transform[:-2]

    transformed_coord = transform_func(im1, transform, origin, **kwargs)
    im1 = scipy.ndimage.map_coordinates(im1, transformed_coord)
    im1 = im1.T
    cc = normalized_cc(im1, im2)

    fit_quality = np.max(cc)

    # print(fit_quality)
    return -fit_quality


def affine_alignment_quality(transform, im1, im2):
    # scale_x, scale_y, shear_x, shear_y, disp_x, disp_y, rot_angle = transform

    origin = np.array(im1.shape) // 2
    im1 = modified_affine(im1, transform, origin)
    cc = normalized_cc(im1, im2)
    fit_quality = np.max(cc)
    print(fit_quality)

    return -fit_quality


def modified_affine(img: np.ndarray, transform: np.ndarray, origin: np.ndarray, order=3, prefilter=True) -> np.ndarray:
    '''
    scale_x, scale_y, shear_x, shear_y, disp_x, disp_y, rot_angle = transform
    '''
    affine_param = affine_params(origin, *transform)
    img = scipy.ndimage.affine_transform(img, *affine_param, order=order, prefilter=prefilter)
    return img


def quadratic_transform(img: np.ndarray, transform: np.ndarray, origin: np.ndarray, old_coord=None) -> np.ndarray:
    """
    Apply a quadratic coordinate transformation of the form:
    x' = transform[0] + transform[1]*x + transform[2]*y + transform[3]*x*y + transform[4]*x^2 + transform[5]*y^2
    y' = transform[6] + transform[7]*x + transform[8]*y + transform[9]*x*y + transform[10]*x^2 + transform[11]*y^2
    about the specified origin = (x = origin[0], y = origin[1])

    returns primed coordinates for use with scipy.ndimage.map_coordinates()
    """

    if old_coord is None:
        coord = get_img_coords(img)
        coord = coord - origin.reshape(2, 1, 1)
    else:
        coord = old_coord - origin.reshape(2, 1, 1)

    coord_power = np.array(
        [np.ones_like(coord[0, :]), coord[0, :], coord[1, :], coord[0, :] * coord[1, :], np.square(coord[0, :]),
         np.square(coord[1, :])])

    x_prime = coord_power * transform[0:6].reshape(6, 1, 1)
    y_prime = coord_power * transform[6:].reshape(6, 1, 1)

    prime_coord = np.array([x_prime.sum(axis=0), y_prime.sum(axis=0)])
    prime_coord = prime_coord + origin.reshape(2, 1, 1)

    return prime_coord


def get_img_coords(img):
    m_shape = img.shape
    coord = np.array(np.meshgrid(np.arange(m_shape[0]), np.arange(m_shape[1])))
    return coord


def affine_params(origin, scale_x, scale_y, shear_x, shear_y, disp_x, disp_y, rot_angle):
    '''
    scale_x, scale_y, shear_x, shear_y, disp_x, disp_y, rot_angle = transform
    '''

    c_theta = np.cos(np.radians(rot_angle))
    s_theta = np.sin(np.radians(rot_angle))

    rot = np.array([[c_theta, -s_theta], [s_theta, c_theta]])
    shear = np.array([[1, shear_x], [shear_y, 1]])
    scale = np.array([[scale_y, 0], [0, scale_x]])

    m = np.matmul(rot, np.matmul(shear, scale))
    m = np.linalg.inv(m)
    offset = np.array(origin) - np.array([disp_x, disp_y]) - np.dot(m, np.array(origin))

    return m, offset


def modified_affine_to_quadratic(transform: np.ndarray, origin: np.ndarray) -> np.ndarray:
    '''
    Since scipy.ndimage.affine_transform goes straight to an image transform instead of through map_coordinates
    this will convert the transform parameters used in modified affine for use with quadratic transform
    '''

    affine_m = affine_params(origin, *transform)[0]
    quad_transform = np.array([-transform[4], affine_m[0, 0], affine_m[0, 1], 0, 0, 0,
                               -transform[5], affine_m[1, 0], affine_m[1, 1], 0, 0, 0])
    return quad_transform

# path = level_3.default_pickle_path
# quadratic_path = path.parent / 'esis_Level3_quadratic.pickle'
#
# lin_lev3 = level_3.Level3.from_pickle(path)
# quad_lev3 = level_3.Level3.from_pickle(quadratic_path)
#
# img_cube_shp = lin_lev3.intensity.shape
# # sequence = np.arange(30-4)+2
# # camera_combos = [[0,1],[1,0],[0,2],[0,3],[1,2],[1,3],[2,3]]
# camera_combos = [[1,2]]
# sequence = [15]
# r0 = [500]
# # r0 = [2000]
#
# for r0 in r0:
#     x, y = np.meshgrid(np.arange(img_cube_shp[-2]),np.arange(img_cube_shp[-1]))
#     x0, y0 = np.array([img_cube_shp[-2],img_cube_shp[-1]])//2
#     r = np.sqrt((x-x0)**2 + (y-y0)**2)
#     window = np.exp(-(r / r0) ** 4)
#     # window = 1
#
#
#
#
#     for camera1, camera2 in camera_combos:
#         for i in sequence:
#             guess = [0,1,0,0,0,0,
#                      0,0,1,0,0,0]   #identity transform
#             origin = [635,635]
#             guess = np.array(guess+origin)
#
#             transform_func = quadratic_transform
#
#
#             image_1 = lin_lev3.intensity[i, camera1, ...] * window
#             image_2 = lin_lev3.intensity[i, camera2, ...] * window
#
#
#             initial_alignment = alignment_quality(guess, image_1, image_2, transform_func)
#
#
#             fit = scipy.optimize.minimize(alignment_quality, guess, (image_1, image_2, transform_func))
#
#             print('Initial Alignment = ', initial_alignment,'  New Alignment = ', fit.fun)
#             initial_coord = transform_func(image_1, guess[:-2], guess[-2:])
#             transformed_coord = transform_func(image_1, fit.x[:-2], fit.x[-2:])
#
#             coord_dif = np.sqrt(np.square(transformed_coord[0,...]-initial_coord[0,...]) + \
#                         np.square(transformed_coord[1,...]-initial_coord[1,...]))
#
#             image_1 = scipy.ndimage.map_coordinates(image_1, transformed_coord)
#             image_1 = image_1.T
#
#             dif_image = (image_1 / np.median(image_1)) - (image_2 / np.median(image_2))
#
#             fig, ax = plt.subplots()
#             ax = plt.imshow(dif_image)
#             plt.pause(1)
#
#
#             fig, ax = plt.subplots()
#             ax = plt.imshow(coord_dif)
#             plt.pause(1)
#
#
#
#
#
#
#
#             # quad_im1 = quad_lev3.intensity[sequence, camera1, ...]
#             # quad_im2 = quad_lev3.intensity[sequence, camera2, ...]
#             #
#             # quad_fit = scipy.optimize.minimize(alignment_quality,x0,args = (quad_im1,quad_im2,quadratic_transform))
#
#
#
#
#
#
#
# # Initial Alignment =  -0.9684151508545974   New Alignment =  -0.9694067021063304
# plt.show()
#
