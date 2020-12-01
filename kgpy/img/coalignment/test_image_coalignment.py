import pytest
from kgpy.img.coalignment import image_coalignment
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def image_and_object():
    test_img = np.random.rand(1000,1000)

    transform = np.array([.9,1.1,.1,.1,.5,-.5,37])
    origin= np.array(test_img.shape)
    transform_func= image_coalignment.modified_affine
    initial_crop= (slice(None),slice(50,None))
    initial_pad= ((50,60),(30,70))
    post_transform_size= (2000,2000)
    post_transform_translation= (500,500)
    post_transform_crop= (slice(300,1500),slice(200,1700))

    transform_obj = image_coalignment.ImageTransform(transform,origin,transform_func,initial_crop,initial_pad,post_transform_size,
                                                     post_transform_translation,post_transform_crop)

    return test_img,transform_obj


@pytest.mark.skip('Test with blocking plots')
def test_transform_image():
    test_img, transform_obj = image_and_object()
    transformed_test_image = transform_obj.transform_image(test_img)
    fig,ax = plt.subplots()
    ax = plt.imshow(test_img)
    fig, ax = plt.subplots()
    ax = plt.imshow(transformed_test_image)
    plt.show()


@pytest.mark.skip('Test with blocking plots')
def test_transform_coordinates():
    test_img, transform_obj = image_and_object()
    transformed_test_image = transform_obj.transform_image(test_img)
    transformed_test_image_2, transformed_coordinates = transform_obj.transform_coordinates(test_img)
    transformed_test_image_2 = scipy.ndimage.map_coordinates(transformed_test_image_2
                                                             ,transformed_coordinates)
    transformed_test_image_2 = transformed_test_image_2.T


    dif = transformed_test_image_2 - transformed_test_image
    fig, ax = plt.subplots()
    ax = plt.imshow(transformed_test_image)
    fig, ax = plt.subplots()
    ax = plt.imshow(transformed_test_image_2)
    fig, ax = plt.subplots()
    ax = plt.imshow(dif)

    plt.show()


@pytest.mark.skip('Test with blocking plots')
def test_modified_affine_to_quadratic():
    test_img = np.random.rand(1001,1001)+1

    modified_affine = np.array([1,2,.1,.1,.5,.5,45])
    origin = np.array(test_img.shape)//2
    test_img1 = image_coalignment.modified_affine(test_img,modified_affine,origin)

    quad_transform = image_coalignment.modified_affine_to_quadratic(modified_affine,origin)
    prime_coord = image_coalignment.quadratic_transform(test_img,quad_transform,origin)
    test_img2 = scipy.ndimage.map_coordinates(test_img,prime_coord)
    test_img2 = test_img2.T

    fig,ax = plt.subplots()
    dif_img = test_img2 - test_img1
    ax = plt.imshow(dif_img)
    plt.show()

    print(dif_img.max())



