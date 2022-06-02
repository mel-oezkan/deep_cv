import os
import numpy as np
import tensorflow as tf
import create_labels
import cv2
import tqdm
tf.keras.backend.clear_session()

"""In this notebook we will smartly resize our data and save it.

For that we will stick to our paths and will write our data to:"""

path_rgb_128 = "../datasets/train/AOI_11_Rotterdam/PS-RGB_128"
path_sar_128 = "../datasets/train/AOI_11_Rotterdam/SAR-Intensity_128"
path_labels_128 = "../datasets/train/AOI_11_Rotterdam/Labels_128"


for dir in [path_rgb_128,path_sar_128,path_labels_128]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Ids of images to resize
image_ids = [i[34:-4] for i in os.listdir("../datasets/train/AOI_11_Rotterdam/PS-RGB")]


def find_end_of_black_bar(img):

    """ This functions finds the end of the black bar for an image, where there is a horizontal black bar on top
    :param img: 3d numpy array of image
    
    :return: Column of end of black bar"""

    # go from top to bottom
    i = 0

    # if row is black bar go on
    while not img[i,:,:].sum():
        i+=1

    return i

def clever_resize(img,end_of_black_bar,target_shape=[128,128]):
    """
    :param img: 3d img array
    :param end_of_black_bar: end of black bar
    :param target_shape: shape to be resized to
    :return: resized imgage:"""

    # image wizout bar
    with_out_black = img[end_of_black_bar+1:]
    # convert
    with_out_black = tf.image.convert_image_dtype(with_out_black, tf.float32)
    # resize
    return tf.image.resize(with_out_black, target_shape)


def resize_images(img_id,target_shape):
    """
    image_id: Id of image you want to resize
    target_shape: Shape to resize to
    return: resized images"""
    # boolean - Whether image has odd id
    is_odd_id = int(img_id[-1])%2

    # reading paths
    read_rgb_path = f"../datasets/train/AOI_11_Rotterdam/PS-RGB/SN6_Train_AOI_11_Rotterdam_PS-RGB_{img_id}.tif"
    read_sar_path = f"../datasets/train/AOI_11_Rotterdam/SAR-Intensity/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"

    # read images: if even id flip horizontally
    rgb_img = cv2.imread(read_rgb_path,cv2.IMREAD_UNCHANGED) if is_odd_id else cv2.imread(read_rgb_path,cv2.IMREAD_UNCHANGED)[::-1,:,:]
    sar_img = cv2.imread(read_sar_path,cv2.IMREAD_UNCHANGED) if is_odd_id else cv2.imread(read_sar_path,cv2.IMREAD_UNCHANGED)[::-1,:,:]

    # create mask without edge
    mask_img = create_labels.mask_from_id(img_id, summary, edges=False)
    # expand dims and flip if horizontal
    mask_img = np.expand_dims(mask_img,2) if is_odd_id else np.expand_dims(mask_img,2)[::-1,:,:]

    # find end of black bar      
    end_of_black_bar = find_end_of_black_bar(rgb_img)

    # resize images
    rgb_img, sar_img, mask_img = map(lambda img: clever_resize(img,end_of_black_bar,target_shape=target_shape),[rgb_img,sar_img,mask_img])
    # to numpy
    mask_img = mask_img.numpy()
    # redo labels/ different values to to bilinear
    mask_img[mask_img>0] = 1

    # if even id reglip horizontally
    if not is_odd_id:
        rgb_img = rgb_img[::-1,:,:]
        sar_img = sar_img[::-1,:,:]
        mask_img = mask_img[::-1,:,:]

    # return images
    return rgb_img.numpy(),sar_img.numpy(),mask_img.reshape(target_shape)

def write_resized(img_id,target_shape=[128,128]):
    """This functions resizes the images of a given id and writes to to the dirs
    image_id: id of image to be changed/str
    return: None"""
    rgb_img_res, sar_img_res, mask_img_res = resize_images(img_id,target_shape)

    # paths to write to
    write_rgb_path = f"../datasets/train/AOI_11_Rotterdam/PS-RGB_128/SN6_Train_AOI_11_Rotterdam_PS-RGB_{img_id}.tif"
    write_sar_path = f"../datasets/train/AOI_11_Rotterdam/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"
    write_label_path = f"../datasets/train/AOI_11_Rotterdam/Labels_128/{img_id}.tif"

    # write images to paths
    for img,path in zip([rgb_img_res, sar_img_res, mask_img_res],[write_rgb_path,write_sar_path,write_label_path]):

        cv2.imwrite(path,img) 


if __name__ == "__main__":
    # resize all images
    for img_id in tqdm.tqdm(image_ids):

        write_resized(img_id)   