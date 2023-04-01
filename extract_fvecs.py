import random
from glob import glob
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import json
import datetime
import git
from tqdm import tqdm
import os
import copy

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
import logging

# tf.get_logger().setLevel(logging.ERROR)


def build_model_EfficientNetB7(seed):
    inputs = keras.layers.Input(shape=(600, 600, 3))

    x = inputs

    img_mean = [152.33031877955463, 106.26509461301819, 104.55854576464021]
    img_std = [36.21155284418057, 30.75150171154211, 31.2230456008511]
    normalization_layer = tf.keras.layers.Normalization(axis=-1, mean=img_mean, variance=img_std)
    x = normalization_layer(x)
    eff_model = keras.applications.efficientnet.EfficientNetB7(include_top=False, input_tensor=x, weights="imagenet")
    eff_model.trainable = False
    x = eff_model(x)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    model = tf.keras.Model(inputs, x, name="EfficientNet")
    # print model summary
    # model.summary()

    return model


def image_preprocessing(tf_im, image_id):
    cropped_im = image_cropping(tf_im)
    resized_im = tf.image.resize(cropped_im, [600, 600])
    return resized_im, image_id


def image_cropping(original_tf_image):
    zero_number = tf.constant(0, dtype=tf.float32)
    none_zero_bool_where = tf.not_equal(original_tf_image, zero_number)
    indices = tf.where(none_zero_bool_where)
    y_nonzero = indices[:, 0]
    x_nonzero = indices[:, 1]
    y_len = tf.math.reduce_max(y_nonzero) - tf.math.reduce_min(y_nonzero) + 1
    x_len = tf.math.reduce_max(x_nonzero) - tf.math.reduce_min(x_nonzero) + 1
    max_len = tf.math.maximum(x_len, y_len)
    tongue_im = original_tf_image[int(tf.math.reduce_min(y_nonzero)):int(tf.math.reduce_max(y_nonzero)) + 1,
                int(tf.math.reduce_min(x_nonzero)):int(tf.math.reduce_max(x_nonzero)) + 1, :]

    if x_len < max_len:
        zero_slice = tf.zeros(shape=(max_len, (max_len - x_len), 3), dtype=tf.float32)
        tongue_im = tf.concat([tongue_im, zero_slice], axis=1)
    elif y_len < max_len:
        zero_slice = tf.zeros(shape=((max_len - y_len), max_len, 3), dtype=tf.float32)
        tongue_im = tf.concat([tongue_im, zero_slice], axis=0)

    # if x_len < max_len:
    #     zero_slice = tf.zeros(shape=(max_len, (max_len - x_len) // 2, 3), dtype=tf.float32)
    #     tongue_im = tf.concat([zero_slice, tongue_im, zero_slice], axis=1)
    # elif y_len < max_len:
    #     zero_slice = tf.zeros(shape=((max_len - y_len) // 2, max_len, 3), dtype=tf.float32)
    #     tongue_im = tf.concat([zero_slice, tongue_im, zero_slice], axis=0)

    return tongue_im


def define_image_decoder(img_height, img_width):
    def image_decoder(file_path):
        img = tf.io.read_file(file_path)
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size

        val = tf.strings.split(file_path, sep='/')[-1]
        image_id = tf.strings.regex_replace(val, ".png", "")
        # image_id = tf.strings.to_number(val, out_type=tf.dtypes.int32)
        return tf.image.resize(img, [img_height, img_width]), image_id

    return image_decoder


def sort_relevant_files(files, processed_ids, flavor):
    # images = [39872, 7265, 40516, 39845, 72774, 49800, 68009, 46762, 46795, 39820, 7278, 67857, 37951, 10968, 39933, 10879]
    # images = [f'B{im}' for im in images]
    relevant_files = []
    for specific_file in files:
        image_id = specific_file.split('/')[-1].replace('.png', '')
        if flavor in image_id and image_id not in processed_ids:
            # if image_id in images:
            relevant_files.append(specific_file)
    return relevant_files


def generate_ds(files, batch_size, height, width):
    list_ds = tf.data.Dataset.list_files(files, shuffle=False)
    image_decoder = define_image_decoder(img_height=height, img_width=width)
    ds = list_ds.map(image_decoder, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(image_preprocessing)
    ds = ds.batch(batch_size)
    return ds


def generate_efficientnet_feature_vectors(db_address: str, dest_path: str,
                                          batch_size, seed, height=4000, width=3000):
    """
      extract feature vectors for all images in the source path
      :param db_address: images path
      :param dest_path: path of the json file to written at
      :param json_file_name: name of the json file
      :param batch_size: batch_size
      :param seed: random seed
      :param height: original height of the images in db_address
      :param width: original width of the images in db_address
      :return: None
      """

    files = glob(f'{db_address}/**.jpg', recursive=True)
    # todo: remove next sentence
    files = files[:10]
    # files = sort_relevant_files(files, processed_ids, flavor)
    if len(files) == 0:
        print('No images to process')
        return
    ds = generate_ds(files=files, batch_size=batch_size, height=640, width=480)
    for val in ds:
        print(val)
        break
    exit()
    # ignores failed elements in the dataset automatically
    # ds = ds.apply(tf.data.experimental.ignore_errors())

    # feature_extractor = build_model_EfficientNetB7(seed=seed)
    #
    # with tqdm(total=len(files) // batch_size) as pbar:
    #     for idx, (images, image_ids) in enumerate(ds):
    #         feature_vectors = feature_extractor(images)
    #         for single_im_id, f_vec in zip(image_ids, feature_vectors):
    #             feature_vectors_dict[single_im_id.numpy().decode('utf8')] = f_vec.numpy().tolist()
    #         pbar.update(1)
    #         if idx % 25 == 0:
    #             with open(f'{dest_path}/augment_{augment}_flavor_{flavor}_{json_file_name}.json',
    #                       'a') as f:
    #                 f.write(json.dumps(feature_vectors_dict, indent=4))
    #                 f.write("\u000a")
    #             dataset.add_files(path=os.path.join(os.path.join(
    #                 f'{dest_path}/augment_{augment}_flavor_{flavor}_{json_file_name}.json')))
    #
    #             dataset.upload()
    #             del feature_vectors_dict
    #             feature_vectors_dict = {}
    #
    # with open(f'{dest_path}/augment_{augment}_flavor_{flavor}_{json_file_name}.json', 'a') as f:
    #     f.write(json.dumps(feature_vectors_dict, indent=4))
    #     f.write("\u000a")
    # dataset.add_files(path=os.path.join(os.path.join(
    #     f'{dest_path}/augment_{augment}_flavor_{flavor}_{json_file_name}.json')))
    # dataset.upload()
    #
    # dataset.finalize()

    # # test is address corresponds to the image in the ds
    # (the ds needs to be unbatched in order for the following code to work)
    # for image, address in complete_ds:
    #     ads = address.numpy().decode("utf-8")
    #     im_address = np.array(Image.open(ads))
    #     im_address = image_cropping(im_address)
    #     im_address = tf.image.resize(im_address, [600, 600], method=tf.image.ResizeMethod.BILINEAR)
    #     im_address = im_address.numpy().astype(np.uint8)
    #     im_ds = image.numpy().astype('uint8')
    #
    #     print(np.linalg.norm(im_ds - im_address))
    #     # test if there is a numeric difference between the images
    #     if np.linalg.norm(im_ds - im_address) > 0:
    #         f, axr = plt.subplots(2, 1)
    #         axr[0].imshow(im_ds)
    #         axr[1].imshow(im_address)
    #         plt.show()


def main():
    db_address = '/datashare/APAS/frames/P016_balloon1_side'
    dest_path = f'{os.getcwd()}/efficientnet/B0'
    batch_size = 64
    seed = 100
    generate_efficientnet_feature_vectors(db_address=db_address,
                                          batch_size=batch_size,
                                          dest_path=dest_path,
                                          seed=seed)


if __name__ == '__main__':
    main()
