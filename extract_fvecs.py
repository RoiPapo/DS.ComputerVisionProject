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
import os
import tensorflow as tf
from tensorflow import keras
import logging


def chose_model(model_type):
    if model_type == 0:
        eff_model = keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet")
    elif model_type == 1:
        eff_model = keras.applications.efficientnet.EfficientNetB1(include_top=False, weights="imagenet")
    elif model_type == 2:
        eff_model = keras.applications.efficientnet.EfficientNetB2(include_top=False, weights="imagenet")
    elif model_type == 3:
        eff_model = keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet")
    elif model_type == 4:
        eff_model = keras.applications.efficientnet.EfficientNetB4(include_top=False, weights="imagenet")
    elif model_type == 5:
        eff_model = keras.applications.efficientnet.EfficientNetB5(include_top=False, weights="imagenet")
    elif model_type == 6:
        eff_model = keras.applications.efficientnet.EfficientNetB6(include_top=False, weights="imagenet")
    elif model_type == 7:
        eff_model = keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet")
    else:
        return 'Wrong model index'
    return eff_model


def build_model_EfficientNet(model_type, original_height, original_width, model_input_size):
    inputs = keras.layers.Input(shape=(original_height, original_width, 3))
    x = keras.layers.Resizing(height=model_input_size, width=model_input_size)(inputs)

    img_mean = [152.33031877955463, 106.26509461301819, 104.55854576464021]
    img_std = [36.21155284418057, 30.75150171154211, 31.2230456008511]
    normalization_layer = tf.keras.layers.Normalization(axis=-1, mean=img_mean, variance=img_std)
    x = normalization_layer(x)
    eff_model = chose_model(model_type=model_type)
    eff_model.trainable = False
    x = eff_model(x)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    model = tf.keras.Model(inputs, x, name="EfficientNet")
    # print model summary
    # model.summary()

    return model


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


def generate_ds(files, batch_size, original_height, original_width):
    list_ds = tf.data.Dataset.list_files(files, shuffle=False)
    image_decoder = define_image_decoder(img_height=original_height, img_width=original_width)
    ds = list_ds.map(image_decoder, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds


def generate_efficientnet_feature_vectors(folder_path: str, dest_path: str,
                                          batch_size, model_index, model_input_size,
                                          seed):
    """
      extract feature vectors for all images in the source path
      :param db_address: images path
      :param dest_path: path of the json file to written at
      :param folder_path: name of the videos folder file
      :param batch_size: batch_size
      :param seed: random seed
      :param height: original height of the images in db_address
      :param width: original width of the images in db_address
      :return: None
      """

    files = glob(f'{folder_path}/**.jpg', recursive=True)

    if len(files) == 0:
        print('No images to process')
        return
    ds = generate_ds(files=files,
                     batch_size=batch_size,
                     original_height=640,
                     original_width=480)

    feature_extractor = build_model_EfficientNet(model_index,
                                                 original_height=640,
                                                 original_width=480,
                                                 model_input_size=model_input_size)
    feature_vectors_dict = {}
    for idx, (images, image_ids) in enumerate(ds):
        feature_vectors = feature_extractor(images)
        for single_im_id, f_vec in zip(image_ids, feature_vectors):
            feature_vectors_dict[single_im_id.numpy().decode('utf8')] = f_vec.numpy().tolist()
        if idx % 25 == 0:
            with open(f'{dest_path}.json', 'a') as f:
                f.write(json.dumps(feature_vectors_dict, indent=4))
                f.write("\u000a")

            del feature_vectors_dict
            feature_vectors_dict = {}

    with open(f'{dest_path}.json', 'a') as f:
        f.write(json.dumps(feature_vectors_dict, indent=4))
        f.write("\u000a")

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
    model_params_dict = {0: 224, 1: 240, 2: 260, 3: 300, 4: 380, 5: 456, 6: 528, 7: 600}
    db_address = '/datashare/APAS/frames'
    batch_size = 64
    seed = 100
    videos_folders_path = glob(f'{db_address}/**', recursive=False)
    for model_index, model_input_size in model_params_dict.items():
        for folder_path in tqdm(videos_folders_path):
            folder = folder_path.split('/')[-1]
            dest_path = f'{os.getcwd()}/efficientnet/B{model_index}'
            os.makedirs(dest_path, exist_ok=True)
            generate_efficientnet_feature_vectors(folder_path=folder_path,
                                                  batch_size=batch_size,
                                                  dest_path=f'{dest_path}/{folder}',
                                                  model_index=model_index,
                                                  model_input_size=model_input_size,
                                                  seed=seed)


if __name__ == '__main__':
    main()
