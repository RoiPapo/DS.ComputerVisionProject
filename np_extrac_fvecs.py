import random
from glob import glob
from PIL import Image
import numpy as np
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


def build_model_EfficientNet(model_type, model_input_size):
    inputs = keras.layers.Input(shape=(model_input_size, model_input_size, 3))

    img_mean = [152.33031877955463, 106.26509461301819, 104.55854576464021]
    img_std = [36.21155284418057, 30.75150171154211, 31.2230456008511]
    normalization_layer = tf.keras.layers.Normalization(axis=-1, mean=img_mean, variance=img_std)
    x = normalization_layer(inputs)
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


def generate_efficientnet_feature_vectors(ds: str,
                                          dest_path: str,
                                          labels_folder_dict,
                                          model_index,
                                          model_input_size,
                                          batch_size):
    """
      extract feature vectors for all images in the source path
      :param ds: images path
      :param dest_path: path of the json file to written at
      :param folder_path: name of the videos folder file
      :param model_index: batch_size
      :param model_input_size: random seed
      :return: None
      """

    feature_extractor = build_model_EfficientNet(model_index, model_input_size=model_input_size)
    out_feature_vectors = None
    processed_labels = []
    with tqdm(total=len(ds.file_paths) // batch_size) as pbar:
        for idx, (images, image_ids) in enumerate(ds):
            feature_vectors = feature_extractor(images)
            for inner_idx, (single_im_id, f_vec) in enumerate(zip(image_ids, feature_vectors)):
                video_name = ds.class_names[single_im_id]
                frame_name = ds.file_paths[idx * batch_size + inner_idx].split('/')[-1]
                if video_name not in processed_labels:
                    if out_feature_vectors is not None:
                        np.save(f'{dest_path}/{video_name}.npy', out_feature_vectors.T)
                    out_feature_vectors = f_vec.numpy()
                    processed_labels.append(video_name)

                else:
                    out_feature_vectors = np.vstack([out_feature_vectors, f_vec.numpy()])

            pbar.update(1)

        np.save(f'{dest_path}/{video_name}.npy', out_feature_vectors.T)


def main():
    model_params_dict = {0: 224, 1: 240, 2: 260, 3: 300, 4: 380, 5: 456, 6: 528, 7: 600}
    folders = ['P016_balloon1_side', 'P016_balloon1_top', 'P016_balloon2_side', 'P016_balloon2_top']
    labels_folder_dict = {i: folders[i] for i in range(len(folders))}

    # db_address = '/datashare/APAS/frames'
    db_address = '/home/user/datasets/frames'
    batch_size = 64
    seed = 100
    for model_index, model_input_size in model_params_dict.items():
        print(f"Starting model: {model_index}")
        ds = tf.keras.utils.image_dataset_from_directory(
            '/home/user/datasets/frames/',
            seed=seed,
            labels='inferred',
            label_mode='int',
            class_names=list(labels_folder_dict.values()),
            shuffle=False,
            image_size=(model_input_size, model_input_size),
            batch_size=batch_size)

        dest_path = f'{os.getcwd()}/efficientnet/B{model_index}'
        os.makedirs(dest_path, exist_ok=True)
        generate_efficientnet_feature_vectors(ds=ds,
                                              dest_path=dest_path,
                                              labels_folder_dict=labels_folder_dict,
                                              model_index=model_index,
                                              model_input_size=model_input_size,
                                              batch_size=batch_size)


if __name__ == '__main__':
    main()
