import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


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

    # feature_extractor = build_model_EfficientNet(model_index, model_input_size=model_input_size)
    # out_feature_vectors = None
    # processed_labels = []
    # with tqdm(total=len(ds.file_paths) // batch_size) as pbar:
    #     for idx, (images, image_ids) in enumerate(ds):
    #         if image_ids[-1].numpy() == 100:
    #             return
    #         feature_vectors = feature_extractor(images)
    #         for inner_idx, (single_im_id, f_vec) in enumerate(zip(image_ids, feature_vectors)):
    #             video_name = ds.class_names[single_im_id]
    #             frame_name = ds.file_paths[idx * batch_size + inner_idx].split('/')[-1]
    #             if video_name not in processed_labels:
    #                 if out_feature_vectors is not None:
    #                     np.save(f'{dest_path}/{video_name}.npy', out_feature_vectors.T)
    #                 out_feature_vectors = f_vec.numpy()
    #                 processed_labels.append(video_name)
    #
    #             else:
    #                 out_feature_vectors = np.vstack([out_feature_vectors, f_vec.numpy()])
    #
    #         pbar.update(1)
    #
    #     np.save(f'{dest_path}/{video_name}.npy', out_feature_vectors.T)

    feature_extractor = build_model_EfficientNet(model_index, model_input_size=model_input_size)
    out_feature_vectors = None
    processed_labels = []
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    idx = 0
    with tqdm(total=len(ds.file_paths) // batch_size) as pbar:
        with tf.compat.v1.Session() as sess:
            while True:
                try:
                    batch = sess.run(iterator.get_next())
                    print(batch)
                    images, image_ids = batch
                    if image_ids[-1] == 100:
                        return
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
                    idx += 1
                except tf.errors.OutOfRangeError:
                    break


def main():
    model_params_dict = {0: 224, 1: 240, 2: 260, 3: 300, 4: 380, 5: 456, 6: 528, 7: 600}
    folders = ['P016_balloon1_side', 'P016_balloon1_top', 'P016_balloon2_side', 'P016_balloon2_top',
               'P016_tissue1_side', 'P016_tissue1_top', 'P016_tissue2_side', 'P016_tissue2_top', 'P017_balloon1_side',
               'P017_balloon1_top', 'P017_balloon2_side', 'P017_balloon2_top', 'P017_tissue1_side', 'P017_tissue1_top',
               'P017_tissue2_side', 'P017_tissue2_top', 'P018_balloon1_side', 'P018_balloon1_top', 'P018_balloon2_side',
               'P018_balloon2_top', 'P018_tissue1_side', 'P018_tissue1_top', 'P018_tissue2_side', 'P018_tissue2_top',
               'P019_balloon1_side', 'P019_balloon1_top', 'P019_balloon2_side', 'P019_balloon2_top',
               'P019_tissue1_side', 'P019_tissue1_top', 'P019_tissue2_side', 'P019_tissue2_top', 'P020_balloon1_side',
               'P020_balloon1_top', 'P020_balloon2_side', 'P020_balloon2_top', 'P020_tissue1_side', 'P020_tissue1_top',
               'P020_tissue2_side', 'P020_tissue2_top', 'P021_balloon1_side', 'P021_balloon1_top', 'P021_balloon2_side',
               'P021_balloon2_top', 'P021_tissue1_side', 'P021_tissue1_top', 'P021_tissue2_side', 'P021_tissue2_top',
               'P022_balloon1_side', 'P022_balloon1_top', 'P022_balloon2_side', 'P022_balloon2_top',
               'P022_tissue1_side', 'P022_tissue1_top', 'P022_tissue2_side', 'P022_tissue2_top', 'P023_balloon1_side',
               'P023_balloon1_top', 'P023_balloon2_side', 'P023_balloon2_top', 'P023_tissue1_side', 'P023_tissue1_top',
               'P023_tissue2_side', 'P023_tissue2_top', 'P024_balloon1_side', 'P024_balloon1_top', 'P024_balloon2_side',
               'P024_balloon2_top', 'P024_tissue1_side', 'P024_tissue1_top', 'P024_tissue2_side', 'P024_tissue2_top',
               'P025_balloon1_side', 'P025_balloon1_top', 'P025_balloon2_side', 'P025_balloon2_top',
               'P025_tissue1_side', 'P025_tissue1_top', 'P025_tissue2_side', 'P025_tissue2_top', 'P026_balloon1_side',
               'P026_balloon1_top', 'P026_balloon2_side', 'P026_balloon2_top', 'P026_tissue1_side', 'P026_tissue1_top',
               'P026_tissue2_side', 'P026_tissue2_top', 'P027_balloon1_side', 'P027_balloon1_top', 'P027_balloon2_side',
               'P027_balloon2_top', 'P027_tissue1_side', 'P027_tissue1_top', 'P027_tissue2_side', 'P027_tissue2_top',
               'P028_balloon1_side', 'P028_balloon1_top', 'P028_balloon2_side', 'P028_balloon2_top',
               'P028_tissue1_side', 'P028_tissue1_top', 'P028_tissue2_side', 'P028_tissue2_top', 'P029_balloon1_side',
               'P029_balloon1_top', 'P029_balloon2_side', 'P029_balloon2_top', 'P029_tissue1_side', 'P029_tissue1_top',
               'P029_tissue2_side', 'P029_tissue2_top', 'P030_balloon1_side', 'P030_balloon1_top', 'P030_balloon2_side',
               'P030_balloon2_top', 'P030_tissue1_side', 'P030_tissue1_top', 'P030_tissue2_side', 'P030_tissue2_top',
               'P031_balloon1_side', 'P031_balloon1_top', 'P031_balloon2_side', 'P031_balloon2_top',
               'P031_tissue1_side', 'P031_tissue1_top', 'P031_tissue2_side', 'P031_tissue2_top', 'P032_balloon1_side',
               'P032_balloon1_top', 'P032_balloon2_side', 'P032_balloon2_top', 'P032_tissue1_side', 'P032_tissue1_top',
               'P032_tissue2_side', 'P032_tissue2_top', 'P033_balloon1_side', 'P033_balloon1_top', 'P033_balloon2_side',
               'P033_balloon2_top', 'P033_tissue1_side', 'P033_tissue1_top', 'P033_tissue2_side', 'P033_tissue2_top',
               'P034_balloon1_side', 'P034_balloon1_top', 'P034_balloon2_side', 'P034_balloon2_top',
               'P034_tissue1_side', 'P034_tissue1_top', 'P034_tissue2_side', 'P034_tissue2_top', 'P035_balloon1_side',
               'P035_balloon1_top', 'P035_balloon2_side', 'P035_balloon2_top', 'P035_tissue1_side', 'P035_tissue1_top',
               'P035_tissue2_side', 'P035_tissue2_top', 'P036_balloon1_side', 'P036_balloon1_top', 'P036_balloon2_side',
               'P036_balloon2_top', 'P036_tissue1_side', 'P036_tissue1_top', 'P036_tissue2_side', 'P036_tissue2_top',
               'P037_balloon1_side', 'P037_balloon1_top', 'P037_balloon2_side', 'P037_balloon2_top',
               'P037_tissue1_side', 'P037_tissue1_top', 'P037_tissue2_side', 'P037_tissue2_top', 'P038_balloon1_side',
               'P038_balloon1_top', 'P038_balloon2_side', 'P038_balloon2_top', 'P038_tissue1_side', 'P038_tissue1_top',
               'P038_tissue2_side', 'P038_tissue2_top', 'P039_balloon1_side', 'P039_balloon1_top', 'P039_balloon2_side',
               'P039_balloon2_top', 'P039_tissue1_side', 'P039_tissue1_top', 'P039_tissue2_side', 'P039_tissue2_top',
               'P040_balloon1_side', 'P040_balloon1_top', 'P040_balloon2_side', 'P040_balloon2_top',
               'P040_tissue1_side', 'P040_tissue1_top', 'P040_tissue2_side', 'P040_tissue2_top']
    # folders = ['P016_balloon1_side', 'P016_balloon1_top', 'P016_balloon2_side', 'P016_balloon2_top']

    side_folders = []
    up_folders = []
    for fo in folders:
        if 'side' in fo:
            side_folders.append(fo)
        elif 'top' in fo:
            up_folders.append(fo)

    print(side_folders)
    exit()
    ordered_folders = side_folders + up_folders
    classes_index_not_to_load = []

    labels_folder_dict = {i: folders[i] for i in range(len(ordered_folders))}

    # db_address = '/datashare/APAS/frames/'
    db_address = '/home/user/datasets/frames'
    batch_size = 256
    seed = 100
    for model_index, model_input_size in model_params_dict.items():
        if model_index == 0:
            print(f"Starting model: {model_index}")
            ds = tf.keras.preprocessing.image_dataset_from_directory(
                db_address,
                seed=seed,
                labels='inferred',
                label_mode='int',
                class_names=list(labels_folder_dict.values()),
                shuffle=False,
                image_size=(model_input_size, model_input_size),
                batch_size=batch_size)

            dest_path = f'{os.getcwd()}/efficientnet/B_exclude{model_index}'
            os.makedirs(dest_path, exist_ok=True)
            generate_efficientnet_feature_vectors(ds=ds,
                                                  dest_path=dest_path,
                                                  labels_folder_dict=labels_folder_dict,
                                                  model_index=model_index,
                                                  model_input_size=model_input_size,
                                                  batch_size=batch_size)


if __name__ == '__main__':
    print("updated 5")
    main()
