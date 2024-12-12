import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import keras_tuner
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split


def data_generator(x, image_format, image_s=(256, 256), batch_s=1, type_gen="train",
                   path=None, col_label="label", col_path="path", save_patches_path=None, save_patches=False,
                   shuf=True):
    """
    Image generator for feeding an CNN built with Keras.
    It flows files from pandas data frame.
    :param image_format: format of input images
    :param x: Data frame containing the image file paths and class labels. # csv file?
    :param image_s: The size of images to return, rescaled, i.e (width, height).
    :param batch_s: The number of images per batch of the generator.
    :param type_gen: Type of data been generated, "train", "validation" or prediction.
    :param col_path: The column name of x with the path of the image files or half path with path provided below.
    :param col_label: The column name of x with the class labels.
    :param shuf: Shuffle the data.
    :param path: The path of the parent folder if, the col_path is not a full path.
    :param save_patches: whether to save augmented patches?
    :param save_patches_path: path to save your data augmented patches if save_patches is True
    :param model: which model architecture to use for preprocessing
    :return: It returns an ImageDataGenerator class, as defined by Keras
    """
    if type_gen == "train":
        # This is set up this way for feature updating with argumentation
        data_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.2, 1.0],
            fill_mode='reflect')
    elif type_gen == "validation" or type_gen == "prediction":
        data_gen = ImageDataGenerator(
            rescale=1.0 / 255)
    else:
        sys.exit("Please, provide the correct type_gen [train or validation] as string")

    if save_patches is not False:
        path_save = os.path.join(save_patches_path, "augmented_patches")
        os.makedirs(path_save, exist_ok=True)

        if type_gen == "train" or type_gen == "validation":
            d_generator = data_gen.flow_from_dataframe(
                dataframe=x,
                shuffle=shuf,
                directory=path,
                validate_filenames=True,
                x_col=col_path,
                y_col=col_label,
                target_size=image_s,
                class_mode="binary",
                save_format=image_format,
                batch_size=batch_s,
                save_to_dir=path_save,
                save_prefix='patches_')
        else:
            d_generator = data_gen.flow_from_dataframe(
                dataframe=x,
                directory=path,
                validate_filenames=True,
                x_col=col_path,
                y_col=None,
                target_size=image_s,
                class_mode=None,
                save_format=image_format,
                batch_size=batch_s,
                shuffle=shuf)
    else:
        if type_gen == "train" or type_gen == "validation":
            d_generator = data_gen.flow_from_dataframe(
                dataframe=x,
                shuffle=shuf,
                directory=path,
                validate_filenames=True,
                x_col=col_path,
                y_col=col_label,
                target_size=image_s,
                class_mode="binary",
                save_format=image_format,
                batch_size=batch_s)
        else:
            d_generator = data_gen.flow_from_dataframe(
                dataframe=x,
                directory=path,
                validate_filenames=True,
                x_col=col_path,
                y_col=None,
                target_size=image_s,
                class_mode=None,
                save_format=image_format,
                batch_size=batch_s,
                shuffle=shuf)
    return d_generator


def table_training_process(history, entropy):
    """
    Function to create plot data for model training history.
    :param entropy: binary entropy or categorical entropy
    :param history: History object.
    :return:
    """
    epochs = len(history.history['loss'])

    if entropy.lower() == "binary":
        table = pd.DataFrame({"training_loss": history.history['loss'],
                              "validation_loss": history.history['val_loss'],
                              "training_accuracy": history.history["binary_accuracy"],
                              "validation_accuracy": history.history["val_binary_accuracy"],
                              "n_epochs": list(range(0, epochs))})
    else:
        table = pd.DataFrame({
            "training_loss": history.history['loss'],
            "validation_loss": history.history['val_loss'],
            "training_accuracy": history.history["categorical_accuracy"],
            "validation_accuracy": history.history["val_categorical_accuracy"],
            "n_epochs": list(range(0, epochs))})
    return table


def build_model(hp):
    image_shape = (128, 128, 3)

    pretrained_network = hp.Choice('pretrained_network', ['ResNet50', 'MobileNetV2', 'VGG16', 'InceptionV3'])
    if pretrained_network == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=image_shape)
    elif pretrained_network == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=image_shape)
    elif pretrained_network == 'VGG16':
        base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
    elif pretrained_network == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=image_shape)

    inputs = tf.keras.Input(shape=image_shape)

    m = base_model(inputs)

    # GlobalAveragePooling2D layer was used to replace flatten layer to avoid overfitting
    m = tf.keras.layers.GlobalAveragePooling2D()(m)

    m = Dropout(hp.Float('dropout', min_value=0.1, max_value=0.2, step=0.05))(m)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(m)
    loss_fn = losses.BinaryCrossentropy(from_logits=False)
    metrics_type = metrics.BinaryAccuracy()
    print("sigmoid activation is used in dense layer, BinaryCrossentropy is used as loss function")

    final = tf.keras.Model(inputs, outputs)
    # compile model
    final.compile(optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]),
                                            beta_1=0.9, beta_2=0.999, decay=0.0001),
                  loss=loss_fn, metrics=[metrics_type])

    return final




def main():
    """
    to add parameters to parser using argparse package
    :return:
    """
    parser = argparse.ArgumentParser(description='This script is train your model')
    parser.add_argument("--input", default="./input",
                        help="Directory name where the input image is saved. default='./input'")
    parser.add_argument("--epoch", type=int, default=20,
                        help="Number of epoch. default=20")
    parser.add_argument("--train_table", default="/data/BCI-DigitalPath/Emilia/QuPath/meta/ROIjpgTrain_short.csv",
                        help="path to train table'")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch_size")
    parser.add_argument("--image_format", type=str, default="jpeg",
                        help="format of input image")
    parser.add_argument("--label_col", type=str, default="Outcome",
                        help="colname of your label in the metafile")
    parser.add_argument("--file_col", type=str, default="File",
                        help="colname of your filenames in the metafile")
    parser.add_argument("--patient_col", type=str, default="Patient",
                        help="colname of patients in the metafile")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="where to save output results")
    parser.add_argument("--save_augmented_patches", default=False, action='store_true',
                        help="save augmented patches or not")
    parser.add_argument("--sample", "-sample", type=str, default=None,
                        help="enter how many tiles you want to capture maximum from each split label")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for val_loss to improve before early stopping")
    parser.add_argument("--strategy", type=str, default="mirrored",
                        help="Distribution strategy (whether to parallelise across GPUs")
    parser.add_argument("--save_folder", type=str, default="tune_met_prediction",
                        help="Folder to save trials in")
    args = parser.parse_args()
    ########
    # input parameters
    train_table = args.train_table
    input_path = args.input
    # image parameters
    image_format = args.image_format
    batch_size = args.batch_size
    # training parameters
    epoch = args.epoch
    patience = args.patience
    # balance your sample or not
    cap_sample = args.sample
    # columns in metadata file
    label_col = args.label_col
    file_col = args.file_col
    patient_col = args.patient_col
    # output parameters
    result_dir = args.result_dir
    save_augmented_patch = args.save_augmented_patches
    distribution_strategy = args.strategy
    save_folder = args.save_folder

    print("Start")

    # Load your data into a DataFrame (assuming it's in a CSV file)
    meta_data = pd.read_csv(train_table)
    # Add column to meta table containing hospital code (extracted from patient column)
    meta_data['Hospital'] = meta_data[patient_col].astype(str).str[:2]

    # Add a cap to the amount of samples from a specific patient
    if str(cap_sample).isdigit():
        cap_sample = int(cap_sample)
        frames = list()
        for patient in meta_data[patient_col].unique():
            if len(meta_data[meta_data[patient_col] == patient]) >= cap_sample:
                frame = meta_data[meta_data[patient_col] == patient].sample(n=cap_sample,
                                                                            replace=False, random_state=42)
            else:
                frame = meta_data[meta_data[patient_col] == patient]
            frames.append(frame)
        meta_data = pd.concat(frames)

    if distribution_strategy.lower() == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise Exception("Please choose a training strategy")

    # set call back seed
    np.random.seed(100)

    groups = meta_data[patient_col]

    with strategy.scope():
        tuner = keras_tuner.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=epoch,
            seed=42,
            project_name=save_folder
        )
        # Split the data into training and validation sets
        train_set, val_set = train_test_split(meta_data, test_size=0.2, random_state=42, stratify=groups)

        train_generator = data_generator(train_set, image_format=image_format, image_s=(128, 128),
                                         batch_s=batch_size,
                                         type_gen="train", path=input_path, col_label=label_col, col_path=file_col,
                                         save_patches=save_augmented_patch, save_patches_path=result_dir)

        validation_generator = data_generator(val_set, image_format=image_format, image_s=(128, 128),
                                              batch_s=batch_size, type_gen="validation", path=input_path,
                                              col_label=label_col, col_path=file_col, save_patches=False)

        early = EarlyStopping(monitor="val_loss", patience=patience)
        tuner.search(train_generator, epochs=epoch, validation_data=validation_generator,
                           callbacks=[early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.values)

        return tuner


if __name__ == "__main__":
    main()
