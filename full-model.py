import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


def data_generator(x, image_format, image_s=(512, 512), batch_s=1, type_gen="train",
                   path=None, col_label="label", col_path="path", save_patches_path="./", save_patches=False,
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
            rotation_range=360,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.2, 1.5],
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
                              "training_accuracy": history.history['binary_accuracy'],
                              "n_epochs": list(range(0, epochs))})
    else:
        table = pd.DataFrame({
            "training_loss": history.history['loss'],
            "training_accuracy": history.history["categorical_accuracy"],
            "n_epochs": list(range(0, epochs))})
    return table


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
    parser.add_argument("--case", type=int, default=2,
                        help="Number of classes. default=2")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="learning rate. default=1e-4")
    parser.add_argument("--train_table", default="/data/Train.csv",
                        help="path to train table'")
    parser.add_argument("--data_shape", type=int, default=512,
                        help="shape")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch_size")
    parser.add_argument("--image_format", type=str, default="jpeg",
                        help="format of input image")
    parser.add_argument("--drop_out", default=0.2, type=float,
                        help="drop out rate default 0.2")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="select model, resent50，resnet101 or inception")
    parser.add_argument("--label_col", type=str, default="Outcome",
                        help="colname of your label in the metafile")
    parser.add_argument("--file_col", type=str, default="File",
                        help="colname of your filenames in the metafile")
    parser.add_argument("--patient_col", type=str, default="Patient",
                        help="colname of patients in the metafile")
    parser.add_argument("--entropy", type=str, default="binary",
                        help="type of entropy, binary or categorical")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="whether to use imagenet pretrained model")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="where to save output results")
    parser.add_argument("--save_augmented_patches", default=False, action='store_true',
                        help="save augmented patches or not")
    parser.add_argument("--model_savename", type=str, default=None,
                        help="name to save for your trained model")
    parser.add_argument("--sample", "-sample", type=str, default=None,
                        help="enter how many tiles you want to capture maximum from each split label")
    parser.add_argument("--early_stopping", default=False, action='store_true',
                        help="early_stopping or not")
    parser.add_argument("--strategy", type=str, default="mirrored",
                        help="Distribution strategy (whether to parallelise across GPUs")
    args = parser.parse_args()
    ########
    # input parameters
    train_table = args.train_table
    input_path = args.input
    # image parameters
    image_format = args.image_format
    data_shape = int(args.data_shape)
    image_shape = (data_shape, data_shape, 3)
    # model parameters
    model = args.model
    batch_size = args.batch_size
    epoch = args.epoch
    drop_out = args.drop_out
    lr = args.learning_rate
    entropy = args.entropy
    pretrained_model = args.pretrained
    # number of classes
    n_class = args.case
    # balance your sample or not
    cap_sample = args.sample
    # columns in metadata file
    label_col = args.label_col
    file_col = args.file_col
    patient_col = args.patient_col
    # output parameters
    result_dir = args.result_dir
    save_augmented_patch = args.save_augmented_patches
    model_name = args.model_savename
    distribution_strategy = args.strategy

    print("Start")

    # Load your data into a DataFrame (assuming it's in a CSV file)
    meta_data = pd.read_csv(train_table)

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

    # allows parallel processing on multiple GPUs
    if distribution_strategy.lower() == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise Exception("Please choose a training strategy")
    with strategy.scope():
        # set model architecture
        if pretrained_model:
            weights = 'imagenet'
        else:
            weights = None
        if model.lower() == "resnet101":
            base_model = tf.keras.applications.resnet.ResNet101(include_top=False, weights=weights,
                                                                input_shape=image_shape, pooling=None)
        elif model.lower() == "resnet50":
            base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=weights,
                                                               input_shape=image_shape, pooling=None)
        elif model.lower() == "inception":
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights=weights,
                                                           input_shape=image_shape, pooling=None)
        else:
            raise Exception("Please enter resnet101 or resnet50 to train")

        inputs = tf.keras.Input(shape=image_shape)

        m = base_model(inputs)

        # GlobalAveragePooling2D layer was used to replace flatten layer to avoid overfitting
        m = tf.keras.layers.GlobalAveragePooling2D()(m)

        m = Dropout(drop_out)(m)

        if n_class > 2:
            outputs = tf.keras.layers.Dense(n_class, activation="softmax")(m)
            loss_fn = losses.CategoricalCrossentropy(from_logits=False)
            metrics_type = metrics.CategoricalAccuracy()
            print("soft max activation is used in dense layer, CategoricalCrossentropy is used as loss function")
        else:
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(m)
            loss_fn = losses.BinaryCrossentropy(from_logits=False)
            # metrics_type = metrics.BinaryAccuracy() used in V1.2
            print("sigmoid activation is used in dense layer, BinaryCrossentropy is used as loss function")

        final = tf.keras.Model(inputs, outputs)
        # compile model
        final.compile(optimizer=optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=0.01), loss=loss_fn,
                      metrics=["accuracy"])  # v1.2 = metrics_type

    # set call back seed
    np.random.seed(100)

    with strategy.scope():
        train_generator = data_generator(meta_data, image_format=image_format, image_s=(data_shape, data_shape),
                                         batch_s=batch_size,
                                         type_gen="train", path=input_path, col_label=label_col, col_path=file_col,
                                         save_patches=save_augmented_patch, save_patches_path=result_dir)

        print("train_generator:" + str(train_generator.class_indices))

        save_model_path = os.path.join(result_dir, model_name + ".keras")

        # learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3, min_lr=1e-7)

        # Configure the ModelCheckpoint callback to monitor training accuracy
        save_model_callback = ModelCheckpoint(filepath=save_model_path,
                                              monitor='accuracy',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='max')
        # Use the callback during training
        hist_final = final.fit(train_generator, epochs=epoch, callbacks=[lr_scheduler, save_model_callback],
                               verbose=1)

        print("Trained model saved at {}".format(result_dir))

        accuracy = hist_final.history["accuracy"][-1]
        print("Final accuracy: {}".format(accuracy))
        out_tab = table_training_process(hist_final, entropy)

        out_tab.to_csv(os.path.join(result_dir, "Training_accuracy.csv"))


if __name__ == "__main__":
    main()
