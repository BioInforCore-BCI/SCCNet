import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import StratifiedGroupKFold


def data_generator(x, image_format, image_s=(512, 512), batch_s=1, type_gen="train",
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
                        help="select model, resent50 or resnet101")
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
    parser.add_argument("--num_folds", type=int, default=5,
                        help="number of folds for k-fold cross-validation")
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
    learning_rate = args.learning_rate
    entropy = args.entropy
    pretrained_model = args.pretrained
    earlystop = args.early_stopping
    # Define the number of splits
    num_folds = args.num_folds
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
    # Add column to meta table containing hospital code (extracted from patient column)
    meta_data['Hospital'] = meta_data[patient_col].astype(str).str[:2]

    # Add a cap to the amount of samples from a specific patient
    if str(cap_sample).isdigit():
        cap_sample = int(cap_sample)
        frames = list()
        for patient in meta_data[patient_col].unique():
            if len(meta_data[meta_data[patient_col] == patient]) >= cap_sample:
                frame = meta_data[meta_data[patient_col] == patient].sample(n=cap_sample,
                                                                            replace=False, random_state=40)
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
            metrics_type = metrics.BinaryAccuracy()
            print("sigmoid activation is used in dense layer, BinaryCrossentropy is used as loss function")

        final = tf.keras.Model(inputs, outputs)
        # compile model
        final.compile(optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01), loss=loss_fn,
                      metrics=[metrics_type])

    # set call back seed

    np.random.seed(88)

    # Variables for k-fold cross-validation
    x = meta_data[file_col]
    y = meta_data[label_col]
    y_concat_hosp = y.astype(str) + '_' + meta_data['Hospital']
    groups = meta_data[patient_col]

    # Initialize variables to store evaluation results
    all_accuracies = []
    all_train_tables = []

    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=88)

    with strategy.scope():
        # Perform the custom stratified k-fold split
        for fold, (train_index, val_index) in enumerate(sgkf.split(x, y_concat_hosp, groups)):
            print(f"Fold {fold + 1}/{num_folds}:")
            print(f"Train set size: {len(x.iloc[train_index])}, Test set size: {len(x.iloc[val_index])}")
            print(f"Train set patients: {groups.iloc[train_index].unique()}")
            print(f"Test set patients: {groups.iloc[val_index].unique()}")
            print(f"Train set outcomes: {y.iloc[train_index].unique()}")
            print(f"Test set outcomes: {y.iloc[val_index].unique()}")
            print('-' * 30)

            # Split the data into training and validation sets for this fold
            train_fold, val_fold = meta_data.iloc[train_index], meta_data.iloc[val_index]

            train_generator = data_generator(train_fold, image_format=image_format, image_s=(data_shape, data_shape),
                                             batch_s=batch_size,
                                             type_gen="train", path=input_path, col_label=label_col, col_path=file_col,
                                             save_patches=save_augmented_patch, save_patches_path=result_dir)

            validation_generator = data_generator(val_fold, image_format=image_format, image_s=(data_shape, data_shape),
                                                  batch_s=batch_size, type_gen="validation", path=input_path,
                                                  col_label=label_col, col_path=file_col, save_patches=False)

            # it was time_now[12:len(time_now)]
            nam = str(model_name) + "_fold" + str(fold + 1) + "_epoch" + str(epoch) + "_LR_" + str(
                learning_rate) + "_DROP_" + str(drop_out) + "_BS_" + str(batch_size) + ".keras"

            model_dir = os.path.join(result_dir, nam)

            # saves models
            check = ModelCheckpoint(filepath=model_dir, verbose=1, save_best_only=True)

            # learning rate scheduler
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3, min_lr=1e-7)

            if earlystop:
                early = EarlyStopping(monitor="val_binary_accuracy", mode="max", patience=5, restore_best_weights=True)
                hist_final = final.fit(train_generator, epochs=epoch, validation_data=validation_generator,
                                       callbacks=[check, early, lr_scheduler], verbose=1)
            else:
                hist_final = final.fit(train_generator, epochs=epoch, validation_data=validation_generator,
                                       callbacks=[check, lr_scheduler], verbose=1)

            print("Trained model saved at {}".format(nam))

            # Store all evaluation metrics for this fold
            accuracy = hist_final.history["val_binary_accuracy"][-1]
            all_accuracies.append(accuracy)

            out_tab = table_training_process(hist_final, entropy)
            all_train_tables.append(out_tab)

            out_tab.to_csv(os.path.join(result_dir, nam[:-6]) + "_accuracy.csv")

        # Check if the dataframes have the same shape
        shape_set = set(df.shape for df in all_train_tables)
        if len(shape_set) > 1:
            print("Training tables have different shapes. Could not average across folds")
        else:
            # Average train table across folds
            tables_concat = pd.concat(all_train_tables, axis=1)
            tables_mean = tables_concat.T.groupby(level=0).mean().T.astype({'n_epochs': 'int'})

            tables_mean.to_csv(os.path.join(result_dir, nam[:-6]) + "_accuracy.csv")

        # Calculate the average accuracy across all folds
        average_accuracy = np.mean(all_accuracies)
        print(f"Average Accuracy Across Folds: {average_accuracy:.4f}")


if __name__ == "__main__":
    main()
