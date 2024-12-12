from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import argparse


def load_model_4_prediction(file="name.h5", features=True):
    """
    Function to load saved model for class prediction of feature extraction.
    :param file:
    :param features:
    :return:
    """
    mod = load_model(file)
    # Check the type of prediction required
    if features:
        # Get the Feature layer flattened
        mod = Model(inputs=mod.input, outputs=mod.get_layer("Features").output)
    mod.summary()
    return mod


def data_generator(x, path, col_path, image_s=(256, 256), batch_s=1, shuf=True):
    """
    Image generator for feeding an CNN built with Keras.
    It flows files from pandas data frame.
    :param x: Data frame containing the image file paths and class labels. # csv file?
    :param image_s: The size of images to return, rescaled, i.e (width, height).
    :param batch_s: The number of images per batch of the generator.
    :param col_path: The column name of x with the path of the image files or half path with path provided below.
    :param shuf: Shuffle the data.
    :param path: The path of the parent folder if, the col_path is not a full path.
    :return: It returns an ImageDataGenerator class, as defined by Keras
    """
    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=x,
        directory=path,
        validate_filenames=True,
        x_col=col_path,
        y_col=None,
        target_size=image_s,
        class_mode=None,
        save_format="png",
        batch_size=batch_s,
        shuffle=shuf)
    return test_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prediction pathes')
    parser.add_argument('--img_path', type=str,
                        help='path to folder containing image pathes')
    parser.add_argument('--process_list', type=str, default=None,
                        help='name of list of images to predict (.csv)')
    parser.add_argument('--model', type=str,
                        help='path to model for prediction')
    parser.add_argument('--img_shape', type=int, default=256,
                        help='img_shape please enter single value such as 128,256 or 512 etc.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save results')
    parser.add_argument('--save_name', type=str, default="prediction",
                        help='result_name_to_save')
    parser.add_argument('--filename_col', type=str, default="Name",
                        help='colname of filenames in metatable')

    args = parser.parse_args()
    img_path = args.img_path
    context_file = args.process_list
    batch_size = args.batch_size
    img_shape = (args.img_shape, args.img_shape)
    result_dir = args.save_dir
    nam = args.model
    save_name = args.save_name
    col_name = args.filename_col

    os.makedirs(result_dir, exist_ok=True)
    # Test generator
    context_file = pd.read_csv(context_file)

    predict_generator = data_generator(x=context_file, image_s=img_shape,
                                       batch_s=batch_size,
                                       path=img_path,
                                       col_path=col_name, shuf=False)

    # Load trained models
    mod_class = load_model_4_prediction(file=nam, features=False)

    # Softmax
    pre_clas = pd.DataFrame(mod_class.predict(predict_generator))

    pre_clas = pre_clas.rename(columns={0: 'Score'})  # this works
    pre_clas = pre_clas.rename(columns={"0": 'Score'})
    pre_clas["Filenames"] = predict_generator.filenames
    # extract SampleID and clean it
    pre_clas['SampleID'] = pre_clas["Filenames"].str.split("_x_").str.get(0)
    pre_clas['SampleID'] = pre_clas['SampleID'].str.replace(" ", "_")
    pre_clas['SampleID'] = pre_clas['SampleID'].str.replace("-", "_")
    pre_clas['SampleID'] = pre_clas['SampleID'].str.replace("__", "_")
    # extract coords
    pre_clas[['coord_x', 'coord_y']] = pre_clas['Filenames'].str.extract(r'_x_(\d+)_y_(\d+)_')
    # pre_clas.columns = ['Score', 'Filenames', 'SampleID', 'coord_x', 'coord_y'] no longer needed
    print("Final column names")
    print(pre_clas.columns)
    # save prediction results together
    pre_clas.to_csv(os.path.join(result_dir, args.save_name + ".csv"), index=False)
    # save process list for heatmap
    process_list = pd.DataFrame({'SampleID': pre_clas['SampleID'].unique()})
    process_list.to_csv(os.path.join(result_dir, args.save_name + "_process_list.csv"), index=False)

    # save score for each sampleID
    score_dir = os.path.join(result_dir, "score_files")
    os.makedirs(score_dir, exist_ok=True)
    for ID in pre_clas['SampleID'].unique():
        tmp = pre_clas[pre_clas['SampleID'] == str(ID)]
        tmp.to_csv(os.path.join(score_dir, ID + "_score_file.csv"), index=False)
