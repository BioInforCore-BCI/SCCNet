# Enhanced Metastasis Risk Prediction in Cutaneous Squamous Cell Carcinoma Using Deep Learning and Computational Histopathology 

Emilia Peleva*, Yue Chen*, Bernhard Finke, Hasan Rizvi, Eugene Healy, Chester Lai, Paul Craig, William Rickaby, Christina Schoenherr, Craig Nourse, Charlotte Proby, Gareth J Inman, Irene M. Leigh, Catherine A. Harwood, Jun Wang

To replicate the workflow of the paper run the python files in the following order:

1. pre-processing: Performs colour normalisation using the Macenko method.
2. hypertuning: Uses the Keras tuner to compare various model architectures and hyperparameters.
3. k-fold: Performs 5-fold cross validation. 
4. full-model: For final model training on the entire training cohort.
5. predictions: Generates tile-level predictions using the trained model.
6. heatmaps: Generates heatmaps based on tile-level predictions.

The utils file contains some additional functions and classes which are required to run (1) and (6).

# Installation instructions

The code was originally developed in Python version 3.9. The file requirements.txt contains the minimal packages required to run the code.
The code was tested in Linux (Rocky Linux 9.4 (Blue Onyx)) and MacOS (14.7.6).

To get started with the code, simply navigate into a new directory, and run:

    git clone https://github.com/BioInforCore-BCI/SCCNet.git
    cd SCCNet
    
    python3 -m venv scc_env
    source scc_env/bin/activate
    pip install -r requirements.txt

# Sample tiles

The tiles folder contains 7 tiles. Norm_tile.jpg is the template we used for colour normalization. 6 other sample tiles (before colour normalisation) from our testing cohort are also given for inspection by users: non_met{1-3}.jpg are tiles from primary cSCC which did not metastasize, while met{1-3}.jpg are tiles from cSCC which metastasized.

This folder also contains the training table samples_table.csv, which is required for training models and prediction, here 'Outcome' refers to metastasis, and 'ROI' refers to whether a tile was inside or outside our annotated region of interest.

# Detailed instructions

To run the entire pipeline on your own dataset, first ensure you have followed the installation instructions above and saved your WSI tiles in a folder named 'tiles'. This folder should also contain a training table with the same columns as the provided 'samples_table.csv', and a tile used as the standard for colour normalization, named Norm_tile.jpg. The user can either use the provided tile or select their own. Finally, there should also exist a folder named 'results' in the current directory. Then, run the following commands in a command prompt or terminal, in the order shown:

    1. python pre-processing.py \
          --patch_dir /tiles \
          --save_dir /tiles_norm \
          --colour_standard /tiles/Norm_tile.jpg

    2. python hypertuning.py \
          --input /tiles_norm \
          --epoch 20 \
          --train_table /tiles/samples_table.csv \
          --batch_size 32 \
          --image_format jpeg \
          --label_col Outcome \
          --file_col File \
          --patient_col Patient \
          --sample 1000 \
          --patience 5 \
          --save_folder tuner_output

    3. python k-fold.py \
          --input \tiles_norm \
          --epoch 20 \
          --learning_rate 0.0001 \
          --train_table /tiles/samples_table.csv \
          --data_shape 512 \
          --batch_size 32 \
          --image_format jpeg \
          --drop_out 0.2 \
          --model resnet50 \
          --label_col Outcome \
          --file_col File \
          --patient_col Patient \
          --pretrained \
          --result_dir /results \
          --model_savename kfold_model_name.h5 \
          --num_folds 5 \
          --sample 1000 \
          --early_stopping

    4. python full-model.py \
          --input \tiles_norm \
          --epoch 40 \
          --learning_rate 0.0001 \
          --train_table /tiles/samples_table.csv \
          --data_shape 512 \
          --batch_size 32 \
          --image_format jpeg \
          --drop_out 0.2 \
          --model resnet50 \
          --label_col Outcome \
          --file_col File \
          --patient_col Patient \
          --pretrained \
          --result_dir /results \
          --model_savename full_model_name.h5 \
          --sample 1000 \
          --early_stopping

    5. python predictions.py \
          --img_path /tiles_orm \
          --process_list ./process_list.csv \
          --model /results/full_model_name.h5 \
          --img_shape 256 \
          --batch_size 32 \
          --save_dir /results \
          --save_name predictions \
          --filename_col File




# Load a trained model

To load a pre-trained model to predict or fine-tune on your own dataset simply download your preferred model from the models folder and run the below code snippet, switching 'model_name' with the correct filename for your model.

    from tensorflow import keras
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
    
    def create_model():
        input_layer = Input(shape=(512, 512, 3))  # Input layer
    
        # Load ResNet50 as backbone (excluding top layers)
        resnet_base = ResNet50(weights=None, include_top=False, input_tensor=input_layer)
    
        x = GlobalAveragePooling2D()(resnet_base.output)  # Global Average Pooling
        x = Dropout(0.5)(x)  # Dropout for regularization
        output_layer = Dense(1, activation='sigmoid')(x)  # Final classification layer
    
        model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")
    
        return model

    new_model = create_model()
    new_model.load_weights('model_name', by_name=True)




