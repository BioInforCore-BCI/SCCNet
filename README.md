# Enhanced Metastasis Risk Prediction in Cutaneous Squamous Cell Carcinoma Using Deep Learning and Computational Histopathology 

Emilia Peleva*, Yue Chen*, Bernhard Finke, Hasan Rizvi, Eugene Healy, Chester Lai, Paul Craig, William Rickaby, Christina Schoenherr, Craig Nourse, Charlotte Proby, Gareth J Inman, Irene M. Leigh, Catherine A. Harwood, Jun Wang

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

The tiles folder contains 7 tiles. Norm_tile.jpg is the template we used for colour normalization. 6 other sample tiles (before colour normalisation) from our testing cohort are also given for inspection by users: non_met{1-3...}jpg are tiles from primary cSCC which did not metastasize, while met{1-3...}.jpg are tiles from cSCC which metastasized.

This folder also contains the training table samples_table.csv, which is required for training models and prediction, here 'Outcome' refers to metastasis, and 'ROI' refers to whether a tile was inside or outside our annotated region of interest.

# Replicating the pipeline

To run the entire pipeline on your own dataset, first ensure you have followed the installation instructions above and saved your WSI tiles in a folder named 'tiles'. WSI tile names should take the form {wsi_filename}\_x\_{x_coord}\_y\_{y_coord}.jpg. This folder should also contain a training table with the same columns as the provided 'samples_table.csv', and a tile used as the standard for colour normalization, named Norm_tile.jpg. The user can either use the provided tile or select their own. Then, run the following commands in a command prompt or terminal, in the order shown:


1. pre-processing: Performs colour normalisation using the Macenko method.

       python pre-processing.py \
              --patch_dir /tiles \
              --save_dir /tiles_norm \
              --colour_standard /tiles/Norm_tile.jpg
   
2. hypertuning: Uses the Keras tuner to compare various model architectures and hyperparameters.

       python hypertuning.py \
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
   
3. k-fold: Performs 5-fold cross validation. This snippet of code assumes the user has obtained the same results from the hypertuning step for optimal learning rate, dropout value, and model architecture using their own data. If this is not the case, the user should substitute other values in these fields.
   
       python k-fold.py \
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
   
4. full-model: For final model training on the entire training cohort.

       python full-model.py \
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
              --model_savename full_model.h5 \
              --sample 1000 \
              --early_stopping

Steps 5 and 6 of the code are concerned with generation of predictions and heatmaps from the model trained in step 4. The following code snippets assume that the user has saved a full model using step 4. If instead the user would like to generate predictions/ heatmaps using the provided model weights see subsection 'Generating predictions using provided weights'.
   
5. predictions: Generates tile-level predictions using the trained model.

        python predictions.py \
              --img_path /tiles_norm \
              --process_list /tiles/samples_table.csv \
              --model /results/full_model.h5 \
              --img_shape 512 \
              --batch_size 32 \
              --save_dir /results \
              --save_name predictions \
              --filename_col File



   
6. heatmaps: Generates heatmaps based on tile-level predictions. This assumes the user has their original WSIs saved in a folder called wsi_files, and they are saved as .svs files.

       python generate_heatmaps.py \
              --process_list /results/predictions_process_list.csv \
              --wsi_dir /wsi_files \
              --score_dir /results/score_files \
              --wsi_format svs
              --heatmap_save_dir /results/heatmaps \
              --patch_size 512 \
              --heatmap_mode percentiles \
              --alpha 0.5 \
              --blur 

## Generating predictions using provided weights

This code snippet provides the same functionality as that in step 5 above, except instead of generating predictions from a full saved model generated in step 4, it loads a model from the saved weights provided (in this example from model 2 for metastasis risk prediction).

        python predictions.py \
            --img_path /tiles_norm \
            --process_list /tiles/samples_table.csv \
            --model /models/risk_model2_weights.h5 \
            --img_shape 512 \
            --batch_size 32 \
            --save_dir /results \
            --save_name predictions \
            --filename_col File \
            --weights_only






# Load a trained model

To load a pre-trained model and integrate it into your own code, simply download your preferred model from the models folder and insert the below code snippet into your .py file, switching 'model_name' with the correct filename for your model.

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


