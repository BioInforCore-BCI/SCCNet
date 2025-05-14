# Enhanced Metastasis Risk Prediction in Cutaneous Squamous Cell Carcinoma Using Deep Learning and Computational Histopathology 

Emilia Peleva*, Yue Chen*, Bernhard Finke, Hasan Rizvi, Eugene Healy, Chester Lai, Paul Craig, William Rickaby, Christina Schoenherr, Craig Nourse, Charlotte Proby, Gareth J Inman, Irene M. Leigh, Catherine A. Harwood, Jun Wang

To replicate the workflow of the paper run the python files in the following order:

1. pre-processing: Performs colour normalisation using the Macenko method.
2. hypertuning: Uses the Keras tuner to compare various model architectures and hyperparameters.
3. k-fold: Performs 5-fold cross validation. 
4. full-model: For final model training on the entire training cohort.
5. predictions: Generates tile-level predictions using the trained model.
6. heatmaps: Generates heatmaps based on tile-level predictions. 


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

# Sample tiles

The folder tiles/ contains 7 tiles. Norm_tile.jpg is the tile we used for normalization. 6 other sample tiles from our test set are also given for inspection by users. non_met{1-3}.jpg are tiles from patients who did not metastasize, while met{1-3}.jpg are tiles from patients who did experience metastasis.
