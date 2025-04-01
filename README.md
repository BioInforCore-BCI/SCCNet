# cSCCNet: Enhanced Metastasis Risk Prediction in Cutaneous Squamous Cell Carcinoma Using Deep Learning and Computational Histopathology by cSCCNet

To replicate the workflow of the paper run the python files in the following order:

1. pre-processing
2. hypertuning
3. k-fold
4. full-model
5. predictions
6. full-model


# Define the model architecture
def create_model():
    input_layer = Input(shape=(512, 512, 3))  # Input layer

    # Load ResNet50 as backbone (excluding top layers)
    resnet_base = ResNet50(weights=None, include_top=False, input_tensor=input_layer)

    x = GlobalAveragePooling2D()(resnet_base.output)  # Global Average Pooling
    x = Dropout(0.5)(x)  # Dropout for regularization
    output_layer = Dense(1, activation='sigmoid')(x)  # Final classification layer

    model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")

    return model

    new_model.load_weights('risk_model2_weights.h5', by_name=True)
