import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GRU, Masking, concatenate

import tf2onnx


weights_path = "/jet/home/rgupta9/A Organized Home/Anomaly Detection/astromcad/Models/NoHostClassifier_New.h5"



def build_model(latent_size, ntimesteps, num_classes, contextual, n_features=4):
    input_1 = Input((ntimesteps, n_features), name='lc')  # X.shape = (Nobjects, Ntimesteps, 4) CHANGE
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)

    dense1 = Dense(100, activation='tanh')(lstm2)

    if (contextual == 0):
        merge1 = dense1
    else:
        input_2 = Input(shape = (contextual, ), name='host') # CHANGE
        dense2 = Dense(10)(input_2)
        merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent')(dense3)

    output = Dense(num_classes, activation='softmax')(dense4)

    if (contextual == 0):
        model = keras.Model(inputs=input_1, outputs=output)
    else:
        model = keras.Model(inputs=[input_1, input_2], outputs=[output])

    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    return model


def export_onnx(model, export_path):
    """
    Export the Keras model to ONNX format.
    
    Args:
        model: Keras model to export.
        export_path: Path to save the ONNX model.
    """
    
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    
    # Save the ONNX model
    with open(export_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"Model exported to {export_path}")

model = build_model(latent_size=100, ntimesteps=656, num_classes=12, contextual=0)

model.load_weights(weights_path)
new_model = keras.Model(inputs=model.input, outputs=model.get_layer('latent').output)
print(new_model.summary())

# Export the model to ONNX format
export_path = "exports/nohost.onnx"
os.makedirs(os.path.dirname(export_path), exist_ok=True)
export_onnx(new_model, export_path)

