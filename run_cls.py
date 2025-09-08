import onnxruntime as ort
import numpy as np
import pickle
from preprocess import load_data 

import os
import argparse
import sys
from sklearn.preprocessing import OneHotEncoder

os.environ["ONNXRUN_DISABLE_CPU_AFFINITY"] = "1"


def run_classification(data_path=None):
    """
    Run classification using ONNX model.
    
    Args:
        data_path (str, optional): Path to the input data file. If None, uses command line argument or default.
    
    Returns:
        tuple: (classification_outputs, predicted_class_names) containing the raw outputs and predicted class names
    """
    class_names = ['SNIa', 'SNIa-91bg', 'SNIax', 'SNIb', 'SNIc', 'SNIc-BL', 'SNII', 'SNIIn', 'SNIIb', 'TDE', 'SLSN-I', 'AGN']
    # Use sklearn's OneHotEncoder to get the ordered class names
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(class_names).reshape(-1, 1))
    ordered_class_names = encoder.categories_[0]
    ordered_class_names = np.array(ordered_class_names)

    encoder_path = "exports/nohost_classifier.onnx"
    
    # Set default data path if not provided
    if data_path is None:
        default_data_path = 'data/example.pickle'
        if len(sys.argv) < 2:
            print("Using default data path:", default_data_path)
            data_path = default_data_path
        else:
            data_path = sys.argv[1]

    # Optional: Limit thread count as well
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1

    # Load the ONNX model
    session = ort.InferenceSession(encoder_path, sess_options=sess_options)

    # Get model input name and shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input layer name: {input_name}")
    print(f"Expected input shape: {input_shape}")

    if (data_path.endswith('.pickle')):
        with open(data_path, "rb") as f:
            input_data = pickle.load(f)
    elif (data_path.endswith('.csv')):
        input_data = load_data(data_path)
        print(len(input_data))
        assert(False)
    else:
        raise ValueError("Unsupported file format. Please provide a .pickle or .csv file.")

    print(input_data)
    input_data = np.array(input_data)

    input_data = input_data.astype(np.float32)
    # Run inference
    outputs = session.run(None, {input_name: input_data})[0]
    # Output is a list of NumPy arrays (one per output node)

    print("Classification output:", outputs)
    print(outputs.shape, len(ordered_class_names))
    predicted_classes = ordered_class_names[np.argmax(outputs, axis=1)]
    print(predicted_classes)
    
    return outputs, predicted_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection using ONNX models.")
    parser.add_argument("data_path", nargs="?", default=None, help="Path to the input data file (.pickle or .csv)")
    args = parser.parse_args()
    run_classification(data_path=args.data_path)
