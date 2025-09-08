import onnxruntime as ort
import numpy as np
import pickle
from preprocess import load_data 

import os
import argparse
import sys
os.environ["ONNXRUN_DISABLE_CPU_AFFINITY"] = "1"


def run_anomaly_detection(data_path=None):
    """
    Run anomaly detection using ONNX models.
    
    Args:
        data_path (str, optional): Path to the input data file. If None, uses command line argument or default.
    """
    encoder_path = "exports/nohost.onnx"
    iforest_path = "exports/iforests"
    iforest_classes = list(range(12))
    
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

    input_data = input_data.astype(np.float32)
    # Run inference
    outputs = session.run(None, {input_name: input_data})[0]
    # Output is a list of NumPy arrays (one per output node)

    print("Latent output:", outputs)
    print(outputs.shape)

    # Run the saved ONNX models using ONNX Runtime
    onnx_sessions = {}
    for cls in iforest_classes:  # Replace with actual class names
        model_path = os.path.join(iforest_path, f"iforest_{cls}.onnx")
        onnx_sessions[cls] = ort.InferenceSession(model_path, sess_options=sess_options)

    onnx_outputs = {}
    for cls, sess in onnx_sessions.items():
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: outputs})
        onnx_outputs[cls] = -output[1].squeeze()
        # print(onnx_outputs[cls])

    mcif_output = []
    for cls in iforest_classes:
        mcif_output.append(onnx_outputs[cls]) 

    print('Discrete MCIF Output', np.array(mcif_output).T)
    anomaly_scores = np.min(mcif_output, axis=0)
    print('Anomaly Scores', anomaly_scores) 
    print('Min Detector', np.argmin(mcif_output, axis=0))
    print('Mean Anomaly Score', np.mean(anomaly_scores))
    
    return anomaly_scores, mcif_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection using ONNX models.")
    parser.add_argument("data_path", nargs="?", default=None, help="Path to the input data file (.pickle or .csv)")
    args = parser.parse_args()
    run_anomaly_detection(data_path=args.data_path)