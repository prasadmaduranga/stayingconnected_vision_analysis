import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from util import DatabaseUtil # Assuming DatabaseUtil is defined elsewhere

def fetch_frame_feature(recording_id, frame_feature):
    db_util = DatabaseUtil()
    query = f"""
    SELECT frame_seq_number, {frame_feature}
    FROM VisionAnalysis.dbo.frame_features
    WHERE recording_id = ?
    ORDER BY frame_seq_number;
    """
    results = db_util.fetch_data(query, [recording_id])
    return results


def moving_average(data, window_size):
    """Compute the moving average of the given data using a specified window size."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def interpolate_missing_values(data):
    """Interpolates missing (None) values in a list."""
    valid_indices = [i for i, v in enumerate(data) if v is not None]
    valid_values = [v for v in data if v is not None]

    # Use numpy interpolation
    interpolated_values = np.interp(range(len(data)), valid_indices, valid_values)
    return interpolated_values

def visualize_data(params, window_size=5):
    plt.figure(figsize=(10, 6))

    for param in params:
        recording_id = param['recordingid']
        frame_feature = param['frame_feature']
        data = fetch_frame_feature(recording_id, frame_feature)
        if not data:
            print(f"No data found for recording_id={recording_id} and frame_feature={frame_feature}")
            continue
        frame_seq_numbers = [row[0] for row in data]
        feature_values = [row[1] for row in data]

        feature_values = interpolate_missing_values(feature_values)

        # Apply moving average to smooth the data
        smoothed_values = moving_average(feature_values, window_size)
        # smoothed_values = feature_values

        # Adjust frame_seq_numbers to align with the length of smoothed_values
        adjusted_frame_seq_numbers = frame_seq_numbers[:len(smoothed_values)]

        plt.plot(adjusted_frame_seq_numbers, smoothed_values, label=f'Recording {recording_id}: {frame_feature}')

    plt.xlabel('Frame Sequence Number')
    plt.ylabel('Smoothed Feature Value')
    plt.title('Frame Feature Visualization with Moving Average')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    params = [
        {'recordingid': 36, 'frame_feature': 'right_wrist_speed'},
        {'recordingid': 38, 'frame_feature': 'left_wrist_speed'},
        # {'recordingid': 17, 'frame_feature': 'right_wrist_speed'}
    ]

    visualize_data(params, window_size=4)
