import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import os
sys.path.append('../')
from util import read_feature_file

def smooth_data_moving_average(series, window_size=3):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()


def remove_outliers_and_interpolate(series, z_threshold=3,window_size=3):
    z_scores = np.abs((series - series.mean()) / series.std())
    series_cleaned = series.where(z_scores < z_threshold, np.nan)
    interpolated_series = series_cleaned.interpolate(method='linear', limit_direction='both')
    return interpolated_series


def visualize_data(graphs):
    plt.figure(figsize=(10, 6 * len(graphs)))

    for i, graph in enumerate(graphs, start=1):
        plt.subplot(len(graphs), 1, i)

        for param in graph['params']:
            participant = ""
            recording_id = param['recording_id']
            feature = param['feature']
            if param['participant'] is not None:
                participant = param['participant']

            data = read_feature_file(recording_id=recording_id)

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            # Ensure the feature exists in the DataFrame
            if feature not in data.columns:
                print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                continue

            # Select the feature and frame_seq_number for the x-axis
            data = data[['frame_seq_number', feature]]
            # get the mean and SD of the data[feature] and display n the side of the graph
            mean = data[feature].mean()
            std = data[feature].std()

            # Interpolate missing values, remove outliers, and apply smoothing
            data[feature] = remove_outliers_and_interpolate(data[feature])
            data[feature] = smooth_data_moving_average(data[feature], window_size=graph.get('window_size', 4))

            plt.plot(data['frame_seq_number'], data[feature], label=f'{feature}_{participant}')
            # plt.plot(data['frame_seq_number'], data[feature])




        plt.xlabel(graph.get('xlabel', 'Frame Sequence Number'))
        plt.ylabel(graph.get('ylabel', 'Value'))
        plt.title(graph.get('title', 'Data Visualization'))
        plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_data_with_dual_y_axis(graphs):
    plt.figure(figsize=(10, 6 * len(graphs)))

    for i, graph in enumerate(graphs, start=1):
        ax1 = plt.subplot(len(graphs), 1, i)

        # Assume the first param uses ax1 and the second uses ax2
        for j, param in enumerate(graph['params']):
            participant = ""
            recording_id = param['recording_id']
            feature = param['feature']
            if param.get('participant') is not None:
                participant = param['participant']

            data = read_feature_file(recording_id=recording_id)

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            if feature not in data.columns:
                print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                continue

            data = data[['frame_seq_number', feature]]

            data[feature] = remove_outliers_and_interpolate(data[feature])
            data[feature] = smooth_data_moving_average(data[feature], window_size=graph.get('window_size', 4))

            if j == 0:
                # Plot on primary y-axis
                ax1.plot(data['frame_seq_number'], data[feature], label=f'{feature}_{participant}')
                ax1.set_xlabel(graph.get('xlabel', 'Frame Sequence Number'))
                ax1.set_ylabel(graph.get('ylabel1', 'Value'), color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.set_title(graph.get('title', 'Data Visualization'))
            else:
                # Plot on secondary y-axis
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.plot(data['frame_seq_number'], data[feature], label=f'{feature}_{participant}', color='tab:red')
                ax2.set_ylabel(graph.get('ylabel2', 'Value'), color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')

        # Only add legend to ax2 if it exists
        if 'ax2' in locals():
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            ax1.legend()

    plt.tight_layout()
    plt.show()
 # visualise_speed_profile of aa single participant
def visualise_wrist_speed_profile():

    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1062, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'}
            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Wrist Speeds'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # effected hand of a stroke survivor

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Wrist Speeds'
        },
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_speed_profile_effected_vs_uneffected():

    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'}

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Wrist Speeds'
        }
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_speed_profile_healthy_vs_stroke():

    graphs = [
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'}

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Wrist Speeds'
        }
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_speed_profile_vs_grip_aperture():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_GRIP_APERTURE', 'participant': '7002'}
            ],
            'window_size': 5,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Grip Aperture (mm)',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_GRIP_APERTURE', 'participant': '7002'}
            ],
            'window_size': 5,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Grip Aperture (mm)',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
         {
             'params': [
                 {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                 {'recording_id': 1065, 'feature': 'RIGHT_GRIP_APERTURE', 'participant': '7002'}
             ],
             'window_size': 5,
             'xlabel': 'Frame Sequence Number',
             'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
             'ylabel2': 'Grip Aperture (mm)',  # Label for the secondary y-axis
             'title': 'Right Wrist Speed vs Grip Aperture'
         },
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_acceleration_profile():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_ACCELERATION', 'participant': '7002'},
                # non effected hand of a stroke survivor
                # {'recording_id': 1062, 'feature': 'LEFT_WRIST_ACCELERATION', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed acceleration',
            'title': 'Right and Left Wrist acceleration'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_ACCELERATION', 'participant': '7002'},
                # non effected hand of a stroke survivor
                # {'recording_id': 1063, 'feature': 'RIGHT_WRIST_ACCELERATION', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed acceleration',
            'title': 'Right and Left Wrist acceleration'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_ACCELERATION', 'participant': '7002'},
                # non effected hand of a stroke survivor
                # {'recording_id': 1065, 'feature': 'LEFT_WRIST_ACCELERATION', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed acceleration',
            'title': 'Right and Left Wrist acceleration'
        },
        # {
        #     'params': [
        #         {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
        #         # non effected hand of a stroke survivor
        #         {'recording_id': 1062, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'}
        #     ],
        #     'window_size': 4,
        #     'xlabel': 'Frame Sequence Number',
        #     'ylabel': 'Smoothed Speed',
        #     'title': 'Right and Left Wrist Speeds'
        # },
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_acceleration_vs_speed_profile():
    graphs = [
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_ACCELERATION', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        {
            'params': [
                # {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_ACCELERATION', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        {
            'params': [
                # {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_ACCELERATION', 'participant': '6001'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_jerk_profile():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_JERK', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_JERK', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_JERK', 'participant': '7002'}
            ],
            'window_size': 1,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_elbow_speed_profile():

    graphs = [
        # {
        #     'params': [
        #         {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'},
        #         # non effected hand of a stroke survivor
        #         {'recording_id': 1062, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'}
        #     ],
        #     'window_size': 4,
        #     'xlabel': 'Frame Sequence Number',
        #     'ylabel': 'Smoothed Speed',
        #     'title': 'Right and Left Elbow Speeds'
        # },
        # {
        #     'params': [
        #         {'recording_id': 1063, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
        #         # non effected hand of a stroke survivor
        #         {'recording_id': 1063, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'},
        #         # effected hand of a stroke survivor
        #
        #     ],
        #     'window_size': 4,
        #     'xlabel': 'Frame Sequence Number',
        #     'ylabel': 'Smoothed Speed',
        #     'title': 'Right and Left Elbow Speeds'
        # },
        # {
        #     'params': [
        #         {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '6001'},
        #         # non effected hand of a stroke survivor
        #         {'recording_id': 1065, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '6001'},
        #         # effected hand of a stroke survivor
        #
        #     ],
        #     'window_size': 4,
        #     'xlabel': 'Frame Sequence Number',
        #     'ylabel': 'Smoothed Speed',
        #     'title': 'Right and Left Elbow Speeds'
        # },
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '6001'}

            ],
            'window_size': 6,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Elbow Speeds'
        },
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_elbow_speed_vs_wrist_speed_profile():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'}
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs elbow speed'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'}
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs elbow speed'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '6001'}
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Acceleration',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs Grip Aperture'
        },
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_elbow_acceleration_profile():

    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_ACCELERATION', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_ACCELERATION', 'participant': '7002'},
                # {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_ACCELERATION', 'participant': '6001'}

            ],
            'window_size': 2,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Elbow Speeds'
        },
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_wrist_flexion_extentin_angle():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1062, 'feature': 'LEFT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'}
            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Elbow Speeds'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'},
                # effected hand of a stroke survivor

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Elbow Speeds'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '6001'},
                # non effected hand of a stroke survivor
                {'recording_id': 1065, 'feature': 'LEFT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '6001'},
                # effected hand of a stroke survivor

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right and Left Elbow Speeds'
        },
        # {
        #     'params': [
        #         {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'},
        #         # non effected hand of a stroke survivor
        #         {'recording_id': 1063, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
        #         {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '6001'}
        #
        #     ],
        #     'window_size': 6,
        #     'xlabel': 'Frame Sequence Number',
        #     'ylabel': 'Smoothed Speed',
        #     'title': 'Right and Left Elbow Speeds'
        # },
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)

def visualise_wrist_flexion_extention_angle_vs_speed():
    graphs = [
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 1065, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'wrst fleexion angle',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs elbow speed'
        },
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

if __name__ == "__main__":
    # recording id: 1062 , frame_coordinate_id: 1017 , file : 7002_right_drink.mp4
    # recording id: 1063 , frame_coordinate_id: 1018 , file : 7002_left_drink.mp4.mp4
    # recording id: 1065 , frame_coordinate_id: 1020 , file : 6001_right_drink.mp4.mp4

    # visualise_speed_profile()
    # visualise_speed_profile_effected_vs_uneffected()
    # visualise_speed_profile_healthy_vs_stroke()
    # visualise_speed_profile_vs_grip_aperture()
    # visualise_acceleration_profile()
    # visualise_acceleration_vs_speed_profile()

    # visualise_elbow_speed_profile()
    # visualise_elbow_speed_vs_wrist_speed_profile()
    # visualise_elbow_acceleration_profile()  # there  is no significaant difference.
    # visualise_wrist_flexion_extentin_angle()
    visualise_wrist_flexion_extention_angle_vs_speed()