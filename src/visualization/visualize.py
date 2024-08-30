import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.fft import fft, fftfreq
sys.path.append('../')
from util import read_feature_file

from util import DatabaseUtil
from IPython.display import display
from tabulate import tabulate
import textwrap
import scipy.signal as signal

def smooth_data_moving_average(series, window_size=3):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()


def remove_outliers_and_interpolate(series, z_threshold=3,window_size=3):
    z_scores = np.abs((series - series.mean()) / series.std())
    series_cleaned = series.where(z_scores < z_threshold, np.nan)
    interpolated_series = series_cleaned.interpolate(method='linear', limit_direction='both')
    return interpolated_series


# def visualize_data(graphs,distribution_boxplots=False):
#     plt.figure(figsize=(10, 6 * len(graphs)))
#
#     for i, graph in enumerate(graphs, start=1):
#         plt.subplot(len(graphs), 1, i)
#
#         for param in graph['params']:
#             participant = ""
#             recording_id = param['recording_id']
#             feature = param['feature']
#             if param['participant'] is not None:
#                 participant = param['participant']
#
#             data = read_feature_file(recording_id=[recording_id])
#
#             if data is None or data.empty:
#                 print(f"No data found for recording_id={recording_id}.")
#                 continue
#
#             # Ensure the feature exists in the DataFrame
#             if feature not in data.columns:
#                 print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
#                 continue
#
#             # Select the feature and frame_seq_number for the x-axis
#             data = data[['frame_seq_number', feature]]
#             # get the mean and SD of the data[feature] and display n the side of the graph
#             mean = data[feature].mean()
#             std = data[feature].std()
#
#             # Interpolate missing values, remove outliers, and apply smoothing
#             data[feature] = remove_outliers_and_interpolate(data[feature])
#             data[feature] = smooth_data_moving_average(data[feature], window_size=graph.get('window_size', 4))
#
#             plt.plot(data['frame_seq_number'], data[feature], label=param['label'])
#
#             # plt.plot(data['frame_seq_number'], data[feature])
#
#
#
#
#         plt.xlabel(graph.get('xlabel', 'Frame Sequence Number'))
#         plt.ylabel(graph.get('ylabel', 'Value'))
#         plt.title(graph.get('title', 'Data Visualization'))
#         plt.legend()
#
#     plt.tight_layout()
#     plt.show()
def wrap_labels(labels, width=10):
    """Wraps text labels to a specified width."""
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def visualize_data(graphs, distribution_boxplots=False):
    num_graphs = len(graphs)
    cols = 2 if distribution_boxplots else 1

    plt.figure(figsize=(10 * cols, 6 * num_graphs))

    participants = list(set(param.get('participant', '') for graph in graphs for param in graph['params']))
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_color_map = {participant: color for participant, color in zip(participants, colors)}


    for i, graph in enumerate(graphs, start=1):
        if distribution_boxplots:
            plt.subplot(num_graphs, cols, 2 * i - 1)
        else:
            plt.subplot(num_graphs, cols, i)

        for param in graph['params']:
            participant = ""
            recording_id = param['recording_id']
            feature = param['feature']
            if param['participant'] is not None:
                participant = param['participant']

            data = read_feature_file(recording_id=[recording_id])

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
            spectral_entrpy = calculate_spectral_entropy(data[feature]) # lower spectral entropy indicates a smoother signal
            wavelet_smoothness_index = calculate_wavelet_smoothness_index(data[feature]) # high wavelet_smoothness_index indicates a smoother signal
            # print smmothness index
            print(f"Smoothness Index - Spectral Entropy for {participant} {recording_id} {feature} is {spectral_entrpy}")
            print(
                f"Smoothness Index - wavelet_smoothness_index for {participant} {recording_id} {feature} is {wavelet_smoothness_index}")
            mean = data[feature].mean()
            std = data[feature].std()

            # Interpolate missing values, remove outliers, and apply smoothing
            data[feature] = remove_outliers_and_interpolate(data[feature])
            data[feature] = smooth_data_moving_average(data[feature], window_size=graph.get('window_size', 4))
            label = param.get('label', f"{participant}_{feature}")
            plt.plot(data['frame_seq_number'], data[feature], label=label)

        plt.xlabel(graph.get('xlabel', 'Frame Sequence Number'))
        plt.ylabel(graph.get('ylabel', 'Value'))
        plt.title(graph.get('title', 'Data Visualization'))
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

        if distribution_boxplots:
            plt.subplot(num_graphs, cols, 2 * i)
            data_list = []
            labels = []
            colors_list = []

            for param in graph['params']:
                recording_id = param['recording_id']
                feature = param['feature']
                participant = param.get('participant', '')
                label = param.get('label', '')

                data = read_feature_file(recording_id=[recording_id])

                if data is None or data.empty:
                    print(f"No data found for recording_id={recording_id}.")
                    continue

                if feature not in data.columns:
                    print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                    continue

                # Prepare the data for the box plot, removing outliers
                series_data = data[feature].dropna()
                clean_data = remove_outliers(series_data)
                data_list.append(clean_data)
                labels.append(label)
                colors_list.append(participant_color_map[participant])

            wrapped_labels = wrap_labels(labels, width=20)
            bp = plt.boxplot(data_list, labels=wrapped_labels, vert=True, patch_artist=True, showmeans=True,
                             boxprops={'alpha': 0.5, 'zorder': 0})


            for box, color in zip(bp['boxes'], colors_list):
                box.set_facecolor(color)

            for j, data in enumerate(data_list):
                x = np.random.normal(1 + j, 0.04, size=len(data))
                plt.scatter(x, data, alpha=0.3, color='black', zorder=3, s=2)

            plt.title(f'{graph.get("title", "Box Plot Visualization")}')
            plt.xlabel(f'{graph.get("xlabel", "Parameters")}')
            plt.ylabel(f'{graph.get("ylabel", "Values")}')

    plt.tight_layout()
    plt.show()



def draw_fourier_transform(graphs):
    plt.figure(figsize=(10, 6 * len(graphs)))

    for i, graph in enumerate(graphs, start=1):
        plt.subplot(len(graphs), 1, i)

        for param in graph['params']:
            recording_id = param['recording_id']
            feature = param['feature']
            data = read_feature_file(recording_id=[recording_id])

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            if feature not in data.columns:
                print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                continue

            # Prepare the signal
            signal = data[feature].dropna()
            signal = remove_outliers_and_interpolate(signal)
            # signal = smooth_data_moving_average(signal, window_size=graph.get('window_size', 4))
            signal_array = signal.to_numpy()
            # Compute the Fast Fourier Transform (FFT)
            N = len(signal_array)
            T = 1.0 / 10.0  # Assuming a frame rate of 30 fps; adjust as necessary
            yf = fft(signal_array)
            xf = fftfreq(N, T)[:N//2]

            # Plot the spectrum
            plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label=f'{feature} (Recording ID: {recording_id})')

        plt.title(graph.get('title', 'Fourier Transform'))
        plt.xlabel(graph.get('xlabel', 'Frequency (Hz)'))
        plt.ylabel(graph.get('ylabel', 'Amplitude'))
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
#


def draw_wavelet_transform(graphs):
    max_graphs_per_row = 2
    num_graphs = len(graphs[0]['params'])
    num_rows = (num_graphs + max_graphs_per_row - 1) // max_graphs_per_row  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, max_graphs_per_row, figsize=(10 * max_graphs_per_row, 6 * num_rows))

    # Ensure axes is always a 2D array
    if num_rows == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, graph in enumerate(graphs):
        for j, param in enumerate(graph['params']):
            ax = axes[j]
            recording_id = param['recording_id']
            feature = param['feature']
            data = read_feature_file(recording_id=[recording_id])
            title = param['label']

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            if feature not in data.columns:
                print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                continue

            # Prepare the signal
            signal = data[feature].dropna()
            signal = remove_outliers_and_interpolate(signal)
            signal_array = signal.to_numpy()

            # Define wavelet and scales
            scales = np.arange(1, 128)  # Adjust scale range as necessary for your data
            wavelet = 'cmor'  # Complex Morlet wavelet

            # Perform the Continuous Wavelet Transform
            coefficients, frequencies = pywt.cwt(signal_array, scales, wavelet, 1.0 / 30.0)  # Assuming 30 fps as sampling rate

            # Plotting the wavelet transform
            cax = ax.imshow(np.abs(coefficients), extent=[0, len(signal_array), frequencies.min(), frequencies.max()], cmap='jet', aspect='auto', interpolation='nearest')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'{feature} - {title}')

            # Add colorbar to each subplot
            fig.colorbar(cax, ax=ax, orientation='vertical', label='Magnitude')

    # Hide any unused subplots
    for k in range(num_graphs, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()
# def draw_wavelet_transform(graphs):
#     plt.figure(figsize=(10, 6 * len(graphs)))
#
#     for i, graph in enumerate(graphs, start=1):
#         plt.subplot(len(graphs), 1, i)
#
#         for param in graph['params']:
#             recording_id = param['recording_id']
#             feature = param['feature']
#             data = read_feature_file(recording_id=[recording_id])
#
#             if data is None or data.empty:
#                 print(f"No data found for recording_id={recording_id}.")
#                 continue
#
#             if feature not in data.columns:
#                 print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
#                 continue
#
#             # Prepare the signal
#             signal = data[feature].dropna()
#             signal = remove_outliers_and_interpolate(signal)
#             # signal = smooth_data_moving_average(signal, window_size=graph.get('window_size', 4))
#
#             # Convert pandas Series to numpy array
#             signal_array = signal.to_numpy()
#
#             # Define wavelet and scales
#             scales = np.arange(1, 128)  # Adjust scale range as necessary for your data
#             wavelet = 'cmor'  # Complex Morlet wavelet
#
#             # Perform the Continuous Wavelet Transform
#             coefficients, frequencies = pywt.cwt(signal_array, scales, wavelet, 1.0 / 30.0)  # Assuming 30 fps as sampling rate
#
#             # Plotting the wavelet transform
#             plt.imshow(np.abs(coefficients), extent=[0, len(signal_array), frequencies.min(), frequencies.max()], cmap='jet', aspect='auto', interpolation='nearest')
#             plt.colorbar(label='Magnitude')
#             plt.xlabel('Time')
#             plt.ylabel('Frequency (Hz)')
#             plt.title(f'{feature} - Wavelet Transform (Recording ID: {recording_id})')
#
#         plt.tight_layout()
#     plt.show()

def visualise_box_plot(graphs):
    num_graphs = len(graphs)
    cols = 2  # Number of columns
    rows = (num_graphs + 1) // cols  # Calculate required number of rows
    plt.figure(figsize=(10 * cols, 6 * rows))  # Scale figure size based on number of subplots

    for i, graph in enumerate(graphs, start=1):
        ax = plt.subplot(rows, cols, i)  # Create a subplot in a grid
        data_list = []  # This will hold the data for each parameter to plot
        labels = []  # This will hold the labels for the box plots

        for param in graph['params']:
            recording_id = param['recording_id']
            feature = param['feature']
            participant = param.get('participant', '')

            data = read_feature_file(recording_id=recording_id)  # Assume this function fetches your data

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            if feature not in data.columns:
                print(f"Feature {feature} not found in the dataset for recording_id={recording_id}.")
                continue

            # Prepare the data for the box plot, removing outliers
            series_data = data[feature].dropna()  # Remove NaN values if any
            clean_data = remove_outliers(series_data)  # Remove outliers
            data_list.append(clean_data)
            labels.append(f'{feature}_{participant}')

        # Create the box plot with reduced opacity
        bp = ax.boxplot(data_list, labels=labels, vert=True, patch_artist=True, showmeans=True,
                        boxprops={'alpha': 0.5, 'zorder': 0})

        # Add scatter plot, ensure it's more prominent
        for j, data in enumerate(data_list):
            x = np.random.normal(1+j, 0.04, size=len(data))
            # ax.scatter(x, data, alpha=0.3, color='red', edgecolors='k', zorder=3)
            ax.scatter(x, data, alpha=0.3, color='black', zorder=3,s=2)

        # Annotate Q1, Q2, Q3
        for line, data in zip(bp['medians'], data_list):
            median_x = line.get_xdata()[0]
            median_y = line.get_ydata()[0]
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            ax.annotate(f'Q1={Q1:.2f}', xy=(median_x, Q1), xytext=(10,-10),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))
            ax.annotate(f'Q2={median_y:.2f}', xy=(median_x, median_y), xytext=(10,0),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))
            ax.annotate(f'Q3={Q3:.2f}', xy=(median_x, Q3), xytext=(10,10),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))

        ax.set_title(graph.get('title', 'Box Plot Visualization'))
        ax.set_xlabel(graph.get('xlabel', 'Parameters'))
        ax.set_ylabel(graph.get('ylabel', 'Values'))

    plt.tight_layout()
    plt.show()
#
# def visualize_data_polar_coordinates(graphs):
#     plt.figure(figsize=(10, 6 * len(graphs)))
#
#     for i, graph in enumerate(graphs, start=1):
#         ax = plt.subplot(len(graphs), 1, i, projection='polar')  # Set up polar plot
#
#         for param in graph['params']:
#             participant = ""
#             recording_id = param['recording_id']
#             hand = param['hand']
#             finger = param['finger']
#             if param['participant'] is not None:
#                 participant = param['participant']
#
#             data = read_feature_file(recording_id=[recording_id])
#
#             if data is None or data.empty:
#                 print(f"No data found for recording_id={recording_id}.")
#                 continue
#
#             # Construct dynamic feature names based on hand and finger
#             ratio_feature = f"{hand}_{finger}_RATIO_MCP_TIP_DISTAL"
#             angle_feature1 = f"{hand}_{finger}_ANGLE_WRIST_MCP_PIP"
#             angle_feature2 = f"{hand}_{finger}_ANGLE_MCP_PIP_DIP"
#             angle_feature3 = f"{hand}_{finger}_ANGLE_PIP_DIP_TIP"
#             angle_wrist_mcp_tip = f"{hand}_{finger}_ANGLE_WRIST_MCP_TIP"
#
#
#             # angle_feature1 = f"{hand}_{finger}_FINGER_ANGLE_MCP_PIP_DIP"
#             # angle_feature2 = f"{hand}_{finger}_FINGER_ANGLE_WRIST_MCP_TIP"
#             # angle_feature3 = f"{hand}_{finger}_FINGER_ANGLE_WRIST_MCP_TIP"
#             # LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP
#             #    LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP
#             #     LEFT_INDEX_FINGER_ANGLE_MCP_PIP_DIP
#             #     LEFT_INDEX_FINGER_ANGLE_PIP_DIP_TIP
#             #     LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP
#
#             if not all(feature in data.columns for feature in [ratio_feature, angle_feature1, angle_feature2]):
#                 print(f"Required features not found in the dataset for recording_id={recording_id}.")
#                 continue
#
#             # Compute the theta as the sum of angles
#             # data['theta'] = data[angle_feature1] + data[angle_feature2] + data[angle_feature3]
#             data['theta'] = data[angle_wrist_mcp_tip] # Normalize to 0-360 degrees
#
#             # Interpolate missing values and remove outliers
#             data[ratio_feature] = remove_outliers_and_interpolate(data[ratio_feature])
#             data['theta'] = remove_outliers_and_interpolate(data['theta'])
#             # Convert degrees to radians for plotting
#             data['theta'] = np.deg2rad(data['theta'])
#
#             # Plot
#             ax.scatter(data['theta'], data[ratio_feature], label=f'{hand}_{finger}_{participant}', s=12,alpha=0.7)
#
#         ax.set_title(graph.get('title', 'Polar Plot Visualization'))
#         ax.set_xlabel('Theta (radians)')
#         ax.set_ylabel('Radius')
#         ax.legend()
#
#     plt.tight_layout()
#     plt.show()


def visualize_data_polar_coordinates(graphs):
    num_graphs = len(graphs)
    cols = 4  # Number of columns (for four fingers)
    rows = (num_graphs + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(10 * cols, 6 * rows))  # Adjust the figure size to fit the number of subplots

    for i, graph in enumerate(graphs):
        # Set up the title for the group of four graphs
        if i % cols == 0:
            task_id = graph['params'][0]['recording_id']
            session_number = graph['params'][0]['recording_id']
            participant = graph['params'][0]['participant']
            plt.suptitle(f'Task ID: {task_id}, Session Number: {session_number}, Participant: {participant}', y=1.02, fontsize=16)

        for j, param in enumerate(graph['params']):
            ax = plt.subplot(rows, cols, i * cols + j + 1, projection='polar')  # Set up polar plot

            recording_id = param['recording_id']
            hand = param['hand']
            finger = param['finger']
            participant = param.get('participant', '')

            data = read_feature_file(recording_id=[recording_id])

            if data is None or data.empty:
                print(f"No data found for recording_id={recording_id}.")
                continue

            # Construct dynamic feature names based on hand and finger
            ratio_feature = f"{hand}_{finger}_RATIO_MCP_TIP_DISTAL"
            angle_feature1 = f"{hand}_{finger}_ANGLE_WRIST_MCP_PIP"
            angle_feature2 = f"{hand}_{finger}_ANGLE_MCP_PIP_DIP"
            angle_feature3 = f"{hand}_{finger}_ANGLE_PIP_DIP_TIP"
            angle_wrist_mcp_tip = f"{hand}_{finger}_ANGLE_WRIST_MCP_TIP"

            if not all(feature in data.columns for feature in [ratio_feature, angle_feature1, angle_feature2]):
                print(f"Required features not found in the dataset for recording_id={recording_id}.")
                continue

            # Compute the theta as the sum of angles
            data['theta'] = data[angle_wrist_mcp_tip]  # Normalize to 0-360 degrees

            # Interpolate missing values and remove outliers
            data[ratio_feature] = remove_outliers_and_interpolate(data[ratio_feature])
            data['theta'] = remove_outliers_and_interpolate(data['theta'])
            # Convert degrees to radians for plotting
            data['theta'] = np.deg2rad(data['theta'])

            # Plot
            ax.scatter(data['theta'], data[ratio_feature], label=f'{hand}_{finger}_{participant}', s=12, alpha=0.7)

            ax.set_title(f'{hand}_{finger}', fontsize=10)
            ax.set_xlabel('Theta (radians)')
            ax.set_ylabel('Radius')
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to fit the title
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

def remove_outliers(data):
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define outliers as those points outside the Q1 - 1.5 * IQR and Q3 + 1.5 * IQR range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Return data without outliers
    return data[(data >= lower_bound) & (data <= upper_bound)]


def calculate_spectral_entropy(time_series):
    cleaned_time_series = time_series[~np.isnan(time_series)]
    freqs, psd = signal.welch(cleaned_time_series)
    psd_norm = psd / np.sum(psd)  # Normalize the power spectrum
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return spectral_entropy


def calculate_wavelet_smoothness_index(time_series, wavelet='db4'):
    cleaned_time_series = time_series[~np.isnan(time_series)]

    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(cleaned_time_series, wavelet)

    # Extract low-frequency coefficients (approximation coefficients)
    approx_coeffs = coeffs[0]

    # Calculate energy of approximation coefficients (low-frequency)
    low_freq_energy = np.sum(np.square(approx_coeffs))

    # Calculate total energy
    total_energy = sum(np.sum(np.square(c)) for c in coeffs)

    # Calculate smoothness index
    smoothness_index = low_freq_energy / total_energy

    return smoothness_index

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
                # {'recording_id': 2171, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2171, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2161, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2161, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2169, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2169, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 2189, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7001'},
                # {'recording_id': 2189, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7001'},

                # non effected hand of a stroke survivor
                {'recording_id': 3152, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 3151, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7001'}

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Wrist Speed',
            'title': 'Right and Left Wrist Speeds'
        }
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)
    # draw_fourier_transform(graphs)
    draw_wavelet_transform(graphs)

def visualise_speed_profile_healthy_vs_stroke():

    graphs = [
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right Wrist Speeds - Stroke Survivor(non-effected hand)'
        },
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Left Wrist Speeds - Stroke Survivor(effected hand)'
        },
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # non effected hand of a stroke survivor
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'}

            ],
            'window_size': 4,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Smoothed Speed',
            'title': 'Right Wrist Speeds - Healthy participant'
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
            'title': 'Right Wrist Speed vs wrist flexion extension angle'
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
            'title': 'Left Wrist Speed vs wrist flexion extension angle'
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
            'title': 'Right and Left Wrist Flexion angles'
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
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 1065, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'wrst fleexion angle',  # Label for the secondary y-axis
            'title': 'Left Wrist Speed vs wrist flexion extension angle'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 1065, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'wrst fleexion angle',  # Label for the secondary y-axis
            'title': 'Right Wrist Speed vs wrist flexion extension angle'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
                # {'recording_id': 1065, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_FLEXION_EXTENSION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'wrst fleexion angle',  # Label for the secondary y-axis
            'title': 'Left Wrist Speed vs wrist flexion extension angle'
        }
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_elbow_flexion_angle():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'LEFT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Elbow Flexion Angle',
            'title': 'Right and Left Elbow Flexion Angle'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Elbow Flexion Angle',
            'title': 'Right and Left Elbow Flexion Angle'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '6001'},
                {'recording_id': 1065, 'feature': 'LEFT_ELBOW_FLEXION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Elbow Flexion Angle',
            'title': 'Right and Left Elbow Flexion Angle'
        }
        # Add more graph configurations if needed
    ]
    visualize_data(graphs)


def visualise_elbow_flexion_angle_vs_speed():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Elbow Flexion Angle',  # Label for the secondary y-axis
            'title': 'Right Elbow Speed vs Elbow Flexion Angle'
        },
        {
            'params': [
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Elbow Flexion Angle',  # Label for the secondary y-axis
            'title': 'Left Elbow Speed vs Elbow Flexion Angle'
        },
        {
            'params': [
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Elbow Flexion Angle',  # Label for the secondary y-axis
            'title': 'Right Elbow Speed vs Elbow Flexion Angle'
        }
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_shoulder_abduction_angle():
    # graphs = [
    #     {
    #         'params': [
    #             {'recording_id': 1062, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
    #             {'recording_id': 1062, 'feature': 'LEFT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
    #         ],
    #         'window_size': 3,
    #         'xlabel': 'Frame Sequence Number',
    #         'ylabel': 'Shoulder Abduction Angle',
    #         'title': 'Right and Left Shoulder Abduction Angle'
    #     },
    #     {
    #         'params': [
    #             {'recording_id': 1063, 'feature': 'LEFT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
    #             {'recording_id': 1063, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
    #         ],
    #         'window_size': 3,
    #         'xlabel': 'Frame Sequence Number',
    #         'ylabel': 'Shoulder Abduction Angle',
    #         'title': 'Right and Left Shoulder Abduction Angle'
    #     },
    #     {
    #         'params': [
    #             {'recording_id': 1065, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '6001'},
    #             {'recording_id': 1065, 'feature': 'LEFT_SHOULDER_ABDUCTION_ANGLE', 'participant': '6001'},
    #         ],
    #         'window_size': 3,
    #         'xlabel': 'Frame Sequence Number',
    #         'ylabel': 'Shoulder Abduction Angle',
    #         'title': 'Right and Left Shoulder Abduction Angle'
    #     }
    #     # Add more graph configurations if needed
    # ]
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Shoulder Abduction Angle',
            'title': 'Right and Left Shoulder Abduction Angle'
        }
        # Add more graph configurations if needed
    ]

    visualize_data(graphs)


#    LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_PIP
#     LEFT_INDEX_FINGER_ANGLE_MCP_PIP_DIP
#     LEFT_INDEX_FINGER_ANGLE_PIP_DIP_TIP
#     LEFT_INDEX_FINGER_ANGLE_WRIST_MCP_TIP
#
#     LEFT_INDEX_FINGER_RATIO_MCP_TIP_DISTAL

def visulise_finger_density_function():

    graphs = [
        {
            'params': [
                {'recording_id': 1065, 'hand': 'RIGHT','finger': 'INDEX_FINGER', 'participant': '7002'},
                # {'recording_id': 1062, 'hand': 'RIGHT','finger': 'MIDDLE_FINGER', 'participant': '7002'},
                # {'recording_id': 1062, 'hand': 'RIGHT','finger': 'RING_FINGER', 'participant': '7002'},
                # {'recording_id': 1062, 'hand': 'RIGHT','finger': 'PINKY', 'participant': '7002'},
                # {'recording_id': 1063, 'hand': 'LEFT', 'finger': 'INDEX', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Shoulder Abduction Angle',
            'title': 'Right and Left Shoulder Abduction Angle'
        },
        {
            'params': [
                # {'recording_id': 1063, 'hand': 'LEFT', 'finger': 'INDEX_FINGER', 'participant': '7002'},
                # {'recording_id': 1063, 'hand': 'LEFT', 'finger': 'PINKY', 'participant': '7002'},
                {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'MIDDLE_FINGER', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Shoulder Abduction Angle',
            'title': 'Right and Left Shoulder Abduction Angle'
        },
        {
            'params': [
                # {'recording_id': 1063, 'hand': 'LEFT', 'finger': 'MIDDLE', 'participant': '7002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'INDEX_FINGER', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'MIDDLE', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'RING', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'PINKY', 'participant': '6002'},
                {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'RING_FINGER', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Shoulder Abduction Angle',
            'title': 'Right and Left Shoulder Abduction Angle'
        },
        {
            'params': [
                # {'recording_id': 1063, 'hand': 'LEFT', 'finger': 'MIDDLE', 'participant': '7002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'INDEX_FINGER', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'MIDDLE', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'RING', 'participant': '6002'},
                # {'recording_id': 1065, 'hand': 'RIGHT', 'finger': 'PINKY', 'participant': '6002'},
                {'recording_id': 1065, 'hand': 'RIGHT','finger': 'PINKY', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Shoulder Abduction Angle',
            'title': 'Right and Left Shoulder Abduction Angle'
        }
        # Add more graph configurations if needed
    ]

    visualize_data_polar_coordinates(graphs)


def visualise_object_speed_profile():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'object_2_speed', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'object_2_speed', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'object_2_speed', 'participant': '7002'},
                # {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                # {'recording_id': 1063, 'feature': 'object_1_speed', 'participant': '7002'},
                # {'recording_id': 1065, 'feature': 'object_1_speed', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Speed',
            'title': 'Object Speed'
        }
    ]

    visualize_data(graphs)

def visualise_object_tracjectory_deviation():
    graphs = [
        {
            'params': [
                # {'recording_id': 1062, 'feature': 'object_1_trajectory_deviation', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'object_1_trajectory_deviation', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Deviation',
            'title': 'Object Deviation'
        }
    ]

    visualize_data(graphs)
def visualise_hand_object_iou():
    graphs = [
        {
            'params': [
                {'recording_id': 1065, 'feature': 'object_1_left_hand_iou', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'object_1_right_hand_iou', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'IoU',
            'title': 'Object IoU'
        }
    ]

    visualize_data(graphs)

def visualise_hand_object_iou_vs_speed():
    graphs = [
        {
            'params': [
                {'recording_id': 1065, 'feature': 'object_1_left_hand_iou', 'participant': '6001'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel1': 'Speed (units/s)',  # Label for the primary y-axis
            'ylabel2': 'Elbow Flexion Angle',  # Label for the secondary y-axis
            'title': 'Object hand iou vs wrist speed'
        }
        # Add more graph configurations if needed
    ]
    visualize_data_with_dual_y_axis(graphs)

def visualise_hand_ious():
    graphs = [
        {
            'params': [
                {'recording_id': 1065, 'feature': 'HAND_BOUNDING_BOX_IOU', 'participant': '7002'},
            ],
            'window_size': 3,
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'IoU',
            'title': 'Object IoU'
        }
    ]

    visualize_data(graphs)

def visualise_speed_distribution():
    graphs = [
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_WRIST_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_WRIST_SPEED', 'participant': '6001'},
            ],
            'window_size': 1,
            'xlabel': 'Recording ID',
            'ylabel': 'Wrist Speed',
            'title': 'Wrist Speed Distribution'
        },
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_SPEED', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_SPEED', 'participant': '6001'},
            ],
            'window_size': 1,
            'xlabel': 'Recording ID',
            'ylabel': 'Elbow Speed',
            'title': 'Elbow Speed Distribution'
        },
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_ELBOW_FLEXION_ANGLE', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_ELBOW_FLEXION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 1,
            'xlabel': 'Recording ID',
            'ylabel': 'Elbow Speed',
            'title': 'Elbow Flexion Extension Angle Distribution'
        },
        {
            'params': [
                {'recording_id': 1062, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
                {'recording_id': 1063, 'feature': 'LEFT_SHOULDER_ABDUCTION_ANGLE', 'participant': '7002'},
                {'recording_id': 1065, 'feature': 'RIGHT_SHOULDER_ABDUCTION_ANGLE', 'participant': '6001'},
            ],
            'window_size': 1,
            'xlabel': 'Recording ID',
            'ylabel': 'Abduction angle',
            'title': 'Shoulder Abduction Angle Distribution'
        },
        # {
        #     'params': [
        #         {'recording_id': 1062, 'feature': 'object_1_trajectory_deviation', 'participant': '7002'},
        #         {'recording_id': 1063, 'feature': 'object_1_trajectory_deviation', 'participant': '7002'},
        #         {'recording_id': 1065, 'feature': 'object_1_trajectory_deviation', 'participant': '6001'},
        #     ],
        #     'window_size': 1,
        #     'xlabel': 'Recording ID',
        #     'ylabel': 'Abduction angle',
        #     'title': 'Object Trajectory Error Distribution'
        # }
    ]

    visualise_box_plot(graphs)



def read_video_feature_data(recording_ids, metaData):
    db_util = DatabaseUtil()
    hand_specific_features = []
    for param in metaData['params']:
        hand = param['hand']
        for feature in metaData['hand_specific_features']:
            hand_specific_features.append(
                f"{hand}_{feature}".lower())  # Adjust feature names according to DB structure

    general_features = [feature.lower() for feature in metaData['general_features']]

    columns = hand_specific_features + general_features
    recording_ids_str = ','.join(map(str, recording_ids))
    fetch_query = f"""
    SELECT recording_id, {', '.join(columns)} FROM VisionAnalysis.dbo.video_features
    WHERE recording_id IN ({recording_ids_str})
    """
    results = db_util.fetch_data(fetch_query)
    df_list = []
    # Convert results to DataFrame
    print(results)
    df = pd.DataFrame(columns=['recording_id'] + columns)

    # Loop through the results and append each row to the DataFrame
    for row in results:
        row_df = pd.DataFrame([list(row)], columns=['recording_id'] + columns)
        df_list.append(row_df)

    #  if df list is empty return empty dataframe
    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    return df


def print_kinematic_stats(metaData):
    recording_ids = [param['recording_id'] for param in metaData['params']]
    df = read_video_feature_data(recording_ids, metaData)

    # in df remove if any duplicate column names appear
    df = df.loc[:, ~df.columns.duplicated()]

    # Prepare the new DataFrame
    new_columns = ['recording_id', 'hand', 'participant'] + \
                  [f"{feature}".lower() for feature in metaData['hand_specific_features']] + \
                  metaData['general_features']
    new_df = pd.DataFrame(columns=new_columns)
    new_df = new_df.loc[:, ~new_df.columns.duplicated()]
    # Populate the new DataFrame
    rows_list = []
    for param in metaData['params']:
        recording_id = param['recording_id']
        hand = param['hand']
        participant = param['participant']
        row_data = {'recording_id': recording_id, 'hand': hand, 'participant': participant}
        for feature in metaData['hand_specific_features']:
            col_name = f"{hand}_{feature}".lower()
            feature_name = f"{feature}".lower()
            if col_name in df.columns:
                feature_data = df[df['recording_id'] == recording_id][col_name]
                if not feature_data.empty:
                    row_data[feature_name] = feature_data.values[0]
                else:
                    row_data[feature_name] = None
        for feature in metaData['general_features']:
            if feature in df.columns:
                feature_data = df[df['recording_id'] == recording_id][feature]
                if not feature_data.empty:
                    row_data[feature] = feature_data.values[0]
        rows_list.append(row_data)

    new_df = pd.concat([new_df, pd.DataFrame(rows_list)], ignore_index=True)

    # Display the new DataFrame
    pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
    # display(new_df)
    print(tabulate(new_df, headers='keys', tablefmt='psql'))
    # print(new_df)


#
# def print_kinematic_stats():
#     metaData = {
#         'params': [
#             {'recording_id': 2127, 'hand': 'right', 'participant': '7002'},
#             {'recording_id': 2128, 'hand': 'right', 'participant': '7002'},
#         ],
#         'hand_specific_features': ['wrist_number_of_velocity_peaks',
#                                    'wrist_number_of_direction_changes', 'wrist_rate_of_direction_changes',
#                                    'wrist_total_traversed_distance'],
#         'general_features': ['total_trajectory_error', 'completion_time']
#     }
#
#     recording_ids = [param['recording_id'] for param in metaData['params']]
#     df = read_video_feature_data(recording_ids, metaData)
#
#     # in df remove if aany duplicate column names aappar
#     df = df.loc[:, ~df.columns.duplicated()]
#
#     # Prepare the new DataFrame
#     new_columns = ['recording_id', 'hand', 'participant'] + \
#                   [f"{feature}".lower() for feature in metaData['hand_specific_features']] + \
#                   metaData['general_features']
#     new_df = pd.DataFrame(columns=new_columns)
#     new_df = new_df.loc[:, ~new_df.columns.duplicated()]
#     # Populate the new DataFrame
#     rows_list = []
#     for param in metaData['params']:
#         recording_id = param['recording_id']
#         hand = param['hand']
#         participant = param['participant']
#         row_data = {'recording_id': recording_id, 'hand': hand, 'participant': participant}
#         for feature in metaData['hand_specific_features']:
#             col_name = f"{hand}_{feature}".lower()
#             feature_name= f"{feature}".lower()
#             if col_name in df.columns:
#                 row_data[feature_name] = df[df['recording_id'] == recording_id][col_name].values[0]
#         for feature in metaData['general_features']:
#             if feature in df.columns:
#                 row_data[feature] = df[df['recording_id'] == recording_id][feature].values[0]
#         rows_list.append(row_data)
#
#     new_df = pd.concat([new_df, pd.DataFrame(rows_list)], ignore_index=True)
#
#
#     # Display the new DataFrame
#     pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
#     # display_dataframe_in_window(new_df)
#     # df.describe()
#     display(new_df)
#     print(tabulate(new_df, headers='keys', tablefmt='psql'))
#     # print(new_df)

# def show_profile():

# userid_assessmentid_taskid_affected




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
    # visualise_elbow_acceleration_profile()  # there is no significant difference.
    # visualise_wrist_flexion_extentin_angle()
    # visualise_wrist_flexion_extention_angle_vs_speed()
    # visualise_elbow_flexion_angle()
    # visualise_elbow_flexion_angle_vs_speed()
    #visualise_shoulder_abduction_angle() # left shoulder, right shoulder , right elbow anggle
    visulise_finger_density_function()
    # visualise_object_speed_profile()
    # visualise_object_tracjectory_deviation()
    # visualise_hand_object_iou()
    # visualise_hand_object_iou_vs_speed()
    # visualise_hand_ious()

    # visualise_speed_distribution()
    # print_kinematic_stats()





