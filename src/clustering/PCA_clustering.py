import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import colormaps
import plotly.express as px

# Import the custom utilities
from src.util.file_util import read_feature_file, get_recording_metadata

# Constants
FEATURE_FILE_PATH = "data/time_series_features_dwt_smoothed_full_{feature_prop}.csv"


def extract_time_series_features(recording_ids, feature_prop):
    # recording_ids remove any nan or null values
    recording_ids = [recording_id for recording_id in recording_ids if recording_id]
    # Read existing feature file if available
    feature_file_path = FEATURE_FILE_PATH.format(feature_prop=feature_prop.lower())
    try:
        existing_df = pd.read_csv(feature_file_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    updated_rows = []

    # if existing_df is not empty, get the recording_ids that are already in the existing_df
    if not existing_df.empty:
        existing_recording_ids = existing_df['recording_id'].astype(str).values
    else:
        existing_recording_ids = []

    # remove recording_ids that are already in the existing_df
    # Filter out recording_ids that already exist in existing_df
    recording_ids = [str(recording_id) for recording_id in recording_ids if
                     str(recording_id) not in existing_recording_ids]
    for recording_id in recording_ids:
        # Check if the recording already exists
        if not existing_df.empty:
            if not existing_df[existing_df['recording_id'] == recording_id].empty:
                continue

        # Get metadata for the recording
        metadata = get_recording_metadata([recording_id])
        meta = metadata[recording_id]
        user_info = meta["user"]

        # Read the feature file for the recording
        feature_df = read_feature_file(recording_id=[recording_id])
        if feature_df is None:
            continue

        # Check hand type and adjust feature_prop accordingly
        if meta["hand"] == "affected":
            feature_column = f"{user_info['affected_hand'].upper()}_{feature_prop}"
            if feature_column not in feature_df.columns:
                continue

            feature_values = feature_df[feature_column].dropna()
            #  smooth the data using smooth_data_moving_average
            # feature_values = smooth_data_moving_average(feature_values, window_size=3)

            mean_val = feature_values.mean()
            median_val = feature_values.median()
            std_val = feature_values.std()

            # Extract wavelet features
            dwt_features = extract_wavelet_features(feature_values)

            # Combine features and metadata into a single row
            row = {
                "recording_id": recording_id,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                **dwt_features,
                "user_id": user_info["user_id"],
                "session_number": meta["session_number"],
                "task": meta["task"],
                "hand": meta["hand"]
                # "affected_hand": user_info["affected_hand"],
                # "unaffected_hand": user_info["unaffected_hand"]
            }
            updated_rows.append(row)

        elif meta["hand"] == "unaffected":
            feature_column = f"{user_info['unaffected_hand'].upper()}_{feature_prop}"
            if feature_column not in feature_df.columns:
                continue

            feature_values = feature_df[feature_column].dropna()
            mean_val = feature_values.mean()
            median_val = feature_values.median()
            std_val = feature_values.std()

            # Extract wavelet features
            dwt_features = extract_wavelet_features(feature_values)

            # Combine features and metadata into a single row
            row = {
                "recording_id": recording_id,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                **dwt_features,
                "user_id": user_info["user_id"],
                "session_number": meta["session_number"],
                "task": meta["task"],
                "hand": meta["hand"]
                # "affected_hand": user_info["affected_hand"],
                # "unaffected_hand": user_info["unaffected_hand"]
            }
            updated_rows.append(row)

        elif meta["hand"] == "bilateral":
            # Add two entries: one for affected hand and one for unaffected hand
            for hand_type in ["affected_hand", "unaffected_hand"]:
                feature_column = f"{user_info[hand_type].upper()}_{feature_prop}"
                if feature_column not in feature_df.columns:
                    continue

                feature_values = feature_df[feature_column].dropna()
                mean_val = feature_values.mean()
                median_val = feature_values.median()
                std_val = feature_values.std()

                # Extract wavelet features
                dwt_features = extract_wavelet_features(feature_values)

                # Combine features and metadata into a single row
                row = {
                    "recording_id": recording_id,
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    **dwt_features,
                    "user_id": user_info["user_id"],
                    "session_number": meta["session_number"],
                    "task": meta["task"],
                    "hand": hand_type.replace("_hand", "")  # 'affected' or 'unaffected'
                    # "affected_hand": user_info["affected_hand"],
                    # "unaffected_hand": user_info["unaffected_hand"]
                }
                updated_rows.append(row)

    # Append the new rows to the existing DataFrame
    updated_df = pd.DataFrame(updated_rows)
    combined_df = pd.concat([existing_df, updated_df], ignore_index=True)

    # Write the updated feature file
    combined_df.to_csv(feature_file_path, index=False)


def filter_recordings(participant_ids, assessment_ids, task_ids, hand):
    # Read the metadata from the excel file
    metadata_df = pd.read_excel('../features/video_recording_info.xlsx')

    metadata_df['participant_id'] = metadata_df['participant_id'].astype(str)
    metadata_df['assessment_id'] = metadata_df['assessment_id'].astype(str)
    metadata_df['task_id'] = metadata_df['task_id'].astype(str)
    metadata_df['hand'] = metadata_df['hand'].astype(str)
    # Apply filters
    if participant_ids:
        metadata_df = metadata_df[metadata_df['participant_id'].isin(participant_ids)]
    if assessment_ids:
        metadata_df = metadata_df[metadata_df['assessment_id'].isin(assessment_ids)]
    if task_ids:
        metadata_df = metadata_df[metadata_df['task_id'].isin(task_ids)]
    if hand:
        metadata_df = metadata_df[metadata_df['hand'].isin(hand)]
    # converrt recording id to strinng values. remove any decimal places
    metadata_df['recording_id'] = metadata_df['recording_id'].astype(str).str.split('.').str[0]

    return metadata_df['recording_id'].tolist()


def smooth_data_moving_average(series, window_size=3):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

def interactive_cluster_plot(recording_ids):
    feature_prop = 'WRIST_SPEED'  # Example feature prop
    feature_file_path = FEATURE_FILE_PATH.format(feature_prop=feature_prop.lower())
    recording_ids = list(map(int, recording_ids))

    # Read feature file
    df = pd.read_csv(feature_file_path)
    df = df[df['recording_id'].isin(recording_ids)]

    # Normalize the data
    features = df[[
        'mean', 'std', 'median',
        'approx_coeff_mean', 'approx_coeff_std',
        'detail1_coeff_mean', 'detail1_coeff_std',
        'detail2_coeff_mean', 'detail2_coeff_std',
        'detail3_coeff_mean', 'detail3_coeff_std',
        'energy_approx', 'energy_detail1', 'energy_detail2', 'energy_detail3',
        'entropy_approx', 'entropy_detail1', 'entropy_detail2', 'entropy_detail3',
        'smoothness_index',
        'peak_detail1', 'peak_detail2', 'peak_detail3',
        'crossings_detail1', 'crossings_detail2', 'crossings_detail3',
        'dominant_frequency_band', 'scale_averaged_wavelet_power',
        'correlation_approx_detail1', 'correlation_approx_detail2', 'correlation_approx_detail3',
        'duration_high_energy_approx', 'duration_high_energy_detail1', 'duration_high_energy_detail2',
        'duration_high_energy_detail3'
    ]]  # Updated feature columns with all new wavelet-based features

    features = features.fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # scaled_features = features
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=40, perplexity=8, learning_rate=500)
    tsne_results = tsne.fit_transform(scaled_features)

    # Clustering using KMeans and DBSCAN
    kmeans = KMeans(n_clusters=3, n_init=10).fit(tsne_results)  # Set n_init explicitly to suppress warning
    dbscan = DBSCAN(eps=0.5).fit(tsne_results)

    # Add t-SNE results and cluster labels to the DataFrame
    df['tsne_1'] = tsne_results[:, 0]
    df['tsne_2'] = tsne_results[:, 1]
    df['kmeans_label'] = kmeans.labels_
    df['dbscan_label'] = dbscan.labels_

    # Convert user_id and assessment_id to string for hover display
    df['user_id'] = df['user_id'].astype(str)
    df['task'] = df['task'].astype(str)
    df['hand'] = df['hand'].astype(str)

    df['label'] = df['user_id'] + ' ' + df['session_number'] + ' ' + df['task'].astype(str) + ' ' + df['hand']

    # Create interactive plot using Plotly Express
    fig = px.scatter(
        df,
        x='tsne_1',
        y='tsne_2',
        color='hand',
        text='label',  # Color by KMeans cluster label
        hover_data={
            'user_id': True,  # Show user ID on hover
            'kmeans_label': True,  # Show cluster label on hover
            'task': True,  # Sho,
            'recording_id': True
        },
        title="Interactive Clustering with Hover Info",
        labels={'kmeans_label': 'Cluster ID'}
    )
    fig.update_traces(textposition='top center')

    # Show the plot
    fig.show()


def extract_wavelet_features(values):
    # Inner function to compute energy of wavelet coefficients
    def compute_energy(coeffs):
        return np.sum(np.square(coeffs))

    # Inner function to compute entropy of wavelet coefficients
    def compute_entropy(coeffs):
        # Normalize the coefficients to get a probability distribution
        p = np.abs(coeffs) / np.sum(np.abs(coeffs))
        p = p[p > 0]  # Avoid log of zero
        entropy = -np.sum(p * np.log(p))
        return entropy

    # Inner function to calculate smoothness index
    def smoothness_index(approx_coeffs, detail_coeffs):
        return np.std(approx_coeffs) / (np.std(detail_coeffs) + 1e-5)  # Avoid division by zero

    # Inner function to compute zero crossings (for crossings between bands)
    def zero_crossings(coeffs):
        return np.sum(np.diff(np.sign(coeffs)) != 0)

    # Inner function to compute the dominant frequency band (scale with most energy)
    def dominant_frequency_band(energies):
        return np.argmax(energies)  # Index of scale with highest energy

    # Inner function to compute scale-averaged wavelet power
    def scale_averaged_wavelet_power(energies):
        return np.mean(energies)

    # Inner function to compute time-frequency correlation between scales
    def time_frequency_correlation(coeffs1, coeffs2):
        # Find the minimum length between the two sets of coefficients
        min_len = min(len(coeffs1), len(coeffs2))
        # Truncate both arrays to the same length
        truncated_coeffs1 = coeffs1[:min_len]
        truncated_coeffs2 = coeffs2[:min_len]
        # Compute the correlation
        return np.corrcoef(truncated_coeffs1, truncated_coeffs2)[0, 1]

    # if values is an empty array, return a dictionary with zeros
    if len(values) == 0:
        features = {
            'approx_coeff_mean': 0,
            'approx_coeff_std': 0,
            'detail1_coeff_mean': 0,
            'detail1_coeff_std': 0,
            'detail2_coeff_mean': 0,
            'detail2_coeff_std': 0,
            'detail3_coeff_mean': 0,
            'detail3_coeff_std': 0,
            'energy_approx': 0,
            'energy_detail1': 0,
            'energy_detail2': 0,
            'energy_detail3': 0,
            'entropy_approx': 0,
            'entropy_detail1': 0,
            'entropy_detail2': 0,
            'entropy_detail3': 0,
            'smoothness_index': 0,
            'peak_detail1': 0,
            'peak_detail2': 0,
            'peak_detail3': 0,
            'crossings_detail1': 0,
            'crossings_detail2': 0,
            'crossings_detail3': 0,
            'dominant_frequency_band': 0,
            'scale_averaged_wavelet_power': 0,
            'correlation_approx_detail1': 0,
            'correlation_approx_detail2': 0,
            'correlation_approx_detail3': 0,
            'duration_high_energy_approx': 0,
            'duration_high_energy_detail1': 0,
            'duration_high_energy_detail2': 0,
            'duration_high_energy_detail3': 0
        }
    else:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(values, 'db1', level=3)  # Daubechies wavelet

        # Approximation and detail coefficients
        cA = coeffs[0]  # Approximation coefficients at the highest level
        cD1 = coeffs[1]  # Detail coefficients at the first level
        cD2 = coeffs[2]  # Detail coefficients at the second level
        cD3 = coeffs[3]  # Detail coefficients at the third level

        # Calculate energies at each level
        energy_approx = compute_energy(cA)
        energy_detail1 = compute_energy(cD1)
        energy_detail2 = compute_energy(cD2)
        energy_detail3 = compute_energy(cD3)

        # Dominant frequency band
        energies = [energy_approx, energy_detail1, energy_detail2, energy_detail3]
        dominant_band = dominant_frequency_band(energies)

        # Scale-averaged wavelet power
        avg_power = scale_averaged_wavelet_power(energies)

        # Calculate correlation between approximation and details
        corr_approx_detail1 = time_frequency_correlation(cA, cD1)
        corr_approx_detail2 = time_frequency_correlation(cA, cD2)
        corr_approx_detail3 = time_frequency_correlation(cA, cD3)

        # Duration of high-energy periods (for simplicity, define high energy as > mean energy)
        high_energy_threshold_approx = np.mean(np.abs(cA))
        high_energy_threshold_detail1 = np.mean(np.abs(cD1))
        high_energy_threshold_detail2 = np.mean(np.abs(cD2))
        high_energy_threshold_detail3 = np.mean(np.abs(cD3))

        # Count the number of coefficients exceeding the threshold in each set
        duration_high_energy_approx = np.sum(np.abs(cA) > high_energy_threshold_approx)
        duration_high_energy_detail1 = np.sum(np.abs(cD1) > high_energy_threshold_detail1)
        duration_high_energy_detail2 = np.sum(np.abs(cD2) > high_energy_threshold_detail2)
        duration_high_energy_detail3 = np.sum(np.abs(cD3) > high_energy_threshold_detail3)
        # Extract features
        features = {
            'approx_coeff_mean': np.mean(cA),
            'approx_coeff_std': np.std(cA),
            'detail1_coeff_mean': np.mean(cD1),
            'detail1_coeff_std': np.std(cD1),
            'detail2_coeff_mean': np.mean(cD2),
            'detail2_coeff_std': np.std(cD2),
            'detail3_coeff_mean': np.mean(cD3),
            'detail3_coeff_std': np.std(cD3),

            # Energy of the approximation and detail coefficients
            'energy_approx': energy_approx,
            'energy_detail1': energy_detail1,
            'energy_detail2': energy_detail2,
            'energy_detail3': energy_detail3,

            # Entropy of the approximation and detail coefficients
            'entropy_approx': compute_entropy(cA),
            'entropy_detail1': compute_entropy(cD1),
            'entropy_detail2': compute_entropy(cD2),
            'entropy_detail3': compute_entropy(cD3),

            # Smoothness index (using approximation and first level of detail coefficients)
            'smoothness_index': smoothness_index(cA, cD1),

            # Peak values of the detail coefficients
            'peak_detail1': np.max(np.abs(cD1)),
            'peak_detail2': np.max(np.abs(cD2)),
            'peak_detail3': np.max(np.abs(cD3)),

            # Zero crossings in the detail coefficients (indicates frequency of oscillations)
            'crossings_detail1': zero_crossings(cD1),
            'crossings_detail2': zero_crossings(cD2),
            'crossings_detail3': zero_crossings(cD3),

            # Dominant frequency band (scale with the most energy)
            'dominant_frequency_band': dominant_band,

            # Scale-averaged wavelet power (mean energy across scales)
            'scale_averaged_wavelet_power': avg_power,

            # Correlations between time-frequency scales
            'correlation_approx_detail1': corr_approx_detail1,
            'correlation_approx_detail2': corr_approx_detail2,
            'correlation_approx_detail3': corr_approx_detail3,

            # Duration of high-energy periods
            'duration_high_energy_approx': duration_high_energy_approx,
            'duration_high_energy_detail1': duration_high_energy_detail1,
            'duration_high_energy_detail2': duration_high_energy_detail2,
            'duration_high_energy_detail3': duration_high_energy_detail3
        }

    return features


def run_tsne():
    # stroke survivors ['7001','7002','7101','7102''7103','7104','7105','7106','7107','7202','7203','7204']
    # heallthy participants ['6101','6102','6103']
    # 'Good stroke survivor sample' : ['7101','7103','7002']
    participant_ids = ['7101', '7103', '7002']
    # participant_ids =['7001', '7002', '7101', '7102''7103', '7104', '7105', '7106', '7107', '7202', '7203', '7204']
    assessment_ids = []
    task_ids = ['DW']
    hand = []

    # Filter recordings
    recording_ids = filter_recordings(participant_ids, assessment_ids, task_ids, hand)

    # Extract features
    extract_time_series_features(recording_ids, 'WRIST_SPEED')

    # Perform clustering
    # cluster(recording_ids)
    interactive_cluster_plot(recording_ids)


if __name__ == "__main__":
    run_tsne()




