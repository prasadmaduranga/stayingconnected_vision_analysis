# import visuaize.py

from src.visualization import visualize

from visualize import (
    visualise_speed_profile_effected_vs_uneffected,
    visualise_speed_profile_effected_vs_uneffected,
    visualise_speed_profile_healthy_vs_stroke,
    visualise_speed_profile_vs_grip_aperture,
    visualise_acceleration_profile,
    visualise_acceleration_vs_speed_profile,
    visualise_elbow_speed_profile,
    visualise_elbow_speed_vs_wrist_speed_profile,
    visualise_elbow_acceleration_profile,
    visualise_wrist_flexion_extentin_angle,
    visualise_wrist_flexion_extention_angle_vs_speed,
    visualise_elbow_flexion_angle,
    visualise_elbow_flexion_angle_vs_speed,
    visualise_shoulder_abduction_angle,
    visulise_finger_density_function,
    visualise_object_speed_profile,
    visualise_object_tracjectory_deviation,
    visualise_hand_object_iou,
    visualise_hand_object_iou_vs_speed,
    visualise_hand_ious,
    visualise_speed_distribution,
    print_kinematic_stats
)
from util import DatabaseUtil
output_dir_path = "output/individual_profiling"


def read_recording_metadata(recording_ids):
    db_util = DatabaseUtil()

    metadata = {}
    for recording_id in recording_ids:
        query = """
            SELECT r.id AS recording_id, u.id AS user_id, u.affected_hand, u.non_affected_hand,
                   s.session_number, r.task, r.hand
            FROM recording r
            JOIN session s ON r.session_id = s.id
            JOIN [user] u ON s.user_id = u.id
            WHERE r.id = ?
        """
        params = (recording_id,)
        result = db_util.fetch_data(query, params)

        if result:
            row = result[0]
            metadata[recording_id] = {
                "user": {
                    "user_id": row.user_id,
                    "affected_hand": row.affected_hand,
                    "unaffected_hand": row.non_affected_hand
                },
                "session_number": row.session_number,
                "task": row.task,
                "hand": row.hand # affected or unaffected or bilateral
            }

    return metadata


def visualise_profile(recording_ids,draw_wavelts=True):
    recording_metadata = read_recording_metadata(recording_ids)


    # ***** visualise features of the video in separate graphs *****
    features = ['WRIST_SPEED', 'GRIP_APERTURE', 'ELBOW_SPEED', 'WRIST_FLEXION_EXTENSION_ANGLE', 'ELBOW_FLEXION_ANGLE',
                'SHOULDER_ABDUCTION_ANGLE']
    # hey value dictionary , key is the feature and value is the window_size for that feature
    feature_window_size = {
        'WRIST_SPEED': 4,
        'GRIP_APERTURE': 4,
        'ELBOW_SPEED': 4,
        'WRIST_FLEXION_EXTENSION_ANGLE': 4,
        'ELBOW_FLEXION_ANGLE': 1,
        'SHOULDER_ABDUCTION_ANGLE': 1
    }


    # Iterate over features and create config for each
    for feature in features:
        graph_config = {
            'params': [],
            'window_size': feature_window_size[feature],
            'xlabel': 'Sequence Number',
            'ylabel': feature.replace('_', ' ').title(),
            'title': f'{feature.replace("_", " ").title()} Profile'
        }

        # Add parameters for each recording_id
        for recording_id in recording_ids:
            metadata = recording_metadata[recording_id]
            if metadata['hand'] == 'affected' or metadata['hand'] == 'unaffected':
                performing_hand = metadata['user']['affected_hand'].upper() if metadata['hand'] == 'affected' else \
                metadata['user']['unaffected_hand'].upper()
                label = f"{metadata['user']['user_id']}_{metadata['session_number']}_{metadata['task']}_{performing_hand.lower()}_{metadata['hand']}"
                graph_config['params'].append({
                    'recording_id': recording_id,
                    'feature': f'{performing_hand}_{feature}',
                    'label': label,
                    'participant': metadata['user']['user_id']
                })
            elif metadata['hand'] == 'bilateral':
                for hand in ['LEFT', 'RIGHT']:
                    affected_status = 'unaffected' if hand.lower() == metadata['user'][
                        'unaffected_hand'].lower() else 'affected'
                    label = f"{metadata['user']['user_id']}_{metadata['session_number']}_{metadata['task']}_{hand.lower()}_{metadata['hand']}_{affected_status}"
                    graph_config['params'].append({
                        'recording_id': recording_id,
                        'feature': f'{hand}_{feature}',
                        'label': label,
                        'participant': metadata['user']['user_id']
                    })



        # Call the appropriate visualization method based on the feature
        visualize.visualize_data([graph_config],distribution_boxplots=True)
        if draw_wavelts:
            visualize.draw_wavelet_transform([graph_config])
            # visualize.draw_fourier_transform([graph_config])

    # ***** visualise video level kinematics statistics *****
    videoKinematicMetaData = {
        'params': [
            {
                'recording_id': recording_id,
                'hand': metadata['user']['affected_hand'].upper() if metadata['hand'] == 'affected' else metadata['user']['unaffected_hand'].upper(),
                'participant': metadata['user']['user_id']
            }
            for recording_id in recording_ids
        ],
        'hand_specific_features': ['wrist_number_of_velocity_peaks', 'wrist_number_of_direction_changes', 'wrist_rate_of_direction_changes', 'wrist_total_traversed_distance'],
        'general_features': ['total_trajectory_error', 'completion_time']
    }

    # Call print_kinematic_stats with the constructed metadata object
    visualize.print_kinematic_stats(metaData=videoKinematicMetaData)


def visualise_profile_finger_movement_density_analysis(recording_ids):
    recording_metadata = read_recording_metadata(recording_ids)
    fingers = ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']

    # Iterate over recording ids and create config for each
    for recording_id in recording_ids:
        graph_config = {
            'params': [],
            'window_size': 1,  # Adjust if needed
            'xlabel': 'Frame Sequence Number',
            'ylabel': 'Finger Density',
            'title': f'Finger Movement Density Analysis for Recording ID {recording_id}'
        }

        metadata = recording_metadata[recording_id]
        hands = []

        if metadata['hand'] in ['affected', 'unaffected']:
            hands.append(
                metadata['user']['affected_hand'].upper() if metadata['hand'] == 'affected' else metadata['user'][
                    'unaffected_hand'].upper())
        elif metadata['hand'] == 'bilateral':
            continue

        # Add parameters for each hand and finger
        for hand in hands:
            for finger in fingers:
                affected_status = 'unaffected' if hand.lower() == metadata['user'][
                    'unaffected_hand'].lower() else 'affected'
                label = f"{metadata['user']['user_id']}_{metadata['session_number']}_{metadata['task']}_{hand.lower()}_{affected_status}_{finger}"
                graph_config['params'].append({
                    'recording_id': recording_id,
                    'hand': hand,
                    'finger': finger,
                    'label': label,
                    'participant': metadata['user']['user_id']
                })

        # Call the finger density visualization method
        visualize.visualize_data_polar_coordinates([graph_config])



if __name__ == "__main__":

    # Experiment 1:
    # 7001 , DW, , efffeced
    # visualise_profile([3263,3262])

    # T&CI600
    # Experiment 2:
    # 7001 , DW, , efffeced,unefffeced
    # visualise_profile([ 3263,3197, 3204, 3211, 3262, 3196, 3203, 3210])
    # 7002 , DW, , efffeced,unefffeced
    # visualise_profile([3252, 3261,3254, 3251,3260,3253])
    # 7001 , FPT, , efffeced,unefffeced
    # visualise_profile([3248,3200,3216])


    # 7002 , DW, , efffeced,unefffeced
    # visualise_profile([2173, 2175, 2172, 2190])
    # visualise_profile([3232, 3233])

    # all users , affected hand , dw
    # visualise_profile([2175, 3197, 3220, 3228,3236])
    # Experiment 3:
    # all users , DW, , efffeced,unefffeced
    visualise_profile([3252, 3197, 3220, 3228, 3236 ,3251, 3196, 3219,3229,3235])


    # read_recording_metadata([1062, 1063, 1065])
    #
    # visualise_profile([2177,2173,2175])
    # visualise_profile([2172, 2173])
    # visualise_profile_finger_movement_density_analysis([1063,1062]) # only for finger movement activities

    # visualise_profile_wavelet_analysis([1062, 1063, 2128])


