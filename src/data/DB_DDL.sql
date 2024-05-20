-- DROP SCHEMA dbo;

CREATE SCHEMA dbo;
-- VisionAnalysis.dbo.frame_coordinates definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.frame_coordinates;

CREATE TABLE VisionAnalysis.dbo.frame_coordinates (
	id int IDENTITY(1,1) NOT NULL,
	recording_id int NULL,
	frame_rate int NULL,
	coordinates nvarchar(MAX) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	coordinates_object1 nvarchar(MAX) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	coordinates_object2 nvarchar(MAX) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	CONSTRAINT PK__frame_co__3213E83F5658F15F PRIMARY KEY (id)
);


-- VisionAnalysis.dbo.frame_feature_files definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.frame_feature_files;

CREATE TABLE VisionAnalysis.dbo.frame_feature_files (
	id int IDENTITY(1,1) NOT NULL,
	frame_coordinate_id int NULL,
	recording_id int NULL,
	frame_rate int NULL,
	file_name varchar(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	CONSTRAINT PK__frame_feature_files PRIMARY KEY (id)
);


-- VisionAnalysis.dbo.frame_features definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.frame_features;

CREATE TABLE VisionAnalysis.dbo.frame_features (
	id uniqueidentifier DEFAULT newid() NOT NULL,
	frame_coordinate_id int NULL,
	frame_seq_number int NULL,
	recording_id int NULL,
	frame_rate int NULL,
	right_elbow_speed float NULL,
	right_wrist_speed float NULL,
	left_elbow_speed float NULL,
	left_wrist_speed float NULL,
	hand_bounding_box_iou float NULL,
	right_wrist_acceleration float NULL,
	right_elbow_acceleration float NULL,
	left_wrist_acceleration float NULL,
	left_elbow_acceleration float NULL,
	right_wrist_jerk float NULL,
	right_elbow_jerk float NULL,
	left_wrist_jerk float NULL,
	left_elbow_jerk float NULL,
	left_index_finger_angle_wrist_mcp_pip float NULL,
	left_middle_finger_angle_wrist_mcp_pip float NULL,
	left_ring_finger_angle_wrist_mcp_pip float NULL,
	left_pinky_angle_wrist_mcp_pip float NULL,
	left_index_finger_angle_mcp_pip_dip float NULL,
	left_middle_finger_angle_mcp_pip_dip float NULL,
	left_ring_finger_angle_mcp_pip_dip float NULL,
	left_pinky_angle_mcp_pip_dip float NULL,
	left_index_finger_angle_pip_dip_tip float NULL,
	left_middle_finger_angle_pip_dip_tip float NULL,
	left_ring_finger_angle_pip_dip_tip float NULL,
	left_pinky_angle_pip_dip_tip float NULL,
	left_index_finger_angle_wrist_mcp_tip float NULL,
	left_middle_finger_angle_wrist_mcp_tip float NULL,
	left_ring_finger_angle_wrist_mcp_tip float NULL,
	left_pinky_angle_wrist_mcp_tip float NULL,
	left_index_finger_ratio_mcp_tip_distal float NULL,
	left_middle_finger_ratio_mcp_tip_distal float NULL,
	left_ring_finger_ratio_mcp_tip_distal float NULL,
	left_pinky_ratio_mcp_tip_distal float NULL,
	right_index_finger_angle_wrist_mcp_pip float NULL,
	right_middle_finger_angle_wrist_mcp_pip float NULL,
	right_ring_finger_angle_wrist_mcp_pip float NULL,
	right_pinky_angle_wrist_mcp_pip float NULL,
	right_index_finger_angle_mcp_pip_dip float NULL,
	right_middle_finger_angle_mcp_pip_dip float NULL,
	right_ring_finger_angle_mcp_pip_dip float NULL,
	right_pinky_angle_mcp_pip_dip float NULL,
	right_index_finger_angle_pip_dip_tip float NULL,
	right_middle_finger_angle_pip_dip_tip float NULL,
	right_ring_finger_angle_pip_dip_tip float NULL,
	right_pinky_angle_pip_dip_tip float NULL,
	right_index_finger_angle_wrist_mcp_tip float NULL,
	right_middle_finger_angle_wrist_mcp_tip float NULL,
	right_ring_finger_angle_wrist_mcp_tip float NULL,
	right_pinky_angle_wrist_mcp_tip float NULL,
	right_index_finger_ratio_mcp_tip_distal float NULL,
	right_middle_finger_ratio_mcp_tip_distal float NULL,
	right_ring_finger_ratio_mcp_tip_distal float NULL,
	right_pinky_ratio_mcp_tip_distal float NULL,
	left_grip_aperture float NULL,
	right_grip_aperture float NULL,
	left_wrist_flexion_extension_angle float NULL,
	right_wrist_flexion_extension_angle float NULL,
	left_elbow_flexion_angle float NULL,
	right_elbow_flexion_angle float NULL,
	left_shoulder_abduction_angle float NULL,
	right_shoulder_abduction_angle float NULL,
	object_1_speed float NULL,
	object_2_speed float NULL,
	object_1_trajectory_deviation float NULL,
	object_1_left_hand_iou float NULL,
	object_1_right_hand_iou float NULL,
	CONSTRAINT PK__frame_fe__3213E83F7F651EBF PRIMARY KEY (id)
);
 CREATE NONCLUSTERED INDEX idx_frame_coordinate_id ON dbo.frame_features (  frame_coordinate_id ASC  )
	 WITH (  PAD_INDEX = OFF ,FILLFACTOR = 100  ,SORT_IN_TEMPDB = OFF , IGNORE_DUP_KEY = OFF , STATISTICS_NORECOMPUTE = OFF , ONLINE = OFF , ALLOW_ROW_LOCKS = ON , ALLOW_PAGE_LOCKS = ON  )
	 ON [PRIMARY ] ;


-- VisionAnalysis.dbo.recording definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.recording;

CREATE TABLE VisionAnalysis.dbo.recording (
	id int IDENTITY(1,1) NOT NULL,
	task nvarchar(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	[date] date NULL,
	[time] time NULL,
	flipped bit NULL,
	session_id int NULL,
	file_name nvarchar(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	hand nvarchar(16) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	CONSTRAINT PK__recordin__3213E83FD764ECD0 PRIMARY KEY (id)
);


-- VisionAnalysis.dbo.[session] definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.[session];

CREATE TABLE VisionAnalysis.dbo.[session] (
	id int IDENTITY(1,1) NOT NULL,
	session_type nvarchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	session_number int NULL,
	user_id int NULL,
	CONSTRAINT PK__session__3213E83F891AACB4 PRIMARY KEY (id)
);


-- VisionAnalysis.dbo.[user] definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.[user];

CREATE TABLE VisionAnalysis.dbo.[user] (
	id int IDENTITY(1,1) NOT NULL,
	name nvarchar(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	age int NULL,
	dominant_hand nvarchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	gender nvarchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	affected_hand nvarchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	non_affected_hand nvarchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
	CONSTRAINT PK__user__3213E83F7A68A2F0 PRIMARY KEY (id)
);


-- VisionAnalysis.dbo.video_features definition

-- Drop table

-- DROP TABLE VisionAnalysis.dbo.video_features;

CREATE TABLE VisionAnalysis.dbo.video_features (
	id int IDENTITY(1,1) NOT NULL,
	frame_coordinate_id int NULL,
	right_max_wrist_speed float NULL,
	left_max_wrist_speed float NULL,
	right_wrist_time_to_peak_velocity float NULL,
	left_wrist_time_to_peak_velocity float NULL,
	recording_id int NULL,
	right_wrist_number_of_velocity_peaks float NULL,
	left_wrist_number_of_velocity_peaks float NULL,
	right_wrist_number_of_direction_changes float NULL,
	left_wrist_number_of_direction_changes float NULL,
	right_wrist_rate_of_direction_changes float NULL,
	left_wrist_rate_of_direction_changes float NULL,
	completion_time float NULL,
	right_wrist_max_speed float NULL,
	right_wrist_avg_speed float NULL,
	right_wrist_std_speed float NULL,
	right_wrist_q1_speed float NULL,
	right_wrist_q2_speed float NULL,
	right_wrist_q3_speed float NULL,
	left_wrist_max_speed float NULL,
	left_wrist_avg_speed float NULL,
	left_wrist_std_speed float NULL,
	left_wrist_q1_speed float NULL,
	left_wrist_q2_speed float NULL,
	left_wrist_q3_speed float NULL,
	right_wrist_total_traversed_distance float NULL,
	left_wrist_total_traversed_distance float NULL,
	total_trajectory_error float NULL,
	CONSTRAINT PK__video_fe__3213E83FEC273EB8 PRIMARY KEY (id)
);
