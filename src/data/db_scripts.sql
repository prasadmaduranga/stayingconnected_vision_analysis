


-- Insert user
INSERT INTO VisionAnalysis.dbo.[user]
(id, name, age, dominant_hand, gender, affected_hand, non_affected_hand)
VALUES
(7203, 'AK', 0,'left', 'female', 'right', 'left')


INSERT INTO session (session_type, session_number, user_id)
VALUES ('vision_assessment', 1, 1);

INSERT INTO recording (task, date, time, flipped, session_id)
VALUES ('drinking_water', CAST(GETDATE() AS DATE), CAST(GETDATE() AS TIME), 0, 1);



INSERT INTO VisionAnalysis.dbo.[user]
(id, name, age, dominant_hand, gender, affected_hand, non_affected_hand)
VALUES
(7203, 'AK', 0,'left', 'female', 'right', 'left'),
(7204, 'KR',0, 'right', 'male', 'left', 'right'),
(7104, 'MW',0, 'right', 'male', 'right', 'left'),
(7105, 'JW',0, 'right', 'male', 'right', 'left'),
(7106, 'DR',0, 'right', 'male', 'left', 'right'),
(7107, 'GW',0, 'right', 'male', 'right', 'left'),
(6101, 'IS',40, 'right', 'male', 'right', 'left'),
(6102, 'PA',30, 'right', 'male', 'right', 'left'),
(6103, 'PM',32, 'right', 'male', 'right', 'left')

INSERT INTO session (session_type, session_number, user_id)
VALUES
('vision_assessment', 'E1', 7202),
('vision_assessment', 'E1', 7203),
    ('vision_assessment', 'E1',7204 ),
    ('vision_assessment', 'P1',7101 ),
    ('vision_assessment', 'R1',7101 ),
    ('vision_assessment', 'S1', 7101),
    ('vision_assessment', 'P1', 7102),
    ('vision_assessment', 'S1', 7102 ),
    ('vision_assessment', 'P1', 7103),
    ('vision_assessment', 'P2', 7103),
    ('vision_assessment', 'E1', 7104),
    ('vision_assessment', 'S1', 7104),
    ('vision_assessment', 'P1', 7105 ),
    ('vision_assessment', 'P2', 7105 ),
    ('vision_assessment', 'S1', 7105 ),
    ('vision_assessment', 'E1', 7106 ),
    ('vision_assessment', 'P1', 7106 ),
    ('vision_assessment', 'S1', 7106 ),
    ('vision_assessment', 'P1', 7107 )
('vision_assessment', 'E1', 6101 ),
('vision_assessment', 'E2', 6101 ),
('vision_assessment', 'E3', 6101 ),
('vision_assessment', 'E1', 6102 ),
('vision_assessment', 'E2', 6102 ),
('vision_assessment', 'E3', 6102 ),
('vision_assessment', 'E1', 6103 ),
('vision_assessment', 'E2', 6103 ),
('vision_assessment', 'E3', 6103 ),






