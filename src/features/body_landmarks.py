
import enum
from typing import NamedTuple

import numpy as np

class BodyLandmark(enum.IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    LEFT_THUMB_CMC = 16
    LEFT_THUMB_MCP = 17
    LEFT_THUMB_IP = 18
    LEFT_THUMB_TIP = 19
    LEFT_INDEX_FINGER_MCP = 20
    LEFT_INDEX_FINGER_PIP = 21
    LEFT_INDEX_FINGER_DIP = 22
    LEFT_INDEX_FINGER_TIP = 23
    LEFT_MIDDLE_FINGER_MCP = 24
    LEFT_MIDDLE_FINGER_PIP = 25
    LEFT_MIDDLE_FINGER_DIP = 26
    LEFT_MIDDLE_FINGER_TIP = 27
    LEFT_RING_FINGER_MCP = 28
    LEFT_RING_FINGER_PIP = 29
    LEFT_RING_FINGER_DIP = 30
    LEFT_RING_FINGER_TIP = 31
    LEFT_PINKY_MCP = 32
    LEFT_PINKY_PIP = 33
    LEFT_PINKY_DIP = 34
    LEFT_PINKY_TIP = 35
    RIGHT_WRIST = 36
    RIGHT_THUMB_CMC = 37
    RIGHT_THUMB_MCP = 38
    RIGHT_THUMB_IP = 39
    RIGHT_THUMB_TIP = 40
    RIGHT_INDEX_FINGER_MCP = 41
    RIGHT_INDEX_FINGER_PIP = 42
    RIGHT_INDEX_FINGER_DIP = 43
    RIGHT_INDEX_FINGER_TIP = 44
    RIGHT_MIDDLE_FINGER_MCP = 45
    RIGHT_MIDDLE_FINGER_PIP = 46
    RIGHT_MIDDLE_FINGER_DIP = 47
    RIGHT_MIDDLE_FINGER_TIP = 48
    RIGHT_RING_FINGER_MCP = 49
    RIGHT_RING_FINGER_PIP = 50
    RIGHT_RING_FINGER_DIP = 51
    RIGHT_RING_FINGER_TIP = 52
    RIGHT_PINKY_MCP = 53
    RIGHT_PINKY_PIP = 54
    RIGHT_PINKY_DIP = 55
    RIGHT_PINKY_TIP = 56