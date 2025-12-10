from collections import deque, namedtuple
import cv2
import numpy as np

def preprocess_frame(frame):
    """Converts frame to grayscale and resizes it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.float32) / 255.0

frame_stack_size = 4
stacked_frames = deque(maxlen=frame_stack_size)

def stack_frames(state, is_new_episode):
    """Stacks frames. Handles initial frames."""
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames.clear()
        for _ in range(frame_stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state 