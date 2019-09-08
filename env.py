from skimage.color import rgb2gray
from skimage.transform import resize
import gym


def preprocess(self, frame, last_frame):
    frame = np.maximum(frame, last_frame)
    frame = np.uint8(resize(rgb2gray(frame), (84,84)) * 255)
    frame = np.reshape(frame, (1, 84,84))
    return frame


class PongEnv:
    def __init__(self):
        self.env = gym.make('Pong-v0')
    
    def reset(self):
