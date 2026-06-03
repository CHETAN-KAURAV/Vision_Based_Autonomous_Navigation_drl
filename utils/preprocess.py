# utils/preprocess.py

import numpy as np

def preprocess_state(obs):

    img = obs["image"]

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))

    return img