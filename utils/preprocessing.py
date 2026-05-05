import numpy as np

from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes, binary_closing


def preprocess_mask(mask: np.ndarray):
    '''
    Preprocessing mask
    '''
    filled = binary_fill_holes(mask).astype(np.uint8)

    preprocessed = binary_closing(filled, iterations=1).astype(np.uint8)

    # delete noise
    lbl, n = label(preprocessed)
    for i in range(1, n + 1):
        if (lbl == i).sum() < 10:
            preprocessed[lbl == i] = 0
            
    return preprocessed