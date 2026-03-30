"""Signal preprocessing utilities."""

import numpy as np

def normalize(data):
    """Normalize signal data."""
    return (data - np.mean(data)) / np.std(data)