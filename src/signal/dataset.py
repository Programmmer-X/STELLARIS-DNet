"""Dataset utilities for signal processing."""

import pandas as pd

class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load(self):
        self.data = pd.read_csv(self.filepath)
        return self.data