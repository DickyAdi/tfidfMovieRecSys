import pandas as pd
from typing import List
import joblib

master = pd.read_pickle('apps/data/processed/data.pkl')

def getAllMoviesTitle() -> List:
    """Return all movies titles within the database

    Returns:
        List: All available movies in the database
    """
    return master['original_title'].tolist()