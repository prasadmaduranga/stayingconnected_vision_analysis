
import sys

import pickle
import os
sys.path.append('../')
from util import DatabaseUtil

from dotenv import load_dotenv

load_dotenv()


def read_feature_file(frame_coordinate_id=None, recording_id=None):
    """
    Fetches the filename from the database for the given frame_coordinate_id and/or recording_id,
    reads the pickle file, and returns the DataFrame.
    """
    db_util = DatabaseUtil()
    pickle_base_path = os.path.join(os.getenv('PROJECT_PATH'), 'data/processed/frame_features')

    # Start building the SQL query dynamically based on provided arguments
    fetch_query = """
    SELECT file_name FROM VisionAnalysis.dbo.frame_feature_files
    WHERE 1=1
    """
    query_params = []

    if frame_coordinate_id is not None:
        fetch_query += " AND frame_coordinate_id = ?"
        query_params.append(frame_coordinate_id[0])

    if recording_id is not None:
        fetch_query += " AND recording_id = ?"
        query_params.append(recording_id[0])

    # Fetch the file name from the database
    file_name = db_util.fetch_data(fetch_query, tuple(query_params))

    # Assuming fetch_data returns a list of tuples and we only need the first result
    if file_name and len(file_name) > 0:
        file_name = file_name[0][0]  # Adjust indexing based on actual return structure
        file_path = os.path.join(pickle_base_path, file_name)

        # Load the pickle file into a DataFrame
        with open(file_path, 'rb') as file:
            df = pickle.load(file)

        return df
    else:
        print("File not found for the given identifiers.")
        return None