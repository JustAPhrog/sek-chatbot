import json


def import_data(file_path):
    with open(file_path, 'r') as f:
        file_data = json.load(f)
    return file_data
