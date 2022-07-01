from glob import glob
import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("started loading artifacts...")
    global __locations
    global __data_columns
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data-columns']
        # In data columns first 3 columns are sqft, bath, bhk
        __locations = __data_columns[3:]
    
    if __model is None:
        with open('./artifacts/bangalore_home_price_prediction.pickle', 'rb') as f:
            __model = pickle.load(f)
    
    print("loaded saved artfacts...")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
