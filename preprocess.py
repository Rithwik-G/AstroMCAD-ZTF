import pandas as pd
import numpy as np
import pickle


def load_data(file_path, ntimesteps=656):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)  # Remove rows with NaN values
    data.sort_values(by='mjd', inplace=True)  # Sort by 'mjd' column
    times = (data['mjd'] - data['mjd'].min()) / 100 # Normalize time
    flux = data['flux'] / 500
    ferr = data['fluxerr'] / 500
    # print(np.unique(data['filter']))
    filter = data['filter'].map({'atlaso': 0.679, 'atlasc' : 0.533, 'ztfr' : 0.626, 'ztfg': 0.472, 'sdssr' : 0.616})

    output = np.array([filter, times, flux, ferr]).T

    output = np.pad(output, ((0, ntimesteps-len(output)), (0, 0)), mode='constant', constant_values=0)

    return np.array([output])
    

if (__name__ == "__main__"):
    data_path = 'data/example.csv'
    data = load_data(data_path)

    print(data)
    print(type(data))
    with open('data/example.pickle', 'wb') as f:
        pickle.dump(data, f)