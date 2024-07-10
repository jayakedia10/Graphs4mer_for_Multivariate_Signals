import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from constants import TUH_CHANNELS, TUH_FREQUENCY
from data.data_utils.eeg_data_utils import (
    resampleData,
    getEDFsignals,
    getOrderedChannels,
)
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py
import scipy



def normalize_channel_names(channels):
    return [ch.replace('-REF', '') for ch in channels]

def resample_all(raw_edf_dir, save_dir):
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if name.endswith(".edf"):
                edf_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]
        relative_path = os.path.relpath(edf_fn, raw_edf_dir)
        save_fn = os.path.join(save_dir, relative_path).replace(".edf", ".h5")
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)  # Ensure directory exists

        if os.path.exists(save_fn):
            print(f"{save_fn} exists. Skipping...")
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)
            orderedChannels = getOrderedChannels(edf_fn, False, f.getSignalLabels(), TUH_CHANNELS)
            normalized_orderedChannels = normalize_channel_names(orderedChannels)
            signals = getEDFsignals(f)
            signal_array = np.array(signals[normalized_orderedChannels, :])
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != TUH_FREQUENCY:
                signal_array = resampleData(
                    signal_array,
                    to_freq=TUH_FREQUENCY,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=TUH_FREQUENCY)
        except BaseException as e:
            print(f"Failed to process file: {edf_fn} with error: {e}")
            failed_files.append(edf_fn)

    print(f"DONE. {len(failed_files)} files failed.")

if __name__ == "__main__":
    raw_edf_dir = r"c:\\Users\\Jaya Kedia\\Desktop\\EEG_DATA\\EEG_GNN_SSL\\edf"
    save_dir = "resampled_signals2"
    resample_all(raw_edf_dir, save_dir)
