import sys 
import multiprocessing
sys.path.append('C:/Users/Jaya Kedia/Downloads/graphs4mer-main/graphs4mer-main/')
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
    """
    Normalize channel names by removing '-REF' suffix.
    """
    return [ch.replace('-REF', '') for ch in channels]

def print_channel_info(edf_fn):
    """
    Function to print and compare channel names in the EDF file with TUH_CHANNELS.
    """
    try:
        f = pyedflib.EdfReader(edf_fn)
        edf_channels = f.getSignalLabels()
        normalized_edf_channels = normalize_channel_names(edf_channels)
        
        print(f"EDF File: {edf_fn}")
        print(f"EDF Channels: {edf_channels}")
        print(f"Normalized EDF Channels: {normalized_edf_channels}")
        print(f"TUH Channels: {TUH_CHANNELS}")

        missing_channels = [ch for ch in TUH_CHANNELS if ch not in normalized_edf_channels]
        extra_channels = [ch for ch in normalized_edf_channels if ch not in TUH_CHANNELS]

        if missing_channels:
            print(f"Missing Channels in EDF: {missing_channels}")
        if extra_channels:
            print(f"Extra Channels in EDF: {extra_channels}")

        if not missing_channels and not extra_channels:
            print("All channels match correctly.")

    except Exception as e:
        print(f"Failed to read file: {edf_fn} with error: {e}")

if __name__ == "__main__":
    raw_edf_dir = r"c:\\Users\\Jaya Kedia\\Desktop\\EEG_DATA\\EEG_GNN_SSL\\edf"
    sample_edf_file = os.path.join(raw_edf_dir, "train", "aaaaagpk", "s013_2014_06_18", "01_tcp_ar", "aaaaagpk_s013_t000.edf")
    print_channel_info(sample_edf_file)


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
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :])
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
    # parser = argparse.ArgumentParser("Resample.")
    # parser.add_argument(
    #     "--raw_edf_dir",
    #     type=str,
    #     default=None,
    #     help="Full path to raw edf files.",
    # )
    # parser.add_argument(
    #     "--save_dir",
    #     type=str,
    #     default=None,
    #     help="Full path to dir to save resampled signals.",
    # )
    # args = parser.parse_args()

    # resample_all(args.raw_edf_dir, args.save_dir)
    raw_edf_dir =  "c:/Users/Jaya Kedia/Desktop/EEG_DATA/EEG_GNN_SSL/edf/"
    save_dir = "resampled_signals1"
    resample_all(raw_edf_dir, save_dir)
