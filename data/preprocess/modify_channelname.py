import sys
import os
import pyedflib
from constants import TUH_CHANNELS

sys.path.append('C:/Users/Jaya Kedia/Downloads/graphs4mer-main/graphs4mer-main/')

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
