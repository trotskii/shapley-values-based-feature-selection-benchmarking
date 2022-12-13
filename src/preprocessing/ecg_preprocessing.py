import numpy as np
import wfdb 
import os 
from typing import Tuple, Union
import stumpy 
from scipy import stats
from wfdb import processing
import pandas as pd

def select_annotation(labels: np.ndarray, samples: np.ndarray, index: int, prev_index: int) -> float:
    """
    Select appropriate annotations for selected window.
    Arguments:
        labels - labels from annotations
        samples - indexes for the labels
        index - end index for the window
        prev_index - start index for the window
    Returns:
        float - 0 if no arrhythmia labels, 1 if there was at least on arrhythmia or other anomaly label
    """

    included_annotations = labels[np.where((samples<index) & (samples>=prev_index))]
    return 0.0 if np.all(included_annotations=='N') else 1.0 # return 0 (no arrhythmia) if all labels are N (normal), otherwise return 1

def format_timeseries(record: wfdb.Record, annotation: wfdb.Annotation, window_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, np.ndarray]]:
    """
    Split continious ECG signal into windows.
    Arguments:
        record - records object with the ECG signals
        annotation - annotations object
        window_length - length of the desired window in seconds
    Returns:
        high_arr - 2d array of the splitted high channel from ECG record
        low_arr - 2d array of the splitted low channel from ECG record
        labels_arr - array of labels for each window
        background - tuple with age, gender (male=0, female=1) and list of medication
    """
    fs = record.__dict__['fs']

    age, gender = record.__dict__['comments'][0].split(' ')[:2]
    medication = [s.strip() for s in record.__dict__['comments'][1].split(',')]
    age = float(age)
    gender = 0.0 if gender=='M' else 1.0

    window_length_samples = fs*window_length
    signal = record.p_signal
    high, low = signal[:, 0], signal[:,1] # we have 2 channels in this dataset

    matrix_profile = stumpy.stump(high.astype(float), int(fs))[:, 0]
    N = matrix_profile.shape[0]%window_length_samples
    high = high[:matrix_profile.shape[0]]
    low = low[:matrix_profile.shape[0]]
    high = high[:-N]
    low = low[:-N]
    matrix_profile = matrix_profile[:-N]


    labels = np.array(annotation.symbol)
    labels[labels=='+'] = 'N' # plus indicates start of the sequence, mark it as normal
    samples = np.array(annotation.sample)

    

    split_idx_arr = np.arange(window_length_samples, high.shape[0], window_length_samples)
    high_arr = np.array(np.split(high, split_idx_arr))
    low_arr = np.array(np.split(low, split_idx_arr))
    mp_arr = np.array(np.split(matrix_profile, split_idx_arr))

    labels_arr = np.zeros(high_arr.shape[0])
    prev_index = 0
    for idx, split_idx in enumerate(split_idx_arr):
        labels_arr[idx] = select_annotation(labels, samples, split_idx, prev_index)
        prev_index = split_idx

    

    assert(high_arr.shape[0] == labels_arr.shape[0])

    
    return high_arr, low_arr, mp_arr, labels_arr, (age, gender, np.array(medication))

def parse_all_records(path: os.PathLike, window_length: int) -> Tuple[np.ndarray, dict]:
    """
    Parse all records in path.
    Arguments:
        path - path to folder with data files
        window_length - window length in seconds to split records into
        use_lower_channel - include or discard lower channel from the dataset (default False = discard)
    Returns:
        dataset_list - np.array with the parsed dataset
        description - dict with index ranges for channels, background info and labels for easy selection. Indices depend on window length and sampling freq
    """
    files = list(set([f[:3] for f in os.listdir(path) if f[:3].isnumeric()]))

    files = files[10:30]

    dataset_list = []

    total = len(files)

    for idx, file in enumerate(files):
        print(f'Processing {idx+1} out of {total}')
        file_path = os.path.join(path, file)
        record = wfdb.rdrecord(file_path)
        annotations = wfdb.rdann(file_path, 'atr')
        high_arr, _, mp_arr, labels_arr, background = format_timeseries(record, annotations, window_length=window_length)
        background = np.broadcast_to(background, (high_arr.shape[0], 3)) # 3 elements in background - age, gender, medication

        dataset_list.append(np.column_stack((high_arr, mp_arr, background, labels_arr))) 

    channel_width = high_arr.shape[1]
    dataset_list = np.vstack(dataset_list)

    description = {
        'channel': np.arange(0, channel_width, 1),
        'matrix_profile': np.arange(channel_width, channel_width*2, 1),
        'age': channel_width*2,
        'gender': channel_width*2+1,
        'medication': channel_width*2+2,
        'label': channel_width*2+3
    }

    return dataset_list, description  

def get_dynamic_features(signal, fs):
    detector = processing.XQRS(signal, fs)
    detector.detect(verbose=False)
    qrs_ind = detector.qrs_inds
    qrs_ind_diff = np.diff(qrs_ind)
    mean_R_R = np.mean(qrs_ind_diff)
    std_R_R = np.std(qrs_ind_diff)
    root_mean_diff_R_R = np.sqrt(np.mean(np.square(qrs_ind_diff)))
    hrt_rate = processing.compute_hr(signal.shape[0], qrs_ind, fs)
    hrt_rate = hrt_rate[~np.isnan(hrt_rate)]
    mean_hrt_rate = np.mean(hrt_rate)
    std_hrt_rate = np.std(hrt_rate)

    return np.asarray([mean_R_R, std_R_R, root_mean_diff_R_R, mean_hrt_rate, std_hrt_rate])

def get_signal_statistical_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    kurtosis = stats.kurtosis(signal)
    skew = stats.skew(signal)

    return np.asarray([mean, std, minimum, maximum, kurtosis, skew])

def build_dataset(raw_dataset, description, fs):
    df = pd.DataFrame(index=np.arange(0, raw_dataset.shape[0], 1))
    signal = raw_dataset[:,description['channel']]
    mp = raw_dataset[:, description['matrix_profile']]
    # print(f'df shape: {df.shape}')
    # print(f'stats shape: {pd.DataFrame(np.apply_along_axis(get_signal_statistical_features, axis=1, arr=signal)).shape}')
    df[['mean_R_R', 'std_R_R', 'root_mean_diff_R_R', 'mean_hrt_rate', 'std_hrt_rate']] = pd.DataFrame(np.apply_along_axis(get_dynamic_features, axis=1, arr=signal, fs=fs))
    df[['mean', 'std', 'minimum', 'maximum', 'kurtosis', 'skew']] = pd.DataFrame(np.apply_along_axis(get_signal_statistical_features, axis=1, arr=signal))
    df[['mean_mp', 'std_mp', 'minimum_mp', 'maximum_mp', 'kurtosis_mp', 'skew_mp']] = pd.DataFrame(np.apply_along_axis(get_signal_statistical_features, axis=1, arr=mp))

    df['age'] = raw_dataset[:, description['age']]
    df['gender'] = raw_dataset[:, description['gender']]
    df['label'] = raw_dataset[:, description['label']]

    return df.astype(float)