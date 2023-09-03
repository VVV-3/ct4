import torch
from torch.utils.data import Dataset
from helper_code import *
import numpy as np, os, sys
import librosa
import mne

from tqdm import tqdm

def featurise_recording(location):
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 
                'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    recording_data, sr, cur_channels = load_recording(location)
    signal_data = reorder_recording_channels(recording_data, cur_channels, channels)
    delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sr,  fmin=0.5,  fmax=8.0, verbose=False)
    theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sr,  fmin=4.0,  fmax=8.0, verbose=False)
    alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sr,  fmin=8.0, fmax=12.0, verbose=False)
    beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sr, fmin=12.0, fmax=30.0, verbose=False)
    final = np.concatenate((delta_psd, theta_psd, alpha_psd, beta_psd), axis=1)
    return np.mean(final, axis=0)

def featurise_locs(locs):
    fin = []
    for loc in locs:
        fin.append(featurise_recording(loc))
    fin = np.array(fin)
#     print(fin.shape)
    return fin

def featurise_labels(label):
#     print(label)
    tp = np.array([0,0])
    tp[label] = 1
    return tp

class patientDataset(Dataset):
    def __init__(self, data_folder, segs):
        
        self.no_of_segments = segs
        
        self.data_folder  = data_folder
        self.patient_ids  = find_data_folders(data_folder)
        self.num_patients = len(self.patient_ids)
        
        self.inputs = []
        self.outcomes = []
        
        for i in tqdm(range(self.num_patients)):
            inp, out = self.getitemx(i)
            self.inputs.append(inp)
            self.outcomes.append(out)
        
    def getMetadata(self, idx):
        # Load data.
        patient_id = self.patient_ids[idx]
        
        # Define file location.
        patient_metadata_file = os.path.join(self.data_folder, patient_id, patient_id + '.txt')
        recording_metadata_file = os.path.join(self.data_folder, patient_id, patient_id + '.tsv')

        # Load non-recording data.
        patient_metadata = load_text_file(patient_metadata_file)
        recording_metadata = load_text_file(recording_metadata_file)
        
        return patient_metadata, recording_metadata
        
    def getitemx(self, index):
        
        patient_metadata, recording_metadata = self.getMetadata(index)
        
        # Load recordings.
        recording_ids = list(get_recording_ids(recording_metadata))
        
        recording_locations = []
        for recording_id in reversed(recording_ids):
            if recording_id != 'nan':
                recording_location = os.path.join(self.data_folder, self.patient_ids[index], recording_id)
                recording_locations.append(recording_location)
            
            if len(recording_locations) >= self.no_of_segments:
                break
        
        return featurise_locs(recording_locations), featurise_labels(get_outcome(patient_metadata))
    
    def __getitem__(self, index):
        
        return self.inputs[index], self.outcomes[index]
        
    def __len__(self):
        return self.num_patients
        
        
class eegDataset(Dataset):
    def __init__(self, patientDataset, idxs):
        
        self.channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 
                         'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        
        self.l_cutoff = 0.1
        self.h_cutoff = 15
        self.order = 3
        
        self.final_sr = 2*self.h_cutoff
        
        self.st_time = 0
        self.end_time = 300
        self.hop_time = 10
        self.segment_length = 20
        
        self.recordings = []
        self.outcomes   = []
        
        for idx in idxs:
            self.addPatient(patientDataset, idx)
        
    def bandpassFilter(self, signal, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_l_cutoff = self.l_cutoff / nyq
        normal_h_cutoff = self.h_cutoff / nyq

        # get the filter coefficients
        b, a = sp.signal.butter( self.order, [normal_l_cutoff, normal_h_cutoff], btype='bp', analog=False)
        filtered_signal = sp.signal.filtfilt(b, a, signal)
        return filtered_signal
        
    def addRecording(self, recording_data, sr, outcome):
        filtered_data  = self.bandpassFilter(recording_data, sr)
        resampled_data = librosa.resample(y=filtered_data, orig_sr=sr, target_sr=self.final_sr)
#         mod_data, _ = mne.time_frequency.psd_array_welch(recording_data, sfreq=sr,  
#                                                       fmin=0.5,  fmax=30.0, verbose=False)
#         print(mod_data.shape)
        for st_time in range(self.st_time, self.end_time, self.hop_time):
            end_time = st_time+self.segment_length
            if end_time > self.end_time:
                continue
                
            st_mkr, end_mkr = st_time*self.final_sr, end_time*self.final_sr
            recording_segment = resampled_data[:, st_mkr:end_mkr]
#             mx_arr = np.max(np.abs(recording_segment) , axis=1)
#             mx_arr[mx_arr==0] = 1
#             recording_segment = (recording_segment.T/mx_arr ).T
            
            self.recordings.append(recording_segment)
            self.outcomes.append(outcome)
        
    def addPatient(self, patientDataset, idx):
        recording_locations, outcomes = patientDataset[idx]
        
        for i in range(len(recording_locations)):
            recording_location, outcome = recording_locations[i], outcomes[i]
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            recording_data = reorder_recording_channels(recording_data, channels, self.channels)
            self.addRecording(recording_data, sampling_frequency, outcome)
        
    def __getitem__(self, index):
        outs =[0,0]
        outs[self.outcomes[index]] = 1
        return self.recordings[index], torch.Tensor(outs)
        
    def __len__(self):
        return len(self.recordings)