#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from src.datasets import patientDataset, eegDataset
from src.resnet import ResNet1d
import torch
from torch.utils.data import DataLoader

import librosa

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    pattrainset = patientDataset(data_folder)
    trainset = eegDataset(pattrainset, [i for i in range(len(pattrainset))])
    trainloader = DataLoader(dataset=trainset, batch_size=16, shuffle=True, num_workers=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Train the models.
    train_config = { 'num_epochs':20, 'learning_rate':1e-4 }
    arch_config  = {
                'n_input_channels':18,
                'signal_length':75,
                'net_filter_size':[ 18],
                'net_signal_length':[ 75],
                'kernel_size': 3,
                'n_classes':2,
                'dropout_rate':0.15
               }

    model = ResNet1d(input_dim=(arch_config['n_input_channels'], arch_config['signal_length']), 
                     blocks_dim=list(zip(arch_config['net_filter_size'], arch_config['net_signal_length'])),
                     kernel_size=arch_config['kernel_size'],
                     dropout_rate=arch_config['dropout_rate'])
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=1e-7)

    # Train the models.
    train_losses = []
    for epoch in range(train_config['num_epochs']):
        train_losses.append(train_one_epoch(model, device, optimizer, criterion, trainloader, epoch))

    # Save the models.
    save_challenge_model(model_folder, [], model, [])

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models['outcome_model'].to(device)
    model.eval()

    # Load data.
    inputs = load_test_data(data_folder, patient_id)

    # Run Model
    inputs  = inputs.type(torch.FloatTensor).to(device)
    outputs = model(inputs).detach().cpu().numpy().mean(axis=0)
    outcome = np.argmax(outputs)
    outcome_probability = outputs[1]
    cpc = int(5*outcome_probability) + 1
    
#     print(inputs.shape, outputs.shape, outcome_probability, outcome)
    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
    
    
def train_one_epoch(model, device, optimizer, criterion ,train_loader, epoch):
    model.train()
    train_loss = []
    for i, (inputs, labels) in enumerate(train_loader): 
            
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).reshape((labels.shape[0],2)).to(device)
        
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.cpu().detach().numpy())
        print("Epoch {}: {}/{} Loss: {}".format(epoch, i, len(train_loader), loss.item()), end='\r')
    return np.mean(train_loss)

def getMetadata(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)
        
    return patient_metadata, recording_metadata

def load_test_data(data_folder, patient_id):
    inps = []
    
    no_of_segments = 1
    self_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 
                         'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    self_l_cutoff = 0.1
    self_h_cutoff = 10
    self_order = 3
        
    self_final_sr = 2*self_h_cutoff
        
    self_st_time = 0
    self_end_time = 300
    self_hop_time = 10
    self_segment_length = 20
    
    def bandpassFilter(signal, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_l_cutoff = self_l_cutoff / nyq
        normal_h_cutoff = self_h_cutoff / nyq

        # get the filter coefficients
        b, a = sp.signal.butter( self_order, [normal_l_cutoff, normal_h_cutoff], btype='bp', analog=False)
        filtered_signal = sp.signal.filtfilt(b, a, signal)
        return filtered_signal
        
    def addRecording(recording_data, sr):
        filtered_data  = bandpassFilter(recording_data, sr)
        resampled_data = librosa.resample(y=filtered_data, orig_sr=sr, target_sr=self_final_sr)
        mod_data, _ = mne.time_frequency.psd_array_welch(recording_data, sfreq=sr,  
                                                      fmin=0.5,  fmax=30.0, verbose=False)
        
#         for st_time in range(self_st_time, self_end_time, self_hop_time):
#             end_time = st_time+self_segment_length
#             if end_time > self_end_time:
#                 continue
                
#             st_mkr, end_mkr = st_time*self_final_sr, end_time*self_final_sr
#             recording_segment = resampled_data[:, st_mkr:end_mkr]
     
        inps.append(mod_data)
    
    patient_metadata, recording_metadata = getMetadata(data_folder, patient_id)
    recording_ids = list(get_recording_ids(recording_metadata))
    ct=0
    for recording_id in reversed(recording_ids):
        if recording_id != 'nan':
            recording_location = os.path.join(data_folder, patient_id, recording_id)
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            recording_data = reorder_recording_channels(recording_data, channels, self_channels)
            addRecording(recording_data, sampling_frequency)
            ct+=1
            
        if ct >= no_of_segments:
            break
            
    return torch.Tensor(np.array(inps))

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features
