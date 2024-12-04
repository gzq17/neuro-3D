from torch.utils.data import Dataset
import os
import numpy as np
import torch
import mne
import sys
import open3d as o3d
import pandas as pd
sys.path.append('.')
from eeg_data_process.check_process import get_makers, correctEvent
   
def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def data_organization(sub='sub02'):
    root_path = '/home/bingxing2/ailab/ailab_share/scxlab0036/ai4neuro-gzq21/EEGdata/'
    save_arr_name = f"{root_path}{sub}/process_data_7s_100Hz.npy"
    label_arr_name = f"{root_path}{sub}/name_label.npy"
    if os.path.exists(save_arr_name):
        data_array = np.load(save_arr_name)
        label_arr = np.load(label_arr_name)
    else:
        print('error ')
        return
    print(sub)
    print(data_array.shape, label_arr.shape)
    # import pdb;pdb.set_trace()
    result_dir = {}
    for ii in range(0, data_array.shape[0]):
        name = label_arr[ii]
        if name in result_dir.keys():
            result_dir[name].append(data_array[ii:ii+1])
        else:
            result_dir[name] = [data_array[ii:ii+1]]
    sort_keys = sorted(list(result_dir.keys()))
    print(sort_keys)
    train_list, test_list, train_label, test_label = [], [], [], []
    for key in sort_keys:
        data_arr_one_cls = np.concatenate(result_dir[key], axis=0)
        if key[-2:] == '08' or key[-2:] == '09':
            print(data_arr_one_cls.shape)
            test_list.append(data_arr_one_cls[np.newaxis, :, :, :])
            test_label.append(key)
        else:
            # print(data_arr_one_cls.shape)
            train_list.append(data_arr_one_cls[np.newaxis, :, :, :])
            train_label.append(key)
    # import pdb;pdb.set_trace()
    train_data = np.concatenate(train_list, axis=0).reshape(72, 8, 2, 64, data_array.shape[-1])
    # for ii in range(0, len(test_list)):
    #     print(test_list[ii].shape)
    test_data = np.concatenate(test_list, axis=0).reshape(72, 2, 4, 64, data_array.shape[-1])
    np.save(f"{root_path}{sub}/{sub}_train_data_7s_100Hz.npy", train_data)
    np.save(f"{root_path}{sub}/{sub}_test_data_7s_100Hz.npy", test_data)

def get_one_data2(ss, events, eeg_filter2, maker, time_b=0, time_e=8, new_sfreq=250):
    new_events = []
    for index in ss:
        onset_sample = events[index][0]
        new_events.append([onset_sample, 0, maker])  # 100 作为事件代码代表 'S100'
    new_events = np.array(new_events)
    # 创建包含 'S100' 事件及其 1000ms 数据段的 Epochs 对象
    epochs = mne.Epochs(eeg_filter2, new_events, event_id={f'S{maker}': maker}, tmin=time_b, tmax=time_e, baseline=(None, 0), preload=True)
    epochs.resample(new_sfreq)
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(epochs)
    epochs_clean = ica.apply(epochs)
    data = epochs.get_data(copy=False)
    return data

def things_eeg_process(sub='sub10'):
    chan_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 
                      'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 
                      'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 
                      'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 
                      'O1', 'OZ', 'O2', 'CB2']
    root_path = '/home/bingxing2/ailab/ailab_share/scxlab0036/ai4neuro-gzq21/EEGdata/'
    txt_path = root_path + sub + '/' + sub + '/'
    data_list, label_list = [], []
    for session in range(1, 25):
        print(sub, session)
        # if session != 21:
        #     continue
        txt_name = f"{txt_path}{sub}_{str(session).zfill(2)}.txt"
        if not os.path.exists(txt_name):
            txt_name = f'{root_path}{sub}/VideoRecord/{sub}_{str(session).zfill(2)}.txt'
        name_list = read_txt(txt_name)
        print(len(name_list))
        label_list = label_list + name_list
        eeg_name = f"{root_path}{sub}/session{session}.cnt"
        if not os.path.exists(eeg_name):
            eeg_name = f"{root_path}{sub}/session{session}/session{session}.cnt"
        if not os.path.exists(eeg_name):
            print(sub, session)
        eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
        eeg_data.pick_channels(chan_order, ordered=True)

        eeg_data = eeg_data.copy().filter(l_freq=0.1, h_freq=100, fir_design='firwin', phase='zero-double')
        eeg_data.set_eeg_reference(ref_channels=['FZ'])
        sfreq = eeg_data.info['sfreq']

        events, event_dict = correctEvent(eeg_data)
        
        if sub == 'sub26' and session == 2:
            events = events[4:]
        if sub == 'sub25' and session == 2:
            time1 = int(events[83][0] + 0.5 * sfreq)
            time2 = int(events[83][0] + 1.0 * sfreq)
            time3 = int(events[84][0] - 1.0 * sfreq)
            add_event = np.array([[time1, 0, 8],
                      [time2, 0, 4],
                      [time3, 0, 6]])
            events = np.concatenate([events[:84], add_event, events[84:]], axis=0)
        if sub == 'sub15' and session == 21:
            time1 = int(events[-5][0] + 2.5 * sfreq)
            time2 = int(events[-4][0] + 2.5 * sfreq)
            time3 = int(events[-3][0] + 2.5 * sfreq)
            time4 = int(events[-2][0] + 2.5 * sfreq)
            time5 = int(events[-1][0] + 2.5 * sfreq)
            add_event = np.array([[time1, 0, 5],
                      [time2, 0, 2],
                      [time3, 0, 1],
                      [time4, 0, 7],
                      [time5, 0, 3]])
            events = np.concatenate([events, add_event], axis=0)
        if sub == 'sub30' and session == 14:
            time1 = int(events[0][0] - 1.0 * sfreq)
            add_event = np.array([[time1, 0, 8]])
            events = np.concatenate([add_event, events], axis=0)
        makers_num_new = get_makers(events)
        # import pdb;pdb.set_trace()
        s100_maker, s120_maker = makers_num_new[0], makers_num_new[1]
        if sub == 'sub30' and session == 14:
            s100_maker, s120_maker = makers_num_new[-1], makers_num_new[0]
        s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
        s120 = [i for i, event in enumerate(events) if event[2] == s120_maker]  # 假设 'S100' 对应的事件代码是 2
        if sub == 'sub15' and session == 24:
            s100 = s100[1:]
            s120 = s120[1:]
        if len(s100) > 72:
            s100 = s100[:72]
        if len(s120) > 72:
            s120 = s120[:72]
        # import pdb;pdb.set_trace()
        data = get_one_data2(s100, events, eeg_data, s100_maker, -0.2, 0.8, new_sfreq=250)###对应1s_250Hz
        # data = get_one_data2(s120, events, eeg_data, s120_maker, -0.2, 5.8, new_sfreq=100)###对应6s_100Hz
        # data = get_one_data2(s100, events, eeg_data, s100_maker, -0.2, 6.8, new_sfreq=100)###对应7s_100Hz
        if sub == 'sub30' and session == 14 and data.shape[0] == 71:
            ###前面1s的数据没有记录上
            print('concat ')
            data = np.concatenate([data[0:1], data], axis=0)
        print(data.shape, data.min(), data.max())
        # data_normalized = data
        mean_per_trial = np.mean(data, axis=(1, 2), keepdims=True)
        std_per_trial = np.std(data, axis=(1, 2), keepdims=True)
        data_normalized = (data - mean_per_trial) / std_per_trial
        print(data_normalized.shape, data_normalized.min(), data_normalized.max())
        if data_normalized.shape[0] != 72:
            print(sub, session)
            exit()
        data_list.append(data_normalized)
    data_array = np.concatenate(data_list, axis=0)
    label_arr = np.array(label_list)
    print(data_array.shape)
    np.save(f"{root_path}{sub}/name_label.npy", label_arr)
    print(data_array.shape)
    np.save(f"{root_path}{sub}/process_data_1s_250Hz.npy", data_array)

if __name__ == '__main__':
    sub_list=['sub30']#'sub30'
    for sub in sub_list:
        things_eeg_process(sub=sub)
        # data_organization(sub=sub)
