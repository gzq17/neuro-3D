import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import math
import pandas as pd
from mne.time_frequency import psd_array_welch
root_path = '/home/bingxing2/ailab/ailab_share/scxlab0036/ai4neuro-gzq21/EEGdata/'
color_list = ['royalblue', 'darkgreen', 'darkorange', 'limegreen', 'deepskyblue', '#CBDE3A', 'violet']
marker_list = ['*', '.', 's', 'X', 'p', 'P', 'v']

def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def data_check():
    '''
    for sub02 sub01
    '''
    sub = 'sub02'
    session = 13
    txt_path = root_path + sub + '/' + 'VideoRecord/'
    txt_name = f"{txt_path}{sub}_{str(session).zfill(2)}.txt"
    png_save_path = './eeg_check_png/'
    os.makedirs(png_save_path, exist_ok=True)
    name_list = read_txt(txt_name)
    print(len(name_list))
    eeg_name = f"{root_path}{sub}/session{session}/session{session}.cnt"
    eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
    
    eeg_data.drop_channels(['Trigger'])
    eeg_resample = eeg_data.copy().resample(250)
    eeg_filter = eeg_resample.copy().filter(l_freq=0.1, h_freq=40, fir_design='firwin', phase='zero-double')
    eeg_filter2 = eeg_filter.copy().notch_filter(50)
    
    events, event_dict = mne.events_from_annotations(eeg_filter2)
    # 输出信息
    print("通道名称:", eeg_filter2.ch_names)
    print("采样频率:", eeg_filter2.info['sfreq'])
    print("通道类型:", eeg_filter2.get_channel_types())
    
    s100 = [i for i, event in enumerate(events) if event[2] == 2]  # 假设 'S100' 对应的事件代码是 2
    s150 = [i for i, event in enumerate(events) if event[2] == 3]  # 假设 'S150' 对应的事件代码是 3
    print(len(s100), len(s150))
    # first_s100_sample = events[s100[50]][0]  # 第一个'S100'事件的样本位置
    # first_s150_sample = events[s150[50]][0]  # 第一个'S150'事件的样本位置
    # time_difference = (first_s150_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
    # print(f"Time difference between the first 'S100' and 'S150' events: {time_difference} seconds") 
    
    # new_events = []
    # for index_s150 in s150:
    #     onset_sample = events[index_s150][0]
    #     new_events.append([onset_sample, 0, 3])  # 100 作为事件代码代表 'S100'
    # new_events = np.array(new_events)
    
    new_events = []
    for index_s100 in s100:
        onset_sample = events[index_s100][0]
        new_events.append([onset_sample, 0, 2])  # 2 作为事件代码代表 'S100'
    new_events = np.array(new_events)
    epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S100': 2}, tmin=-.2, tmax=0.8, baseline=(0, 0), preload=True, reject=None)
    import pdb;pdb.set_trace()
    # 创建包含 'S100' 事件及其 1000ms 数据段的 Epochs 对象
    epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S150': 3}, tmin=-.2, tmax=8, baseline=(None, 0), preload=True)
    eeg_filter2.set_eeg_reference('average', projection=True)
    epochs.apply_baseline(baseline=(None, None))
    
    data = epochs.get_data(copy=False)
    print(data.shape, data.min(), data.max())
    # import pdb;pdb.set_trace()
    mean_per_trial = np.mean(data, axis=(1, 2), keepdims=True)
    std_per_trial = np.std(data, axis=(1, 2), keepdims=True)
    data_normalized = (data - mean_per_trial) / std_per_trial
    epochs_normalized = mne.EpochsArray(data_normalized, epochs.info, tmin=epochs.tmin)
    normalized_data = epochs_normalized.get_data(copy=False)
    print(normalized_data.shape, normalized_data.min(), normalized_data.max())

def data_check2():
    '''
    for sub03, sub01
    '''
    sub = 'sub03'
    session = 2
    name_list_error = []
    for session in range(1, 25):
        # if session != 14:
        #     continue
        txt_path = root_path + sub + '/' + sub + '/'
        txt_name = f"{txt_path}{sub}_{str(session).zfill(2)}.txt"
        png_save_path = './eeg_check_png/'
        os.makedirs(png_save_path, exist_ok=True)
        name_list = read_txt(txt_name)
        print(len(name_list))
        eeg_name = f"{root_path}{sub}/session{session}.cnt"
        eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
        
        # eeg_data.drop_channels(['Trigger'])
        # eeg_resample = eeg_data.copy().resample(250)
        # eeg_filter = eeg_resample.copy().filter(l_freq=0.1, h_freq=40, fir_design='firwin', phase='zero-double')
        # eeg_filter2 = eeg_filter.copy().notch_filter(50)
        
        eeg_filter2 = eeg_data
        events, event_dict = mne.events_from_annotations(eeg_filter2)
        print(events)
        # import pdb;pdb.set_trace()
        # 输出信息
        sfreq = eeg_filter2.info['sfreq']
        print("通道名称:", eeg_filter2.ch_names)
        print("采样频率:", eeg_filter2.info['sfreq'])
        print("通道类型:", eeg_filter2.get_channel_types())
        s100_maker, s150_maker = events[-2][-1], events[-1][-1]
        s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
        if sub == 'sub01' and session == 14:
            s150 = [i for i, event in enumerate(events) if event[2] == s150_maker or event[2] == 3]  # 假设 'S150' 对应的事件代码是 3
        else:
            s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]
        print(len(s100), len(s150))
        for ii in range(len(s100)):
            first_s100_sample = events[s100[ii]][0]  # 第一个'S100'事件的样本位置
            first_s150_sample = events[s150[ii]][0]  # 第一个'S150'事件的样本位置
            time_difference = (first_s150_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
            # if time_difference > 8.1 or time_difference < 7.9:
            #     print(session, time_difference, ii)
            #     name_list_error.append(name_list[ii])
            print(f"Time difference between the first 'S100' and 'S150' events: {time_difference} seconds") 
    print(sorted(name_list_error))
    # import pdb;pdb.set_trace()
    # # new_events = []
    # # for index_s150 in s150:
    # #     onset_sample = events[index_s150][0]
    # #     new_events.append([onset_sample, 0, 3])  # 100 作为事件代码代表 'S100'
    # # new_events = np.array(new_events)
    
    # new_events = []
    # for index_s100 in s100:
    #     onset_sample = events[index_s100][0]
    #     new_events.append([onset_sample, 0, s100_maker])  # 2 作为事件代码代表 'S100'
    # new_events = np.array(new_events)
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S100': s100_maker}, tmin=-.2, tmax=0.8, baseline=(0, 0), preload=True, reject=None)
    # import pdb;pdb.set_trace()
    # # 创建包含 'S100' 事件及其 1000ms 数据段的 Epochs 对象
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S150': s150_maker}, tmin=-.2, tmax=8, baseline=(None, 0), preload=True)
    # eeg_filter2.set_eeg_reference('average', projection=True)
    # epochs.apply_baseline(baseline=(None, None))
    
    # data = epochs.get_data(copy=False)
    # print(data.shape, data.min(), data.max())
    # # import pdb;pdb.set_trace()
    # mean_per_trial = np.mean(data, axis=(1, 2), keepdims=True)
    # std_per_trial = np.std(data, axis=(1, 2), keepdims=True)
    # data_normalized = (data - mean_per_trial) / std_per_trial
    # epochs_normalized = mne.EpochsArray(data_normalized, epochs.info, tmin=epochs.tmin)
    # normalized_data = epochs_normalized.get_data(copy=False)
    # print(normalized_data.shape, normalized_data.min(), normalized_data.max())
    # erp_check(normalized_data[:, :, :125], eeg_filter2.ch_names, png_save_path)
    # STFT_check(normalized_data[:, :, :-1], eeg_filter2.ch_names, png_save_path)

def get_makers(events):
    makers_dir={}
    for ii in range(0, len(events)):
        if events[ii][-1] in makers_dir.keys():
            makers_dir[events[ii][-1]] += 1
        else:
            makers_dir[events[ii][-1]] = 1
    makers_num = []
    for key in makers_dir.keys():
        if makers_dir[key] > 70:
            makers_num.append(key)
    makers_num_new = []
    for ii in range(0, 10):
        if events[ii][-1] in makers_num and events[ii][-1] not in makers_num_new:
            makers_num_new.append(events[ii][-1])
    return makers_num_new

def data_check3():
    '''
    for sub04~sub06
    '''
    sub = 'sub06' ###sub06 session17有问题
    session = 2
    name_list_error = []
    for session in range(1, 25):
        print(session)
        if session != 17:
            continue
        txt_path = root_path + sub + '/' + sub + '/'
        txt_name = f"{txt_path}{sub}_{str(session).zfill(2)}.txt"
        png_save_path = './eeg_check_png/'
        os.makedirs(png_save_path, exist_ok=True)
        name_list = read_txt(txt_name)
        print(len(name_list))
        eeg_name = f"{root_path}{sub}/session{session}.cnt"
        eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
        
        # eeg_data.drop_channels(['Trigger'])
        # eeg_resample = eeg_data.copy().resample(250)
        # eeg_filter = eeg_resample.copy().filter(l_freq=0.1, h_freq=40, fir_design='firwin', phase='zero-double')
        # eeg_filter2 = eeg_filter.copy().notch_filter(50)
        
        eeg_filter2 = eeg_data
        events, event_dict = mne.events_from_annotations(eeg_filter2)
        events2, event_dict2 = correctEvent(eeg_filter2)
        for ii in range(0, len(events)):
            if events2[ii][0] - events[ii][0] != 0:
                print(sub, session)
        print(events)
        # 输出信息
        sfreq = eeg_filter2.info['sfreq']
        print("通道名称:", eeg_filter2.ch_names)
        print("采样频率:", eeg_filter2.info['sfreq'])
        print("通道类型:", eeg_filter2.get_channel_types())
        import pdb;pdb.set_trace()
        makers_num_new = get_makers(events)

        # s100_maker, s150_maker, s050_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2]

        # s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
        # s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]  # 假设 'S100' 对应的事件代码是 2
        # s050 = [i for i, event in enumerate(events) if event[2] == s050_maker]  # 假设 'S100' 对应的事件代码是 2

        # print(len(s100), len(s150), len(s050))
        # for ii in range(len(s100)):
        #     first_s100_sample = events[s100[ii]][0]  # 第一个'S100'事件的样本位置
        #     first_s150_sample = events[s150[ii]][0]  # 第一个'S150'事件的样本位置
        #     first_s050_sample = events[s050[ii]][0]
        #     time_difference1 = (first_s150_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
        #     time_difference2 = (first_s050_sample - first_s150_sample) / sfreq  # 时间差，以秒为单位
        #     print(f'{time_difference1:.4f}, {time_difference2:.4f}')
            # if time_difference2 > 6.6:
            #     print(session, time_difference2, ii)
            # if time_difference > 8.1 or time_difference < 7.9:
            #     print(session, time_difference, ii)
            #     name_list_error.append(name_list[ii])
            # print(f"Time difference between the first 'S100' and 'S150' events: {time_difference} seconds") 
    print(sorted(name_list_error))
    # import pdb;pdb.set_trace()
    # # new_events = []
    # # for index_s150 in s150:
    # #     onset_sample = events[index_s150][0]
    # #     new_events.append([onset_sample, 0, 3])  # 100 作为事件代码代表 'S100'
    # # new_events = np.array(new_events)
    
    # new_events = []
    # for index_s100 in s100:
    #     onset_sample = events[index_s100][0]
    #     new_events.append([onset_sample, 0, s100_maker])  # 2 作为事件代码代表 'S100'
    # new_events = np.array(new_events)
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S100': s100_maker}, tmin=-.2, tmax=0.8, baseline=(0, 0), preload=True, reject=None)
    # import pdb;pdb.set_trace()
    # # 创建包含 'S100' 事件及其 1000ms 数据段的 Epochs 对象
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S150': s150_maker}, tmin=-.2, tmax=8, baseline=(None, 0), preload=True)
    # eeg_filter2.set_eeg_reference('average', projection=True)
    # epochs.apply_baseline(baseline=(None, None))
    
    # data = epochs.get_data(copy=False)
    # print(data.shape, data.min(), data.max())
    # # import pdb;pdb.set_trace()
    # mean_per_trial = np.mean(data, axis=(1, 2), keepdims=True)
    # std_per_trial = np.std(data, axis=(1, 2), keepdims=True)
    # data_normalized = (data - mean_per_trial) / std_per_trial
    # epochs_normalized = mne.EpochsArray(data_normalized, epochs.info, tmin=epochs.tmin)
    # normalized_data = epochs_normalized.get_data(copy=False)
    # print(normalized_data.shape, normalized_data.min(), normalized_data.max())
    # erp_check(normalized_data[:, :, :125], eeg_filter2.ch_names, png_save_path)
    # STFT_check(normalized_data[:, :, :-1], eeg_filter2.ch_names, png_save_path)

def correctEvent(raw):
    events, event_dict = mne.events_from_annotations(raw)

    if '255' in event_dict.keys():
        events = events[events[:, -1] != event_dict['255']]

    if 'Trigger' in raw.ch_names:

        x = raw['Trigger'][0][0]
        onset = np.squeeze(np.argwhere(np.diff(x)>0))

        # ! deal with the issue, a 1 ms trigger before the real trigger 
        events = np.delete(events, np.where(np.diff(events[:, 0]) == 1), axis=0)
        onset = np.delete(onset, np.where(np.diff(onset) == 1))

        if len(events)>len(onset):
            events[:len(onset), 0] = onset[:]
        else:
            events[:, 0] = onset[:len(events)]
    else:
        pass

    return events, event_dict

def data_check4():
    '''
    for sub07+
    '''
    sub = 'subtest'
    session = 2
    name_list_error = []
    error_list1_all, error_list2_all, error_list3_all = [], [], []
    for session in range(2, 5):
        # if session != 7:
        #     continue
        # eeg_name = f"{root_path}{sub}/session{session}.cnt"
        eeg_name = f"{root_path}{sub}/Acquisition 0{session} Data.cnt"
        eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
        print(session)
        # eeg_data.drop_channels(['Trigger'])
        # eeg_resample = eeg_data.copy().resample(250)
        # eeg_filter = eeg_resample.copy().filter(l_freq=0.1, h_freq=40, fir_design='firwin', phase='zero-double')
        # eeg_filter2 = eeg_filter.copy().notch_filter(50)
        
        eeg_filter2 = eeg_data
        # events, event_dict = mne.events_from_annotations(eeg_filter2)
        events, event_dict = correctEvent(eeg_filter2)
        # import pdb;pdb.set_trace()
        # # 输出信息
        sfreq = eeg_filter2.info['sfreq']
        print("通道名称:", eeg_filter2.ch_names)
        # print("采样频率:", eeg_filter2.info['sfreq'])
        # # print("通道类型:", eeg_filter2.get_channel_types())
        makers_num_new = get_makers(events)

        s100_maker, s120_maker, s150_maker, s050_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[3]

        s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
        s120 = [i for i, event in enumerate(events) if event[2] == s120_maker]  # 假设 'S100' 对应的事件代码是 2
        s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]  # 假设 'S100' 对应的事件代码是 2
        s050 = [i for i, event in enumerate(events) if event[2] == s050_maker]  # 假设 'S100' 对应的事件代码是 2

        # s100_new, s120_new, s150_new, s050_new = [], [], [], []
        # for ii in range(s100[0], s050[-1] + 1):
        #     if (ii - s100[0]) % 4 == 0:
        #         s100_new.append(ii)
        #     elif (ii - s100[0]) % 4 == 1:
        #         s120_new.append(ii)
        #     elif (ii - s100[0]) % 4 == 2:
        #         s150_new.append(ii)
        #     elif (ii - s100[0]) % 4 == 3:
        #         s050_new.append(ii)
        # s100, s120, s150, s050 = s100_new, s120_new, s150_new, s050_new
        print(len(s100), len(s120), len(s150), len(s050))
        # import pdb;pdb.set_trace()
        error_list1, error_list2, error_list3 = [], [], []
        for ii in range(len(s050)):
            first_s100_sample = events[s100[ii]][0]  # 第一个'S100'事件的样本位置
            first_s120_sample = events[s120[ii]][0]
            first_s150_sample = events[s150[ii]][0]  # 第一个'S150'事件的样本位置
            first_s050_sample = events[s050[ii]][0]
            time_difference1 = (first_s120_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
            time_difference2 = (first_s150_sample - first_s120_sample) / sfreq
            time_difference3 = (first_s050_sample - first_s150_sample) / sfreq  # 时间差，以秒为单位
            # print(f'{time_difference1:.4f}, {time_difference2:.4f}, {time_difference3:.4f}')
            error_list1.append(round(time_difference1 - 1.0, 3))
            error_list2.append(round(time_difference2 - 6.0, 3))
            error_list3.append(round(time_difference3 - 0.5, 3))
        error_list1_all, error_list2_all, error_list3_all = error_list1_all + error_list1, error_list2_all + error_list2, error_list3_all + error_list3
        print(len(s100), len(s120), len(s150), len(s050))
        print(sorted(error_list1, reverse=True))
        print(sorted(error_list2, reverse=True))
        print(sorted(error_list3, reverse=True))
    print('---'*10)
    print(sorted(error_list1_all, reverse=True))
    print('---'*10)
    print(sorted(error_list2_all, reverse=True))
    print('---'*10)
    print(sorted(error_list3_all, reverse=True))
        #     # if time_difference2 > 6.1 or time_difference1 > 1.1 or time_difference3 > 0.6:
        #     #     print(session, f'{time_difference1:.4f}, {time_difference2:.4f}, {time_difference3:.4f}', ii)
        #     # if time_difference > 8.1 or time_difference < 7.9:
        #     #     print(session, time_difference, ii)
        #     #     name_list_error.append(name_list[ii])
        #     # print(f"Time difference between the first 'S100' and 'S150' events: {time_difference} seconds") 
    # print(sorted(name_list_error))
    # import pdb;pdb.set_trace()
    # # new_events = []
    # # for index_s150 in s150:
    # #     onset_sample = events[index_s150][0]
    # #     new_events.append([onset_sample, 0, 3])  # 100 作为事件代码代表 'S100'
    # # new_events = np.array(new_events)
    
    # new_events = []
    # for index_s100 in s100:
    #     onset_sample = events[index_s100][0]
    #     new_events.append([onset_sample, 0, s100_maker])  # 2 作为事件代码代表 'S100'
    # new_events = np.array(new_events)
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S100': s100_maker}, tmin=-.2, tmax=0.8, baseline=(0, 0), preload=True, reject=None)
    # import pdb;pdb.set_trace()
    # # 创建包含 'S100' 事件及其 1000ms 数据段的 Epochs 对象
    # epochs = mne.Epochs(eeg_filter2, new_events, event_id={'S150': s150_maker}, tmin=-.2, tmax=8, baseline=(None, 0), preload=True)
    # eeg_filter2.set_eeg_reference('average', projection=True)
    # epochs.apply_baseline(baseline=(None, None))
    
    # data = epochs.get_data(copy=False)
    # print(data.shape, data.min(), data.max())
    # # import pdb;pdb.set_trace()
    # mean_per_trial = np.mean(data, axis=(1, 2), keepdims=True)
    # std_per_trial = np.std(data, axis=(1, 2), keepdims=True)
    # data_normalized = (data - mean_per_trial) / std_per_trial
    # epochs_normalized = mne.EpochsArray(data_normalized, epochs.info, tmin=epochs.tmin)
    # normalized_data = epochs_normalized.get_data(copy=False)
    # print(normalized_data.shape, normalized_data.min(), normalized_data.max())
    # erp_check(normalized_data[:, :, :125], eeg_filter2.ch_names, png_save_path)
    # STFT_check(normalized_data[:, :, :-1], eeg_filter2.ch_names, png_save_path)

def data_check5():
    '''
    for sub07+
    '''
    sub = 'sub30'
    error_list1_all, error_list2_all, error_list3_all, error_list4_all = [], [], [], []
    for session in range(11, 12):
        # if session != 7:
        #     continue
        # eeg_name = f"{root_path}{sub}/session{session}.cnt"
        print(session)
        # if session != 21:
        #     continue
        eeg_name = f"{root_path}{sub}/session{session}.cnt"
        eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
        # eeg_data = eeg_data.copy().resample(250)
        eeg_filter2 = eeg_data
        # # events, event_dict = mne.events_from_annotations(eeg_filter2)
        # # import pdb;pdb.set_trace()
        
        events, event_dict = correctEvent(eeg_filter2)
        sfreq = eeg_filter2.info['sfreq']
        print("通道名称:", eeg_filter2.ch_names)
        
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
            time1 = int(events[-1][0] + 2.5 * sfreq)
            time2 = int(events[-1][0] + 3.5 * sfreq)
            time3 = int(events[-1][0] + 9.5 * sfreq)
            time4 = int(events[-1][0] + 10 * sfreq)
            time5 = int(events[-1][0] + 10.5 * sfreq)
            add_event = np.array([[time1, 0, 5],
                      [time2, 0, 2],
                      [time3, 0, 1],
                      [time4, 0, 7],
                      [time5, 0, 3]])
            events = np.concatenate([events, add_event], axis=0)
        makers_num_new = get_makers(events)
        if sub == 'sub30' and session == 14:
            time1 = int(events[0][0] - 1.0 * sfreq)
            add_event = np.array([[time1, 0, 8]])
            events = np.concatenate([add_event, events], axis=0)
        # for ii in range(0, len(events)):
        #     print(str(ii).zfill(3), events[ii][-1], events[ii][0])
        # import pdb;pdb.set_trace()
        # s100_maker, s120_maker, s150_maker, s050_maker = 8, 4, 2, 5
        s100_maker, s120_maker, s150_maker, s050_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[-1]
        if sub == 'sub30' and session == 14:
            s100_maker, s120_maker, s150_maker, s050_maker = makers_num_new[-1], makers_num_new[0], makers_num_new[1], makers_num_new[3]
        print(s100_maker, s120_maker, s150_maker, s050_maker)
        # s100_maker, s120_maker, s150_maker, s050_maker, s180_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[3], makers_num_new[4]

        s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]
        s120 = [i for i, event in enumerate(events) if event[2] == s120_maker]
        s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]
        s050 = [i for i, event in enumerate(events) if event[2] == s050_maker]
        print(len(s100), len(s120), len(s150), len(s050))
        if len(s100) < 72 or len(s120) < 72:
            print(sub, session)
            exit()
        print('---*' * 10)
        print('---*' * 10)
        
        if sub == 'sub15' and session == 24:
            s100 = s100[1:]
            s120 = s120[1:]
        error_list1, error_list2, error_list3, error_list4 = [], [], [], []
        for ii in range(len(s050)):
            first_s100_sample = events[s100[ii]][0]  # 第一个'S100'事件的样本位置
            first_s120_sample = events[s120[ii]][0]
            first_s150_sample = events[s150[ii]][0]
            first_s050_sample = events[s050[ii]][0]
            # first_s180_sample = events[s180[ii]][0]
            time_difference1 = (first_s120_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
            time_difference2 = (first_s150_sample - first_s120_sample) / sfreq
            time_difference3 = (first_s050_sample - first_s150_sample) / sfreq  # 时间差，以秒为单位
            error_list1.append(np.fabs(round(time_difference1 - 1.0, 3)))
            error_list2.append(np.fabs(round(time_difference2 - 6.0, 3)))
            error_list3.append(np.fabs(round(time_difference3 - 1.0, 3)))
            # error_list4.append(round(time_difference4 - 0.5, 3))
        print(len(s100), len(s120), len(s150), len(s050))
        print(sorted(error_list1, reverse=True))
        print(sorted(error_list2, reverse=True))
        print(sorted(error_list3, reverse=True))
        print(f'session:{session}, image error:{np.array(error_list1).mean()}, std:{np.array(error_list1).std()}')
        print(f'session:{session}, video error:{np.array(error_list2).mean()}, std:{np.array(error_list2).std()}')
        # print(sorted(error_list4, reverse=True))
        error_list1_all += error_list1
        error_list2_all += error_list2
        error_list3_all += error_list3
        error_list4_all += error_list4
    print(sorted(error_list1_all, reverse=True)[:300])
    print(sorted(error_list2_all, reverse=True)[:300])
    print(sorted(error_list3_all, reverse=True)[:300])
    print(f'all image error:{np.array(error_list1_all).mean()}, std:{np.array(error_list1_all).std()}')
    print(f'all video error:{np.array(error_list2_all).mean()}, std:{np.array(error_list2_all).std()}')

def draw_eeg():
    chan_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 
                      'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 
                      'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 
                      'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 
                      'O1', 'OZ', 'O2', 'CB2']
    sub = 'sub09'
    session = 1
    eeg_name = f"{root_path}{sub}/session{session}.cnt"
    eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
    eeg_data.pick_channels(chan_order, ordered=True)
    eeg_resample = eeg_data.copy().resample(100)
    eeg_filter2 = eeg_resample
    events, event_dict = correctEvent(eeg_filter2)
    sfreq = eeg_filter2.info['sfreq']
    print("通道名称:", eeg_filter2.ch_names)
    makers_num_new = get_makers(events)
    # import pdb;pdb.set_trace()
    s100_maker, s120_maker, s150_maker, s050_maker, s180_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[3], makers_num_new[4]
    s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
    s120 = [i for i, event in enumerate(events) if event[2] == s120_maker]  # 假设 'S100' 对应的事件代码是 2
    s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]  # 假设 'S100' 对应的事件代码是 2
    s050 = [i for i, event in enumerate(events) if event[2] == s050_maker]  # 假设 'S100' 对应的事件代码是 2
    s180 = [i for i, event in enumerate(events) if event[2] == s180_maker]
    new_events = []
    for index_s100 in s100:
        onset_sample = events[index_s100][0]
        new_events.append([onset_sample, 0, s100_maker])  # 100 作为事件代码代表 'S100'
    new_events = np.array(new_events)
    epochs = mne.Epochs(eeg_filter2, new_events, event_id={f'S100': s100_maker}, tmin=0, tmax=8, baseline=(0, 0), preload=True)
    eeg_filter2.set_eeg_reference('average', projection=True)
    epochs.apply_baseline(baseline=(None, None))
    data = epochs.get_data(copy=False)
    print(data.shape)
    plt.figure(figsize=(8, 5))
    index = 1
    plt.plot(data[0, index], color=color_list[index], linewidth=1)
    dist = data[0, index].max() - data[0, index].min()
    plt.ylim(data[0, index].min() - dist, data[0, index].max() + dist)
    plt.savefig(chan_order[index] + '.png')

def error_count():
    # sub_list = ['sub09', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 
    #             'sub17', 'sub18', 'sub19', 'sub20', 'sub22', 'sub24', 'sub25', 'sub26',
    #             'sub27', 'sub28', 'sub29', 'sub30', 'sub31', 'sub32']
    sub_list = ['sub16', 'sub30']
    result_all_count = [[f'session{ii}'] for ii in range(1, 25)]
    result_all_count.append(['All'])
    column_list = ['Key']
    for sub in sub_list:
        error_list1_all, error_list2_all, error_list3_all, error_list4_all = [], [], [], []
        for session in range(1, 25):
            eeg_name = f"{root_path}{sub}/session{session}.cnt"
            eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
            eeg_filter2 = eeg_data
            events, event_dict = correctEvent(eeg_filter2)
            sfreq = eeg_filter2.info['sfreq']
            print("通道名称:", eeg_filter2.ch_names)
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
                time1 = int(events[-1][0] + 2.5 * sfreq)
                time2 = int(events[-1][0] + 3.5 * sfreq)
                time3 = int(events[-1][0] + 9.5 * sfreq)
                time4 = int(events[-1][0] + 10 * sfreq)
                time5 = int(events[-1][0] + 10.5 * sfreq)
                add_event = np.array([[time1, 0, 5],
                        [time2, 0, 2],
                        [time3, 0, 1],
                        [time4, 0, 7],
                        [time5, 0, 3]])
                events = np.concatenate([events, add_event], axis=0)
            makers_num_new = get_makers(events)
            if sub == 'sub30' and session == 14:
                time1 = int(events[0][0] - 1.0 * sfreq)
                add_event = np.array([[time1, 0, 8]])
                events = np.concatenate([add_event, events], axis=0)
            s100_maker, s120_maker, s150_maker, s050_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[-1]
            if sub == 'sub30' and session == 14:
                s100_maker, s120_maker, s150_maker, s050_maker = makers_num_new[-1], makers_num_new[0], makers_num_new[1], makers_num_new[3]
            print(s100_maker, s120_maker, s150_maker, s050_maker)
            s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]  # 假设 'S100' 对应的事件代码是 2
            s120 = [i for i, event in enumerate(events) if event[2] == s120_maker]  # 假设 'S100' 对应的事件代码是 2
            s150 = [i for i, event in enumerate(events) if event[2] == s150_maker]  # 假设 'S100' 对应的事件代码是 2
            s050 = [i for i, event in enumerate(events) if event[2] == s050_maker]  # 假设 'S100' 对应的事件代码是 2
            print(len(s100), len(s120), len(s150), len(s050))
            if len(s100) < 72 or len(s120) < 72:
                print(sub, session)
                exit()
            print('---*' * 10)
            print('---*' * 10)
            if sub == 'sub15' and session == 24:
                s100 = s100[1:]
                s120 = s120[1:]
            error_list1, error_list2, error_list3, error_list4 = [], [], [], []
            for ii in range(len(s050)):
                first_s100_sample = events[s100[ii]][0]  # 第一个'S100'事件的样本位置
                first_s120_sample = events[s120[ii]][0]
                first_s150_sample = events[s150[ii]][0]  # 第一个'S150'事件的样本位置
                first_s050_sample = events[s050[ii]][0]
                time_difference1 = (first_s120_sample - first_s100_sample) / sfreq  # 时间差，以秒为单位
                time_difference2 = (first_s150_sample - first_s120_sample) / sfreq
                time_difference3 = (first_s050_sample - first_s150_sample) / sfreq
                error_list1.append(np.fabs(round(time_difference1 - 1.0, 3)) * 1000)
                error_list2.append(np.fabs(round(time_difference2 - 6.0, 3)) * 1000)
                error_list3.append(np.fabs(round(time_difference3 - 1.0, 3)) * 1000)
            print(len(s100), len(s120), len(s150), len(s050))
            # print(sorted(error_list1, reverse=True))
            # print(sorted(error_list2, reverse=True))
            # print(sorted(error_list3, reverse=True))
            s_mean, s_std = np.array(error_list1).mean(), np.array(error_list1).std()
            t_mean, t_std = np.array(error_list2).mean(), np.array(error_list2).std()
            if sub == 'sub16' and session==5:
                t_mean, t_std = 0, 0
                error_list1_new = [0] * len(error_list1)
                error_list2_new = [0] * len(error_list2)
                error_list1, error_list2 = error_list1_new, error_list2_new
            if sub == 'sub30' and session == 11:
                t_mean, t_std = 0, 0
                error_list1_new = [0] * len(error_list1)
                error_list2_new = [0] * len(error_list2)
                error_list1, error_list2 = error_list1_new, error_list2_new
            result_all_count[session - 1] += [round(s_mean, 2), round(s_std, 2), round(t_mean, 2), round(t_std, 2)]
            
            print(f'session:{session}, image error:{s_mean}, std:{s_std}')
            print(f'session:{session}, video error:{t_mean}, std:{t_std}')
            # print(sorted(error_list4, reverse=True))
            error_list1_all += error_list1
            error_list2_all += error_list2
            error_list3_all += error_list3
            error_list4_all += error_list4
        print(sorted(error_list1_all, reverse=True)[:300])
        print(sorted(error_list2_all, reverse=True)[:300])
        print(sorted(error_list3_all, reverse=True)[:300])
        sub_s_mean, sub_s_std = np.array(error_list1_all).mean(), np.array(error_list1_all).std()
        sub_t_mean, sub_t_std = np.array(error_list2_all).mean(), np.array(error_list2_all).std()
        print(f'all image error:{np.array(error_list1_all).mean()}, std:{np.array(error_list1_all).std()}')
        print(f'all video error:{np.array(error_list2_all).mean()}, std:{np.array(error_list2_all).std()}')
        column_list += [f"{sub[-2:]}-S-M(ms)", f"{sub[-2:]}-S-S", f"{sub[-2:]}-T-M(ms)", f"{sub[-2:]}-T-S"]
        result_all_count[-1] += [round(sub_s_mean, 2), round(sub_s_std, 2), round(sub_t_mean, 2), round(sub_t_std, 2)]
    df = pd.DataFrame(result_all_count)
    df.columns = column_list
    df.to_excel(f"{root_path}result_error_count.xlsx", index=False)

def plot_all_channel():
    chan_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 
                      'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 
                      'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 
                      'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 
                      'O1', 'OZ', 'O2', 'CB2']
    sub = 'sub13'
    session = 1
    eeg_name = f"{root_path}{sub}/session{session}.cnt"
    eeg_data = mne.io.read_raw_cnt(eeg_name, preload=True)
    eeg_data.pick_channels(chan_order, ordered=True)
    eeg_resample = eeg_data.copy().resample(100)
    eeg_filter2 = eeg_resample
    events, event_dict = correctEvent(eeg_filter2)
    sfreq = eeg_filter2.info['sfreq']
    print("通道名称:", eeg_filter2.ch_names)
    makers_num_new = get_makers(events)
    s100_maker, s120_maker, s150_maker, s050_maker, s180_maker = makers_num_new[0], makers_num_new[1], makers_num_new[2], makers_num_new[3], makers_num_new[4]
    s100 = [i for i, event in enumerate(events) if event[2] == s100_maker]
    new_events = []
    for index_s100 in s100:
        onset_sample = events[index_s100][0]
        new_events.append([onset_sample, 0, s100_maker])  # 100 作为事件代码代表 'S100'
    new_events = np.array(new_events)
    epochs = mne.Epochs(eeg_filter2, new_events[:1], event_id={f'S100': s100_maker}, tmin=0, tmax=100, baseline=(0, 0), preload=True)
    eeg_filter2.set_eeg_reference('average', projection=True)
    epochs.apply_baseline(baseline=(None, None))
    data = epochs.get_data(copy=False)
    print(data.shape)
    plt.figure(figsize=(8, 5))
    index = 1
    plt.plot(data[0, index], linewidth=1)
    dist = data[0, index].max() - data[0, index].min()
    plt.ylim(data[0, index].min() - dist, data[0, index].max() + dist)
    plt.savefig(chan_order[index] + '.png')
    
    fig, axes = plt.subplots(8, 8, figsize=(20, 16))
    y_min = np.min(data)
    y_max = np.max(data)

    # 绘制每个子图
    for i in range(64):
        ax = axes[i // 8, i % 8]
        dist = data[0, i].max() - data[0, i].min()
        ax.plot(data[0, i])
        ax.set_ylim(data[0, i].min() - dist, data[0, i].max() + dist)
        ax.set_title(f"{chan_order[i]}")
    plt.tight_layout()
    plt.show()
    plt.savefig('all_channel.png')
    plt.close()
    
    ####画功率谱时，不要resample
    # fmin, fmax = 0, 100  # 感兴趣的频段
    # psd, frequencies = psd_array_welch(data[0], sfreq=1000, fmin=fmin, fmax=fmax, n_fft=2048, average='mean')
    # # import pdb;pdb.set_trace()
    # # 绘制功率谱
    # log_psd = 10 * np.log10(psd)
    # plt.figure(figsize=(12, 6))
    # for channel in range(64):
    #     plt.plot(frequencies, log_psd[channel], alpha=0.7)

    # plt.title('Power Spectral Density (PSD)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density 10*log10 (µV²/Hz)')
    # plt.xlim(fmin, fmax)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('psd.png')
    

if __name__ == '__main__':
    data_check5()
    # error_count()
    # plot_all_channel()
    # preprocessing2()
    # draw_eeg()
