import os
import numpy as np

def data_organization(sub='sub02'):
    root_path = '/disk3/guozhanqiang/CVPR2025Data/EEG_data/'
    npy_name_list = ['1s_250Hz.npy', '6s_100Hz.npy']
    for npy_name in npy_name_list:
        save_arr_name = f"{root_path}{sub}/process_data_{npy_name}"
        label_arr_name = f"{root_path}{sub}/name_label.npy"
        if os.path.exists(save_arr_name):
            data_array = np.load(save_arr_name)
            label_arr = np.load(label_arr_name)
        else:
            return
        print(sub)
        print(data_array.shape, label_arr.shape)
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
                train_list.append(data_arr_one_cls[np.newaxis, :, :, :])
                train_label.append(key)
        # import pdb;pdb.set_trace()
        train_data = np.concatenate(train_list, axis=0).reshape(72, 8, 2, 64, data_array.shape[-1])
        test_data = np.concatenate(test_list, axis=0).reshape(72, 2, 4, 64, data_array.shape[-1])
        np.save(f"{root_path}{sub}/{sub}_train_data_{npy_name}", train_data)
        np.save(f"{root_path}{sub}/{sub}_test_data_{npy_name}", test_data)

if __name__ == '__main__':
    data_organization(sub='sub01')
