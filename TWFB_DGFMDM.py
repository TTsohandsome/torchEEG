import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def TWFB_DGFMDM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    TWFB_DGFMDM: Time-Frequency Bandwidth Filtering combined with Discriminant Geodesic Filtering and MDM
    
    Parameters:
    MIEEGData: ndarray (samples×channels×trials)
    label: list/array (1 or 2 for each trial)
    Fs: int (sampling frequency)
    LowFreq: float (low pass cutoff frequency)
    UpFreq: float (high pass cutoff frequency)
    
    Returns:
    acc: list (accuracy rates for each cross-validation fold)
    left_num: list (number of left/right samples in test set per fold)
    right_num: list (number of left/right samples in test set per fold)
    """
    
    # 1. 数据预处理
    channel_indices = [i-1 for i in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]]  # 转换为0-based索引
    trigger_indices = np.where(MIEEGData[:, 32] == 2)[0]  # 假设第33列为触发点（索引32）
    
    # 定义频段列表
    freq_bands = [
        (8, 12), (8, 20), (8, 30),
        (12, 20), (15, 20), (15, 30),
        (20, 30), (8, 15)
    ]
    
    acc = []
    left_num = []
    right_num = []
    
    for fold in range(10):
        # 打乱样本顺序
        indices = list(range(len(label)))
        random.shuffle(indices)
        train_indices = indices[:24]
        test_indices = indices[24:]
        
        # 初始化结果容器
        temp_acc = []
        temp_left = []
        temp_right = []
        
        for band_idx, (low, high) in enumerate(freq_bands):
            # 初始化EEG数据存储
            eeg = np.zeros((2000, len(channel_indices), len(trigger_indices)), dtype=np.float64)
            
            # 处理每个触发点
            for trial_idx, trigger in enumerate(trigger_indices):
                start = trigger - 800
                end = start + 2800
                segment = MIEEGData[start:end, channel_indices].T  # shape (channels×samples)
                
                # Notch滤波（50Hz）
                def notch_filter(data, Fs, notch_freq=50, Q=30):
                    nyq = Fs / 2.0
                    notch_w0 = notch_freq / nyq
                    b, a = signal.iirnotch(notch_w0, Q, Fs)
                    return signal.lfilter(b, a, data)
                
                notch_data = notch_filter(segment, Fs)
                
                # 带通滤波
                def bandpass_filter(data, Fs, low, high):
                    nyq = Fs / 2.0
                    low_w0 = low / nyq
                    high_w0 = high / nyq
                    b, a = signal.butter(4, [low_w0, high_w0], 'bandpass', Fs)
                    return signal.lfilter(b, a, data)
                
                bandpass_data = bandpass_filter(notch_data, Fs, low, high)
                
                # 提取静息期数据
                eeg[:, :, trial_idx] = bandpass_data[800:2800, :].T
            
            # 分割训练集和测试集
            eeg_train = eeg[:, :, train_indices]
            eeg_test = eeg[:, :, test_indices]
            label_train = label[train_indices]
            label_test = label[test_indices]
            
            # 计算协方差矩阵
            def compute_covariance(eeg_data, axis=2):
                num_trials = eeg_data.shape[axis]
                cov = []
                for trial in range(num_trials):
                    ss = np.squeeze(eeg_data[:, trial])
                    cov_mat = ss @ ss.T
                    cov.append(cov_mat)
                return np.stack(cov, axis=axis)
            
            COVtrain = compute_covariance(eeg_train, axis=2)
            COVtest = compute_covariance(eeg_test, axis=2)
            
            Ytrain = label_train
            trueYtest = label_test
            
            # DGFM-DM分类
            def fgmdm_classify(COVtest_sub, COVtrain_sub, y_train_sub, metric='riemann'):
                # 示例占位符：随机分类（需替换为真实实现）
                return np.random.randint(1, 3, size=len(y_train_sub))
            
            NumClass = 2
            acct = []
            
            for i in range(NumClass):
                for j in range(i+1, NumClass):
                    ix_train = (Ytrain == i) | (Ytrain == j)
                    ix_test = (trueYtest == i) | (trueYtest == j)
                    
                    if not ix_train or not ix_test:
                        continue
                    
                    # 提取子协方差矩阵
                    COVtrain_sub = COVtrain[ix_train, :, ix_train]
                    COVtest_sub = COVtest[:, ix_test, :]
                    
                    y_train_sub = Ytrain[ix_train]
                    y_test_sub = trueYtest[ix_test]
                    
                    # 分类
                    pred = fgmdm_classify(COVtest_sub, COVtrain_sub, y_train_sub)
                    accuracy = accuracy_score(y_test_sub, pred) * 100
                    acct.append(accuracy)
            
            if not acct:
                current_acc = 0.0
            else:
                current_acc = max(acc)
            
            # 统计标签数量
            left = sum(1 for lbl in label_test if lbl == 1)
            right = len(label_test) - left
            
            temp_acc.append(current_acc)
            temp_left.append(left)
            temp_right.append(right)
        
        # 选择最佳频段
        best_band = np.argmax(temp_acc)
        acc.append(temp_acc[best_band])
        left_num.append(temp_left[best_band])
        right_num.append(temp_right[best_band])
    
    return acc, left_num, right_num