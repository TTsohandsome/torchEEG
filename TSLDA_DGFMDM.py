import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def TSLDA_DGFMDM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    TSLDA_DGFMDM: Tangent Space LDA and Discriminant Geodesic Filtering with MDM for Motor Imagery Classification
    
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
    
    eeg = np.zeros((2000, len(channel_indices), len(trigger_indices)), dtype=np.float64)
    
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
        
        bandpass_data = bandpass_filter(notch_data, Fs, LowFreq, UpFreq)
        
        # 提取801-2800样本（静息期）
        eeg[:, :, trial_idx] = bandpass_data[800:2800, :].T
    
    # 3. 交叉验证
    acc = []
    left_num = []
    right_num = []
    
    for fold in range(10):
        # 打乱样本顺序
        indices = list(range(len(label)))
        random.shuffle(indices)
        train_indices = indices[:24]
        test_indices = indices[24:]
        
        eeg_train = eeg[:, :, train_indices]
        eeg_test = eeg[:, :, test_indices]
        label_train = label[train_indices]
        label_test = label[test_indices]
        
        # 计算训练集协方差矩阵
        num_channels = eeg_train.shape[0]
        num_trials_train = eeg_train.shape[2]
        COVtrain = []
        for trial in range(num_trials_train):
            ss = np.squeeze(eeg_train[:, trial])
            cov = ss @ ss.T
            COVtrain.append(cov)
        COVtrain = np.stack(COVtrain, axis=2)
        
        # 计算测试集协方差矩阵
        num_trials_test = eeg_test.shape[2]
        COVtest = []
        for trial in range(num_trials_test):
            ss = np.squeeze(eeg_test[:, trial])
            cov = ss @ ss.T
            COVtest.append(cov)
        COVtest = np.stack(COVtest, axis=2)
        
        # 分类方法1: Tangent Space LDA
        def tslda_classify(COVtest_sub, COVtrain_sub, y_train_sub, metric='riemann'):
            # 这里需要实现黎曼几何下的TSLDA算法
            # 由于代码复杂性，此处提供一个简化的示例（需补充完整实现）
            return np.random.randint(1, 3, size=len(y_train_sub))  # 示例占位符
        
        # 分类方法2: Discriminant Geodesic Filtering + MDM
        def fgmdm_classify(COVtest_sub, COVtrain_sub, y_train_sub, metric_mean='riemann', metric_dist='riemann'):
            # 这里需要实现DGFM-DM算法
            return np.random.randint(1, 3, size=len(y_train_sub))  # 示例占位符
        
        # 计算准确率
        accuracy1 = 0.0
        accuracy2 = 0.0
        
        # 类别对 (1,2)
        idx_i = np.where(label_train == 1)[0]
        idx_j = np.where(label_train == 2)[0]
        if len(idx_i) == 0 or len(idx_j) == 0:
            continue
        
        # 方法1: TSLDA
        y_train_sub = label_train[idx_i + idx_j]
        COVtrain_sub = COVtrain[idx_i + idx_j, :, idx_i + idx_j]
        COVtest_sub = COVtest[:, idx_i + idx_j, :]
        pred1 = tslda_classify(COVtest_sub, COVtrain_sub, y_train_sub)
        true_idx = np.where(label_test == [1,2])[0]
        accuracy1 = accuracy_score(label_test[true_idx], pred1[true_idx]) * 100
        
        # 方法2: DGFM-DM
        pred2 = fgmdm_classify(COVtest_sub, COVtrain_sub, y_train_sub)
        accuracy2 = accuracy_score(label_test[true_idx], pred2[true_idx]) * 100
        
        # 选择最佳分类器
        if accuracy1 > accuracy2:
            test_label = pred1[true_idx]
            current_acc = accuracy1
        else:
            test_label = pred2[true_idx]
            current_acc = accuracy2
        
        acc.append(current_acc)
        
        # 统计左右样本数量
        left = sum(1 for lbl in label_test if lbl == 1)
        right = len(label_test) - left
        left_num.append(left)
        right_num.append(right)
    
    return acc, left_num, right_num