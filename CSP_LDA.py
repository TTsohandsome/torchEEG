import numpy as np
import scipy.signal as signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def CSP_LDA(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Common Spatial Pattern + Linear Discriminant Analysis for Motor Imagery Classification
    
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
    channel_indices = list(range(0, 17)) + list(range(18, 30))  # 对应 MATLAB 的 [1:17 19:30]
    trigger_indices = np.where(MIEEGData[:, 32] == 2)[0]  # 假设第33列是触发点（索引32）
    print(len(channel_indices), len(trigger_indices))
    
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
        
        # CSP特征提取
        def compute_csp(eeg_train, label_train, n_components=2):
            # 计算类协方差矩阵
            class1 = eeg_train[label_train == 1]
            class2 = eeg_train[label_train == 2]
            
            cov1 = np.cov(class1, rowvar=False)
            cov2 = np.cov(class2, rowvar=False)
            
            # 计算总协方差矩阵
            cov_total = (len(class1)*cov1 + len(class2)*cov2) / (len(class1) + len(class2))
            
            # 计算特征值和特征向量
            eig_vals, eig_vecs = np.linalg.eigh(cov_total, cov1)
            
            # 选择前n_components个特征向量
            csp_matrix = eig_vecs[:, ::-1][:, :n_components]
            
            # 投影数据
            projected_train = np.dot(eeg_train, csp_matrix.T)
            
            return projected_train, csp_matrix
        
        train_feature, csp_matrix = compute_csp(eeg_train, label_train, n_components=2)
        
        # 对测试数据应用CSP投影
        test_feature = np.dot(eeg_test, csp_matrix.T)[:2, :]  # 取前2个特征
        
        # LDA分类
        lda = LinearDiscriminantAnalysis()
        lda.fit(train_feature, label_train)
        predictions = lda.predict(test_feature)
        
        # 计算准确率
        accuracy = accuracy_score(label_test, predictions) * 100
        acc.append(accuracy)
        
        # 统计左右样本数量
        left = sum(1 for lbl in label_test if lbl == 1)
        right = len(label_test) - left
        left_num.append(left)
        right_num.append(right)
    
    return acc, left_num, right_num