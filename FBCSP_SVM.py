import numpy as np
import scipy.signal as signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

def FBCSP_SVM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Filter Bank Common Spatial Pattern + Support Vector Machine for Motor Imagery Classification
    
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
        
        # FBCSP参数设置
        freq_bands = [(10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]  # 频率带
        csp_para = 2  # CSP组件数
        filter_order = 4  # 滤波器阶数
        
        # FBCSP特征提取函数
        def compute_fbcsp(eeg_train, label_train, csp_para, freq_bands, Fs, filter_order):
            num_bands = len(freq_bands)
            num_channels = eeg_train.shape[0]
            num_trials = eeg_train.shape[2]
            
            # 初始化特征矩阵
            feature_matrix = []
            
            for band in freq_bands:
                low, high = band
                
                # 设计带通滤波器
                def design_bandpass(low, high, Fs, order=4):
                    nyq = Fs / 2.0
                    low_w0 = low / nyq
                    high_w0 = high / nyq
                    b, a = signal.butter(order, [low_w0, high_w0], 'bandpass', Fs)
                    return b, a
                
                b, a = design_bandpass(low, high, Fs, filter_order)
                
                # 对每个试验应用滤波器
                band_eeg = []
                for trial in range(num_trials):
                    trial_data = eeg_train[:, trial]
                    filtered = signal.lfilter(b, a, trial_data)
                    band_eeg.append(filtered)
                band_eeg = np.array(band_eeg).T  # shape (samples×channels×num_trials)
                
                # 计算CSP
                def compute_csp(band_eeg, label_train, n_components=2):
                    class1 = band_eeg[label_train == 1]
                    class2 = band_eeg[label_train == 2]
                    
                    cov1 = np.cov(class1, rowvar=False)
                    cov2 = np.cov(class2, rowvar=False)
                    
                    # 总协方差矩阵
                    total_samples = len(class1) + len(class2)
                    cov_total = (len(class1)*cov1 + len(class2)*cov2) / total_samples
                    
                    # 特征值分解
                    eig_vals, eig_vecs = np.linalg.eigh(cov_total, cov1)
                    
                    # 选择特征向量
                    csp_matrix = eig_vecs[:, ::-1][:, :n_components]
                    
                    # 投影数据
                    projected = np.dot(band_eeg, csp_matrix.T)
                    return projected, csp_matrix
                
                trial_features, csp_matrix = compute_csp(band_eeg, label_train, csp_para)
                feature_matrix.append(trial_features)
            
            # 拼接所有频率带的特征
            full_feature = np.hstack(feature_matrix)
            return full_feature, csp_matrix
        
        # 训练FBCSP
        train_feature, fbcsp_w = compute_fbcsp(eeg_train, label_train, csp_para, freq_bands, Fs, filter_order)
        
        # 测试数据投影
        test_feature = []
        for band in freq_bands:
            low, high = band
            
            # 设计带通滤波器
            b, a = signal.butter(4, [low, high], 'bandpass', Fs)
            
            # 对测试数据应用滤波器
            band_test = []
            for trial in range(eeg_test.shape[2]):
                trial_data = eeg_test[:, trial]
                filtered = signal.lfilter(b, a, trial_data)
                band_test.append(filtered)
            band_test = np.array(band_test).T
            
            # 应用CSP矩阵投影
            band_proj = np.dot(band_test, fbcsp_w)
            test_feature.append(band_proj)
        
        test_feature = np.hstack(test_feature)[:csp_para * num_bands, :]  # 根据CSP组件数截取
        
        # SVM分类
        # 使用标准化提高SVM性能
        scaler = StandardScaler()
        scaler.fit(train_feature)
        train_scaled = scaler.transform(train_feature)
        test_scaled = scaler.transform(test_feature)
        
        # 训练SVM
        svm = SVC(kernel='linear', C=1.0, probability=True)
        svm.fit(train_scaled, label_train)
        
        # 预测
        test_label = svm.predict(test_scaled)
        
        # 计算准确率
        accuracy = accuracy_score(label_test, test_label) * 100
        acc.append(accuracy)
        
        # 统计左右样本数量
        left = sum(1 for lbl in label_test if lbl == 1)
        right = len(label_test) - left
        left_num.append(left)
        right_num.append(right)
    
    return acc, left_num, right_num