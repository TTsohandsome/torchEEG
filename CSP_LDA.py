import numpy as np
import scipy.signal as signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def CSP_LDA(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Common Spatial Pattern + Linear Discriminant Analysis for Motor Imagery Classification
    """
    # print(f"Input data shape: {MIEEGData.shape}")
    # print(f"Label shape: {label.shape}")
    # print(f"Unique labels: {np.unique(label)}")

    # 确保标签是一维数组
    label = label.ravel()
    
    # 数据预处理
    if len(MIEEGData.shape) == 2:  # 如果数据是2D的 (samples, channels)
        n_samples, n_channels = MIEEGData.shape
        n_trials = len(label)
        samples_per_trial = n_samples // n_trials
        # 重塑为 (trials, time_points, channels)
        MIEEGData = MIEEGData.reshape(n_trials, samples_per_trial, n_channels)
    
    # print(f"Reshaped data: {MIEEGData.shape}")
    
    # 带通滤波
    def bandpass_filter(data, Fs, low, high):
        nyq = Fs / 2.0
        low_w0 = low / nyq
        high_w0 = high / nyq
        b, a = signal.butter(4, [low_w0, high_w0], 'bandpass')
        filtered = np.zeros_like(data)
        # 对每个试次的每个通道进行滤波
        for trial in range(data.shape[0]):
            for channel in range(data.shape[2]):
                filtered[trial, :, channel] = signal.filtfilt(b, a, data[trial, :, channel])
        return filtered
    
    # 应用滤波
    filtered_data = bandpass_filter(MIEEGData, Fs, LowFreq, UpFreq)
    # print(f"Filtered data shape: {filtered_data.shape}")
    
    # 计算每个试次的协方差矩阵
    def compute_covariance(trial_data):
        # trial_data shape: (time_points, channels)
        return np.cov(trial_data, rowvar=False)
    
    covs = np.array([compute_covariance(trial) for trial in filtered_data])
    # print(f"Covariance matrices shape: {covs.shape}")
    
    # 分离类别
    # print(f"Computing class-wise covariance matrices for labels: {np.unique(label)}")
    covs_class1 = covs[label == np.unique(label)[0]]
    covs_class2 = covs[label == np.unique(label)[1]]
    # print(f"Class 1 covs shape: {covs_class1.shape}")
    # print(f"Class 2 covs shape: {covs_class2.shape}")
    
    # 计算平均协方差矩阵
    mean_cov1 = np.mean(covs_class1, axis=0)
    mean_cov2 = np.mean(covs_class2, axis=0)
    
    # CSP变换矩阵计算
    composite_cov = mean_cov1 + mean_cov2
    eigenvals, eigenvects = np.linalg.eigh(composite_cov)
    
    # 确保特征值为正
    eigenvals = np.maximum(eigenvals, 1e-10)
    
    # 白化变换
    whitening = np.dot(np.diag(np.power(eigenvals, -0.5)), eigenvects.T)
    transformed_cov1 = np.dot(np.dot(whitening, mean_cov1), whitening.T)
    
    # CSP投影矩阵
    eigenvals_csp, eigenvects_csp = np.linalg.eigh(transformed_cov1)
    csp_matrix = np.dot(eigenvects_csp.T, whitening)
    
    # 选择最显著的特征
    n_components = min(4, csp_matrix.shape[0])  # 确保不超过可用的特征数
    spatial_filters = np.vstack([
        csp_matrix[:n_components//2],
        csp_matrix[-n_components//2:]
    ])
    
    # 特征提取
    features = np.zeros((filtered_data.shape[0], n_components))
    for i, trial in enumerate(filtered_data):
        projected = np.dot(spatial_filters, trial.T)
        features[i] = np.log(np.var(projected, axis=1))
    
    # print(f"Final features shape: {features.shape}")
    
    # 使用交叉验证
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    left_nums = []
    right_nums = []
    
    for train_idx, test_idx in skf.split(features, label):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        # 统计分类结果
        class_labels = np.unique(label)
        left_nums.append([
            np.sum((y_pred == class_labels[0]) & (y_test == class_labels[0])),
            np.sum((y_pred == class_labels[1]) & (y_test == class_labels[0]))
        ])
        right_nums.append([
            np.sum((y_pred == class_labels[0]) & (y_test == class_labels[1])),
            np.sum((y_pred == class_labels[1]) & (y_test == class_labels[1]))
        ])
    
    # 修改返回值的格式
    mean_accuracy = np.mean(accuracies)
    mean_left = np.mean(left_nums, axis=0)  # 确保是一维数组 [correct, incorrect]
    mean_right = np.mean(right_nums, axis=0)  # 确保是一维数组 [correct, incorrect]
    
    # 打印调试信息
    # print(f"Mean accuracy: {mean_accuracy:.4f}")
    # print(f"Left hand results: {mean_left}")
    # print(f"Right hand results: {mean_right}")
    
    return mean_accuracy, mean_left, mean_right