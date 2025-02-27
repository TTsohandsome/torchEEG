import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def TWFB_DGFMDM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Time-Warped Filter Bank with DGFMDM
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
    
    # 定义多个频带
    freq_bands = [
        (4, 8),    # theta
        (8, 13),   # alpha
        (13, 30),  # beta
        (30, 50)   # gamma
    ]
    
    # 对每个频带进行处理
    all_features = []
    
    for low_freq, high_freq in freq_bands:
        # 带通滤波
        b, a = signal.butter(4, [low_freq/(Fs/2), high_freq/(Fs/2)], btype='band')
        filtered = np.zeros_like(MIEEGData)
        
        # 对每个试次的每个通道进行滤波
        for trial in range(MIEEGData.shape[0]):
            for channel in range(MIEEGData.shape[2]):
                filtered[trial, :, channel] = signal.filtfilt(b, a, MIEEGData[trial, :, channel])
        
        # 时间窗口特征
        window_size = 100  # 200ms at 500Hz
        stride = 50       # 100ms overlap
        n_windows = (filtered.shape[1] - window_size) // stride + 1
        
        band_features = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window_data = filtered[:, start:end, :]
            
            # 计算时间窗口特征
            power = np.mean(np.square(window_data), axis=1)  # 功率特征
            variance = np.var(window_data, axis=1)           # 方差特征
            
            # 合并特征
            window_features = np.concatenate([power, variance], axis=1)
            band_features.append(window_features)
        
        # 合并该频带的所有时间窗口特征
        band_features = np.concatenate(band_features, axis=1)
        all_features.append(band_features)
    
    # 合并所有频带的特征
    features = np.concatenate(all_features, axis=1)
    # print(f"Final features shape: {features.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 使用LDA进行分类
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    left_nums = []
    right_nums = []
    
    for train_idx, test_idx in skf.split(features, label):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        
        # LDA分类
        clf = LinearDiscriminantAnalysis(solver='svd')
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
    
    # 返回平均结果
    mean_accuracy = np.mean(accuracies)
    mean_left = np.mean(left_nums, axis=0)
    mean_right = np.mean(right_nums, axis=0)
    
    # print(f"Mean accuracy: {mean_accuracy:.4f}")
    # print(f"Left hand results: {mean_left}")
    # print(f"Right hand results: {mean_right}")
    
    return mean_accuracy, mean_left, mean_right