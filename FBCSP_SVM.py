import numpy as np
from scipy import signal
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def FBCSP_SVM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Filter Bank Common Spatial Pattern + Support Vector Machine
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
    
    # 定义频带
    freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]
    
    # 对每个频带进行滤波和特征提取
    all_features = []
    
    for low, high in freq_bands:
        # 带通滤波
        b, a = signal.butter(4, [low/(Fs/2), high/(Fs/2)], btype='band')
        filtered = np.zeros_like(MIEEGData)
        
        # 对每个试次的每个通道进行滤波
        for trial in range(MIEEGData.shape[0]):
            for channel in range(MIEEGData.shape[2]):
                filtered[trial, :, channel] = signal.filtfilt(b, a, MIEEGData[trial, :, channel])
        
        # 计算每个试次的协方差矩阵
        covs = np.array([np.cov(trial.T) for trial in filtered])
        
        # 分离类别
        covs_class1 = covs[label == np.unique(label)[0]]
        covs_class2 = covs[label == np.unique(label)[1]]
        
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
        n_components = min(4, csp_matrix.shape[0])
        spatial_filters = np.vstack([
            csp_matrix[:n_components//2],
            csp_matrix[-n_components//2:]
        ])
        
        # 特征提取
        band_features = np.zeros((filtered.shape[0], n_components))
        for i, trial in enumerate(filtered):
            projected = np.dot(spatial_filters, trial.T)
            band_features[i] = np.log(np.var(projected, axis=1))
        
        all_features.append(band_features)
    
    # 合并所有频带的特征
    features = np.hstack(all_features)
    # print(f"Final features shape: {features.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # SVM分类
    clf = SVC(kernel='rbf', random_state=42)
    
    # 使用交叉验证
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    left_nums = []
    right_nums = []
    
    for train_idx, test_idx in skf.split(features, label):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        
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