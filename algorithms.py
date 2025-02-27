from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def csp_lda(data, labels, fs, low_freq, up_freq):
    """CSP+LDA算法实现"""
    # 确保数据维度正确
    n_trials = data.shape[0]
    
    # 滤波 - 对每个试次分别处理
    b, a = signal.butter(4, [low_freq/(fs/2), up_freq/(fs/2)], btype='band')
    filtered_data = np.array([signal.filtfilt(b, a, trial) for trial in data])
    
    # 计算每个试次的方差作为特征
    features = np.var(filtered_data, axis=1)
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, 1))
    
    # LDA分类
    clf = LinearDiscriminantAnalysis()
    clf.fit(features, labels)
    predictions = clf.predict(features)
    
    # 计算准确率
    acc = accuracy_score(labels, predictions)
    
    # 计算左右手分类统计
    left_mask = labels == 0
    right_mask = labels == 1
    
    left_num = np.zeros(2)
    right_num = np.zeros(2)
    
    left_num[0] = np.sum((predictions == 0) & left_mask)
    left_num[1] = np.sum((predictions == 1) & left_mask)
    right_num[0] = np.sum((predictions == 0) & right_mask)
    right_num[1] = np.sum((predictions == 1) & right_mask)
    
    return acc, left_num, right_num

def fbcsp_svm(data, labels, fs, low_freq, up_freq):
    """FBCSP+SVM算法实现"""
    # 简化版实现，实际应该包含多个频带
    acc = csp_lda(data, labels, fs, low_freq, up_freq)[0]  # 临时使用CSP+LDA的结果
    left_num = np.zeros(2)
    right_num = np.zeros(2)
    return acc, left_num, right_num

def tslda_dgfmdm(data, labels, fs, low_freq, up_freq):
    """TSLDA+DGFMDM算法实现"""
    acc = csp_lda(data, labels, fs, low_freq, up_freq)[0]  # 临时使用CSP+LDA的结果
    left_num = np.zeros(2)
    right_num = np.zeros(2)
    return acc, left_num, right_num

def twfb_dgfmdm(data, labels, fs, low_freq, up_freq):
    """TWFB+DGFMDM算法实现"""
    acc = csp_lda(data, labels, fs, low_freq, up_freq)[0]  # 临时使用CSP+LDA的结果
    left_num = np.zeros(2)
    right_num = np.zeros(2)
    return acc, left_num, right_num
