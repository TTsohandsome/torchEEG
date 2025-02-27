import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import seaborn as sns
from pathlib import Path
from algorithms import csp_lda, fbcsp_svm, tslda_dgfmdm, twfb_dgfmdm  # 添加算法导入

class EEGProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.acc = []
        self.left = []
        self.right = []
        self.all_left = []
        self.all_right = []
        
    def normalize_data(self, data):
        # Z-score normalization
        valid_channels = np.std(data, axis=1) != 0
        data_zscore = np.zeros_like(data)
        data_zscore[valid_channels] = (data[valid_channels] - np.mean(data[valid_channels], axis=1, keepdims=True)) / \
                                    np.std(data[valid_channels], axis=1, keepdims=True)
        return data_zscore
    
    def process_subject(self, subject_num):
        # 构建主体文件夹路径
        if subject_num < 10:
            subject_folder = os.path.join(self.base_path, f'sub-0{subject_num}')
        else:
            subject_folder = os.path.join(self.base_path, f'sub-{subject_num}')
            
        print(f'Processing folder: {subject_folder}')
        
        # 获取所有.mat文件
        mat_files = [f for f in os.listdir(subject_folder) if f.endswith('.mat')]
        
        acc0, acc1, acc2, acc3 = [], [], [], []
        left_num0 = np.zeros((2,2))
        right_num0 = np.zeros((2,2))
        left_num1 = np.zeros((2,2))
        right_num1 = np.zeros((2,2))
        left_num2 = np.zeros((2,2))
        right_num2 = np.zeros((2,2))
        left_num3 = np.zeros((2,2))
        right_num3 = np.zeros((2,2))
        
        for mat_file in mat_files:
            file_path = os.path.join(subject_folder, mat_file)
            print(f'Loading file: {file_path}')
            
            # 加载.mat文件
            eeg_data = sio.loadmat(file_path)
            raw_data = eeg_data['eeg']['rawdata'][0,0]
            labels = eeg_data['eeg']['label'][0,0]
            
            # 修改数据处理部分
            # 假设raw_data的形状是(trials, channels, time_points)
            n_trials, n_channels, n_times = raw_data.shape
            
            # 重新组织数据，每个试次作为一个样本
            data = raw_data.reshape(n_trials, -1)  # 展平为(trials, channels*time_points)
            
            # 标准化
            data = self.normalize_data(data)
            
            # 应用分类算法
            fs = 500
            low_freq = 4
            up_freq = 30
            
            # 确保labels的形状正确
            labels = labels.reshape(-1)
            
            # 调用不同的分类算法
            acc0_tmp, left_num0_tmp, right_num0_tmp = csp_lda(data, labels, fs, low_freq, up_freq)
            acc0.append(acc0_tmp)
            left_num0 += left_num0_tmp.reshape(2,1) @ np.ones((1,2))
            right_num0 += right_num0_tmp.reshape(2,1) @ np.ones((1,2))
            
            acc1_tmp, left_num1_tmp, right_num1_tmp = fbcsp_svm(data, labels, fs, low_freq, up_freq)
            acc1.append(acc1_tmp)
            left_num1 += left_num1_tmp.reshape(2,1) @ np.ones((1,2))
            right_num1 += right_num1_tmp.reshape(2,1) @ np.ones((1,2))
            
            acc2_tmp, left_num2_tmp, right_num2_tmp = tslda_dgfmdm(data, labels, fs, low_freq, up_freq)
            acc2.append(acc2_tmp)
            left_num2 += left_num2_tmp.reshape(2,1) @ np.ones((1,2))
            right_num2 += right_num2_tmp.reshape(2,1) @ np.ones((1,2))
            
            acc3_tmp, left_num3_tmp, right_num3_tmp = twfb_dgfmdm(data, labels, fs, low_freq, up_freq)
            acc3.append(acc3_tmp)
            left_num3 += left_num3_tmp.reshape(2,1) @ np.ones((1,2))
            right_num3 += right_num3_tmp.reshape(2,1) @ np.ones((1,2))
            
        # 计算平均值
        results = {
            'acc': np.array([np.mean(acc0), np.mean(acc1), np.mean(acc2), np.mean(acc3)]),
            'left': np.array([np.mean(left_num0[:,1]), np.mean(left_num1[:,1]), 
                            np.mean(left_num2[:,1]), np.mean(left_num3[:,1])]),
            'right': np.array([np.mean(right_num0[:,1]), np.mean(right_num1[:,1]),
                             np.mean(right_num2[:,1]), np.mean(right_num3[:,1])])
        }
        
        return results
    
    def plot_acc_bar(self, acc):
        plt.figure(figsize=(10,6))
        methods = ['CSP+LDA', 'FBCSP+SVM', 'TSLDA+DGFMDM', 'TWFB+DGFMDM']
        plt.bar(methods, np.mean(acc, axis=1))
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Method')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        methods = ['CSP+LDA', 'FBCSP+SVM', 'TSLDA+DGFMDM', 'TWFB+DGFMDM']
        
        for idx, (ax, method) in enumerate(zip(axes.flat, methods)):
            sns.heatmap(cm[idx], annot=True, fmt='d', ax=ax)
            ax.set_title(method)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        for subject in range(49, 51):
            results = self.process_subject(subject)
            self.acc.append(results['acc'])
            self.left.append(results['left'])
            self.right.append(results['right'])
            
        # 绘制结果
        self.plot_acc_bar(np.array(self.acc).T)
        
        # 保存结果
        np.savez('FeatureData1.npz', 
                 accuracy=self.acc,
                 confusion_matrices=self.calculate_confusion_matrices())

if __name__ == "__main__":
    processor = EEGProcessor("sourcedata")
    processor.run_analysis()
