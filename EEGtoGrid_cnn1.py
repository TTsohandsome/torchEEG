import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os.path as osp

class EEGGrid:
    def __init__(self, data, labels):
        self.trials = data.shape[0]  # 试验次数
        self.channels = data.shape[1]  # 通道数
        self.samples = data.shape[2]  # 采样点数
        self.data = data
        self.labels = labels
        self.fs = 1000  # 采样频率，默认1000Hz
        
    def get_trial(self, trial_idx):
        """获取单次试验的所有通道数据"""
        return self.data[trial_idx, :, :]
    
    def get_channel(self, channel_idx):
        """获取单个通道的所有试验数据"""
        return self.data[:, channel_idx, :]
    
    def get_trial_channel(self, trial_idx, channel_idx):
        """获取特定试验特定通道的数据"""
        return self.data[trial_idx, channel_idx, :]
    
    def filter_data(self, low_freq=8, high_freq=30):
        """对数据进行带通滤波"""
        nyq = self.fs * 0.5
        b = firwin(101, [low_freq/nyq, high_freq/nyq], pass_zero=False)
        filtered_data = np.zeros_like(self.data)
        
        for trial in range(self.trials):
            for channel in range(self.channels):
                filtered_data[trial, channel, :] = filtfilt(b, 1, self.data[trial, channel, :])
        
        self.data = filtered_data
        return self
    
    def extract_features(self, window_size=1000):
        """提取时间窗口特征"""
        n_windows = self.samples // window_size
        features = np.zeros((self.trials, self.channels, n_windows))
        
        for trial in range(self.trials):
            for channel in range(self.channels):
                for w in range(n_windows):
                    start = w * window_size
                    end = (w + 1) * window_size
                    window_data = self.data[trial, channel, start:end]
                    features[trial, channel, w] = np.var(window_data)  # 使用方差作为特征
                    
        return features
    
    def get_data_info(self):
        """获取数据基本信息"""
        return {
            'trials': self.trials,
            'channels': self.channels,
            'samples': self.samples,
            'unique_labels': np.unique(self.labels),
            'label_counts': np.bincount(self.labels.astype(int).flatten())
        }
    
    def to_cnn_format(self, window_size=1000, stride=None):
        """将数据转换为CNN可处理的格式
        Returns:
            x: shape (n_segments, 1, channels, window_size)
            y: shape (n_segments,)
        """
        if stride is None:
            stride = window_size
            
        # 计算可以得到多少个段
        n_segments = (self.samples - window_size) // stride + 1
        
        # 初始化结果数组
        x = np.zeros((self.trials * n_segments, 1, self.channels, window_size))
        y = np.zeros(self.trials * n_segments)
        
        # 分段处理数据
        for trial in range(self.trials):
            for seg in range(n_segments):
                start = seg * stride
                end = start + window_size
                
                # 获取当前窗口的数据
                window_data = self.data[trial, :, start:end]
                
                # 重组数据格式 (1, channels, window_size)
                segment_idx = trial * n_segments + seg
                x[segment_idx, 0, :, :] = window_data
                y[segment_idx] = self.labels[trial]
                
        return x, y
    
    def create_cnn_batch(self, batch_size=32):
        """创建CNN训练用的批次数据"""
        x, y = self.to_cnn_format()
        indices = np.random.permutation(len(x))
        
        for start_idx in range(0, len(x), batch_size):
            end_idx = min(start_idx + batch_size, len(x))
            batch_indices = indices[start_idx:end_idx]
            
            yield x[batch_indices], y[batch_indices]

    def plot_training_history(self, history):
        """绘制训练历史
        Args:
            history: 包含'loss', 'accuracy', 'val_loss', 'val_accuracy'的字典
        """
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_channel_activity(self, trial_idx=0):
        """绘制通道活动图
        Args:
            trial_idx: 要显示的试验索引
        """
        trial_data = self.get_trial(trial_idx)
        
        plt.figure(figsize=(15, 8))
        channels_to_plot = min(self.channels, 10)  # 最多显示10个通道
        
        for i in range(channels_to_plot):
            plt.subplot(channels_to_plot, 1, i+1)
            plt.plot(trial_data[i, :])
            plt.title(f'Channel {i+1}')
            plt.ylabel('Amplitude')
            if i == channels_to_plot-1:
                plt.xlabel('Time (samples)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_topographic_map(self, feature_values, channel_positions=None):
        """绘制地形图
        Args:
            feature_values: 每个通道的特征值
            channel_positions: 通道位置坐标字典（可选）
        """
        if channel_positions is None:
            # 默认圆形布局
            theta = np.linspace(0, 2*np.pi, self.channels)
            x = np.cos(theta)
            y = np.sin(theta)
        else:
            x = [pos[0] for pos in channel_positions.values()]
            y = [pos[1] for pos in channel_positions.values()]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, c=feature_values, cmap='RdBu_r', s=100)
        plt.colorbar(label='Activity')
        plt.title('Topographic Map')
        plt.axis('equal')
        plt.show()

import PlotConfusionMatrix as PCM

# 初始化数据结构
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []
# 添加可视化数据存储列表
visualization_data = []

# 获取当前文件路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 循环处理被试数据
for i in range(1, 51):  # 原始MATLAB代码的49:50对应Python的range(49,51)
    if i < 10:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-0{i}')
    else:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-{i}')
    
    # print(f'Processing folder: {sub_folder}')
    
    mat_files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

    # 初始化算法结果容器 - 修改初始化方式
    acc_results = []  # 存储准确率
    left_counts = []  # 存储左手分类结果
    right_counts = []  # 存储右手分类结果

    for mat_file in mat_files:
        file_path = os.path.join(sub_folder, mat_file)
        print(f'Loading file: {file_path}')
        
        # 加载MAT文件
        data = sio.loadmat(file_path)['eeg']
        
        # 数据预处理
        eeg_data = data['rawdata'][0, 0]
        labels = data['label'][0, 0]
        
        # 创建grid对象并进行处理
        grid = EEGGrid(eeg_data, labels)
        
        # 数据处理示例
        print("原始数据信息:", grid.get_data_info())
        
        # 滤波处理
        grid.filter_data(low_freq=8, high_freq=30)
        
        # 转换为CNN格式
        x_cnn, y_cnn = grid.to_cnn_format(window_size=1000)
        print(f"CNN数据格式: {x_cnn.shape}, 标签形状: {y_cnn.shape}")
        
        # 创建批次数据示例
        for batch_x, batch_y in grid.create_cnn_batch(batch_size=32):
            print(f"批次数据形状: {batch_x.shape}, 批次标签形状: {batch_y.shape}")
            break  # 仅显示第一个批次

        # 创建示例训练历史数据
        example_history = {
            'loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.85],
            'val_accuracy': [0.65, 0.75, 0.8]
        }
        
        # 收集需要可视化的数据
        vis_data = {
            'history': example_history,
            'confusion': {
                'y_true': labels.flatten(),
                'y_pred': np.random.choice([0, 1], size=len(labels.flatten()))  # 示例预测结果
            },
            'channel_data': grid.get_trial(0),  # 第一个试验的通道数据
            'topo_features': np.random.randn(grid.channels)  # 示例特征值
        }
        visualization_data.append(vis_data)
        
        # 只打印准确率等文本信息
        accuracy = np.mean(vis_data['confusion']['y_true'] == vis_data['confusion']['y_pred']) * 100
        print("\n=== 分类准确率 ===")
        print(f"准确率: {accuracy:.2f}%")

# 循环结束后统一进行可视化
print("\n=== 开始显示可视化结果 ===")
for idx, vis_data in enumerate(visualization_data):
    print(f"\n显示第 {idx+1} 组结果:")
    
    # 绘制训练历史
    grid.plot_training_history(vis_data['history'])
    
    # 绘制混淆矩阵
    grid.plot_confusion_matrix(vis_data['confusion']['y_true'], 
                             vis_data['confusion']['y_pred'])
    
    # 绘制通道活动
    plt.figure(figsize=(15, 8))
    channels_to_plot = min(grid.channels, 10)
    for i in range(channels_to_plot):
        plt.subplot(channels_to_plot, 1, i+1)
        plt.plot(vis_data['channel_data'][i, :])
        plt.title(f'Channel {i+1}')
        plt.ylabel('Amplitude')
        if i == channels_to_plot-1:
            plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()
    
    # 绘制地形图
    grid.plot_topographic_map(vis_data['topo_features'])

# 保存结果到本地
sio.savemat('FeatureData_cnn1.mat', {'data': data})
