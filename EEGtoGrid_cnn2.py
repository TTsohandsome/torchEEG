import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib  # for saving sklearn models

class EEGGrid:
    def __init__(self, data, labels):
        self.trials = data.shape[0]
        self.channels = data.shape[1]
        self.samples = data.shape[2]
        self.data = data
        self.labels = labels
        self.fs = 1000

    def get_trial(self, trial_idx):
        return self.data[trial_idx, :, :]

    def get_channel(self, channel_idx):
        return self.data[:, channel_idx, :]

    def get_trial_channel(self, trial_idx, channel_idx):
        return self.data[trial_idx, channel_idx, :]

    def filter_data(self, low_freq=8, high_freq=30):
        nyq = self.fs * 0.5
        b = firwin(101, [low_freq/nyq, high_freq/nyq], pass_zero=False)
        filtered_data = np.zeros_like(self.data)

        for trial in range(self.trials):
            for channel in range(self.channels):
                filtered_data[trial, channel, :] = filtfilt(b, 1, self.data[trial, channel, :])

        self.data = filtered_data
        return self

    def extract_features(self, window_size=1000):
        n_windows = self.samples // window_size
        features = np.zeros((self.trials, self.channels, n_windows))

        for trial in range(self.trials):
            for channel in range(self.channels):
                for w in range(n_windows):
                    start = w * window_size
                    end = (w + 1) * window_size
                    window_data = self.data[trial, channel, start:end]
                    features[trial, channel, w] = np.var(window_data)

        return features

    def get_data_info(self):
        return {
            'trials': self.trials,
            'channels': self.channels,
            'samples': self.samples,
            'unique_labels': np.unique(self.labels),
            'label_counts': np.bincount(self.labels.astype(int).flatten())
        }

    def to_features_format(self, window_size=1000, stride=None):
        if stride is None:
            stride = window_size

        n_segments = (self.samples - window_size) // stride + 1
        if n_segments <= 0:
            n_segments = 1

        x = np.zeros((self.trials * n_segments, self.channels * window_size))
        y = np.zeros(self.trials * n_segments)

        for trial in range(self.trials):
            for seg in range(n_segments):
                start = seg * stride
                end = min(start + window_size, self.samples)
                window_data = self.data[trial, :, start:end]
                window_len = window_data.shape[-1]

                segment_idx = trial * n_segments + seg
                if window_len == window_size:
                    x[segment_idx, :] = window_data.reshape(-1)
                else:
                    padded_window_data = np.zeros((self.channels, window_size))
                    padded_window_data[:, :window_len] = window_data
                    x[segment_idx, :] = padded_window_data.reshape(-1)

                y[segment_idx] = self.labels[trial]

        return x, y


def plot_history(history, subject_id, save_path='.'):
    """绘制训练历史曲线并保存图像"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    filename = os.path.join(save_path, f'accuracy_loss_sub-{subject_id}.png')
    plt.savefig(filename)
    plt.close() # 关闭图像，防止显示


def plot_confusion_matrix(y_true, y_pred, classes, subject_id, save_path='.'):
    """绘制混淆矩阵并保存图像"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Subject {subject_id}')

    filename = os.path.join(save_path, f'confusion_matrix_sub-{subject_id}.png')
    plt.savefig(filename)
    plt.close() # 关闭图像，防止显示


# 初始化数据结构
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []

# 获取当前文件路径
current_path = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_path, 'output_visualizations') # 保存图像的文件夹
os.makedirs(output_folder, exist_ok=True) # 确保文件夹存在

# 存储图像文件路径的列表
visualization_files = []

# 循环处理被试数据
for i in range(1, 51):
    if i < 10:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-0{i}')
    else:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-{i}')

    mat_files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

    acc_results = []
    left_counts = []
    right_counts = []

    for mat_file in mat_files:
        file_path = os.path.join(sub_folder, mat_file)
        print(f'Loading file: {file_path}')

        data = sio.loadmat(file_path)['eeg']

        eeg_data = data['rawdata'][0, 0]
        labels = data['label'][0, 0]

        grid = EEGGrid(eeg_data, labels)

        print("原始数据信息:", grid.get_data_info())

        grid.filter_data(low_freq=8, high_freq=30)

        features_x, features_y = grid.to_features_format(window_size=1000, stride=500) # Example stride
        print(f"Features 输入数据 x 的形状: {features_x.shape}")
        print(f"Features 输入标签 y 的形状: {features_y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.2, random_state=42, stratify=features_y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f'Subject {i} - Test Accuracy: {accuracy:.4f}')
        acc_results.append(accuracy)

        y_pred = model.predict(X_test)

        class_names = [str(c) for c in np.unique(labels)]

        # 保存可视化图像，并将文件路径添加到列表
        cm_filename = os.path.join(output_folder, f'confusion_matrix_sub-{i}.png')
        plot_confusion_matrix(y_test, y_pred, class_names, i, save_path=output_folder)
        visualization_files.append(cm_filename)

        print(f"Subject {i} - Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Subject {i} - Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # 保存模型 (可选)
        # model_filename = os.path.join(output_folder, f'eeg_logistic_regression_model_sub-{i}.pkl')
        # joblib.dump(model, model_filename)


# 保存结果到本地
sio.savemat('FeatureData2.mat', {'data': data})

print("All subjects' test accuracies:", acc_results)
print("Average test accuracy:", np.mean(acc_results))

# 统一显示所有保存的图像
for file_path in visualization_files:
    img = plt.imread(file_path)
    plt.figure()
    plt.imshow(img)
    plt.title(os.path.basename(file_path)) # 图像标题为文件名
    plt.axis('off') # 不显示坐标轴
plt.show()
