import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # No longer directly used, but kept for comparison if needed
from sklearn.neural_network import MLPClassifier # Import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # for saving sklearn models

class EEGGrid:
    # ... (EEGGrid class remains the same) ...
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


def plot_confusion_matrix_combined(all_cms, classes, save_path='.'):
    """绘制平均混淆矩阵"""
    avg_cm = np.mean(all_cms, axis=0) # 计算平均混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues', # fmt='.2f' for float annotation
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Average Confusion Matrix (All Subjects)')

    filename = os.path.join(save_path, 'average_confusion_matrix_all_subjects.png')
    plt.savefig(filename)
    plt.close()

def plot_accuracies(all_accuracies, subject_ids, save_path='.'):
    """绘制所有被试的准确率条形图"""
    plt.figure(figsize=(12, 6))
    plt.bar(subject_ids, all_accuracies)
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Subject')
    plt.xticks(subject_ids) # 确保每个 subject id 都有刻度

    filename = os.path.join(save_path, 'accuracy_per_subject.png')
    plt.savefig(filename)
    plt.close()

def plot_train_test_accuracies_line(train_accuracies, test_accuracies, subject_ids, save_path='.'):
    """绘制训练集和测试集准确率对比折线图"""
    plt.figure(figsize=(12, 6))
    plt.plot(subject_ids, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(subject_ids, test_accuracies, label='Test Accuracy', marker='x')

    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Training vs. Test Accuracy per Subject')
    plt.xticks(subject_ids) # 设置x轴刻度为被试ID
    plt.legend()
    plt.grid(True) # 添加网格线，可选
    plt.tight_layout()

    filename = os.path.join(save_path, 'train_test_accuracy_line_per_subject.png')
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(train_losses, val_losses, val_accuracies, subject_id, save_path='.'):
    """绘制学习曲线 (loss 和 accuracy vs. epochs)"""
    epochs = range(1, len(train_losses) + 1) # Epoch numbers for x-axis

    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first subplot
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Subject {subject_id} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second subplot
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='x') # Removed training accuracy plot for now
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Subject {subject_id} - Validation Accuracy') # Modified title
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    filename = os.path.join(save_path, f'learning_curve_sub-{subject_id}.png')
    plt.savefig(filename)
    plt.close()


# 初始化数据结构
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []

# 获取当前文件路径
current_path = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_path, 'output_visualizations')
os.makedirs(output_folder, exist_ok=True)

all_confusion_matrices = [] # 存储所有被试的混淆矩阵
all_accuracies = [] # 存储所有被试的测试集准确率
all_train_accuracies = [] # 存储所有被试的训练集准确率
subject_ids = [] # 存储被试ID列表，用于x轴标签

# 循环处理被试数据
for i in range(1, 51):
    if i < 10:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-0{i}')
    else:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-{i}')
    subject_ids.append(i) # 添加被试ID

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

        # print("原始数据信息:", grid.get_data_info())

        grid.filter_data(low_freq=8, high_freq=30)

        features_x, features_y = grid.to_features_format(window_size=1000, stride=500) # Example stride
        # print(f"Features 输入数据 x 的形状: {features_x.shape}")
        # print(f"Features 输入标签 y 的形状: {features_y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.2, random_state=42, stratify=features_y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train) # Scale training data
        X_test_scaled = scaler.transform(X_test) # Apply same scaling to test data

        # Use MLPClassifier with SGD solver and increased max_iter, and track learning curve
        # validation_fraction=0.1: Use 10% of training data as validation set for early stopping and tracking
        # early_stopping=True: Stop training when validation score does not improve for n_iter_no_change epochs (default 10)
        model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='sgd',
                              random_state=42, max_iter=500, alpha=0.0001, learning_rate_init=0.01,
                              validation_fraction=0.1, early_stopping=True) # Added validation and early stopping

        model.fit(X_train_scaled, y_train)

        # Access training loss and validation scores from the model's attributes
        train_losses = model.loss_curve_  # Training loss at each epoch
        val_scores = model.validation_scores_ # Validation accuracy at each epoch
        val_losses = [-score for score in val_scores] # Approximate validation loss (negate accuracy, not true loss but indicative)

        # Calculate train accuracy (final epoch)
        train_accuracy = model.score(X_train_scaled, y_train)
        print(f'Subject {i} - Train Accuracy: {train_accuracy:.4f}')
        all_train_accuracies.append(train_accuracy)

        # Calculate test accuracy
        test_accuracy = model.score(X_test_scaled, y_test)
        print(f'Subject {i} - Test Accuracy: {test_accuracy:.4f}')
        acc_results.append(test_accuracy)
        all_accuracies.append(test_accuracy) # 存储测试集准确率

        y_pred = model.predict(X_test_scaled)

        class_names = [str(c) for c in np.unique(labels)]

        cm = confusion_matrix(y_test, y_pred) # 计算混淆矩阵
        all_confusion_matrices.append(cm) # 存储混淆矩阵

        print(f"Subject {i} - Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Subject {i} - Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Plot learning curve for each subject
        plot_learning_curve(train_losses, val_losses, val_scores, subject_id=i, save_path=output_folder) # Corrected function call, using val_scores for validation accuracy
        print(f"Learning curve plot for subject {i} saved to: {output_folder}/learning_curve_sub-{i}.png")


        # 保存模型 (可选)
        # model_filename = os.path.join(output_folder, f'eeg_mlp_sgd_model_sub-{i}.pkl') # Changed filename to mlp_sgd
        # joblib.dump(model, model_filename)


# 保存结果到本地
sio.savemat('FeatureData_cnn2_mlp_sgd_history.mat', {'data': data}) # Changed filename to mlp_sgd_history

print("All subjects' test accuracies:", all_accuracies)
print("Average test accuracy:", np.mean(all_accuracies))
print("Average train accuracy:", np.mean(all_train_accuracies))

# 绘制平均混淆矩阵
plot_confusion_matrix_combined(all_confusion_matrices, class_names, save_path=output_folder)
print(f"Average confusion matrix plot saved to: {output_folder}/average_confusion_matrix_all_subjects.png")

# 绘制准确率条形图 (仅测试集准确率)
plot_accuracies(all_accuracies, subject_ids, save_path=output_folder)
print(f"Accuracy per subject plot saved to: {output_folder}/accuracy_per_subject.png")

# 绘制训练集和测试集准确率对比折线图
plot_train_test_accuracies_line(all_train_accuracies, all_accuracies, subject_ids, save_path=output_folder)
print(f"Train vs Test Accuracy line plot saved to: {output_folder}/train_test_accuracy_line_per_subject.png")


# 合并所有混淆矩阵为一个矩阵 (Summation)
combined_confusion_matrix = np.sum(all_confusion_matrices, axis=0)
print("\nCombined Confusion Matrix (Sum of all subjects' CMs):\n", combined_confusion_matrix)

print("Visualizations are saved in the 'output_visualizations' folder.")
