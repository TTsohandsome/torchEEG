import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
import os.path as osp

import CSP_LDA
import FBCSP_SVM
import TSLDA_DGFMDM
import TWFB_DGFMDM
import PlotConfusionMatrix as PCM

# 初始化数据结构
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []

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
        # print(f'rawdata: {eeg_data.shape, labels.shape}')
        
        eeg_data = np.transpose(eeg_data, (0, 2, 1))  # 置换维度
        # print(f'after transpose: {eeg_data.shape}')
        
        # 数据重塑
        # processed_data = np.vstack([np.squeeze(eeg_data[t, ...]) for t in range(eeg_data.shape[0])])

        # 降维
        # 提取单层数据并压缩冗余维度（类似 MATLAB squeeze）
        single_trial = eeg_data[0, :, :].squeeze()  # 结果形状 (4000, 33)
        # 合并所有试验数据为二维矩阵（垂直拼接）
        data = np.concatenate([eeg_data[i].squeeze() for i in range(40)], axis=0)  # 结果形状 (160000, 33)
        # print(f'data_type: {data.shape}')

        # 算法参数
        Fs = 500
        LowFreq = 4
        UpFreq = 30

        # 调用分类算法
        acc0, left_num0, right_num0 = CSP_LDA.CSP_LDA(data, labels, Fs, LowFreq, UpFreq)
        acc1, left_num1, right_num1 = FBCSP_SVM.FBCSP_SVM(data, labels, Fs, LowFreq, UpFreq)
        acc2, left_num2, right_num2 = TSLDA_DGFMDM.TSLDA_DGFMDM(data, labels, Fs, LowFreq, UpFreq)
        acc3, left_num3, right_num3 = TWFB_DGFMDM.TWFB_DGFMDM(data, labels, Fs, LowFreq, UpFreq)
        
        # 收集每个算法的结果
        acc_results.extend([acc0, acc1, acc2, acc3])
        left_counts.extend([left_num0, left_num1, left_num2, left_num3])
        right_counts.extend([right_num0, right_num1, right_num2, right_num3])

    # 修改结果统计部分
    Acc.append([np.mean(acc_results[i::4]) for i in range(4)])  # 每4个取一个平均
    Left.append([np.mean([cnt[1] for cnt in left_counts[i::4]]) for i in range(4)])  # 取第二个元素的平均
    Right.append([np.mean([cnt[1] for cnt in right_counts[i::4]]) for i in range(4)])
    AllLeft.append([np.mean([cnt[0] for cnt in left_counts[i::4]]) for i in range(4)])  # 取第一个元素的平均
    AllRight.append([np.mean([cnt[0] for cnt in right_counts[i::4]]) for i in range(4)])

    # 统计准确率 - 修改为直接使用列表
    print("算法准确率:")
    for i, acc in enumerate([np.mean(acc_results[i::4]) for i in range(4)]):
        print(f'Algorithm {i}: {acc:.4f}')

# 修改 pad_matrix 函数
def pad_matrix(mat, target_rows=4):
    if not mat:  # 如果输入列表为空
        return [np.zeros(4) for _ in range(target_rows)]
    
    # 将列表转换为numpy数组以便处理
    mat_array = np.array(mat)
    current_rows = len(mat_array)
    
    if current_rows < target_rows:
        # 创建需要填充的行数
        padding = [np.zeros_like(mat_array[0]) for _ in range(target_rows - current_rows)]
        mat.extend(padding)
    
    return mat

# 修改结果统计部分
Acc = np.array(Acc)
Left = pad_matrix(Left)
Right = pad_matrix(Right)
AllLeft = pad_matrix(AllLeft)
AllRight = pad_matrix(AllRight)

# 调试信息
# print('Results shapes:')
# print('Acc shape:', np.array(Acc).shape)
# print('Left shape:', np.array(Left).shape)
# print('Right shape:', np.array(Right).shape)
# print('AllLeft shape:', np.array(AllLeft).shape)
# print('AllRight shape:', np.array(AllRight).shape)

# 绘制准确率柱状图
def plot_acc_bar(acc_data):
    plt.figure(figsize=(10,6))
    methods = ['CSP+LDA', 'FBCSP+SVM', 'TSLDA+DGFMDM', 'TWFB+DGFMDM']
    acc_means = np.mean(acc_data, axis=0) if len(acc_data.shape) > 1 else acc_data
    
    plt.bar(methods, acc_means)
    plt.title('Classification Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for i, v in enumerate(acc_means):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# 调试信息
# print('Size of Right:', len(Right))
# print('Size of AllRight:', len(AllRight))
# print('Size of Left:', len(Left))
# print('Size of AllLeft:', len(AllLeft))


# 确保混淆矩阵计算正确
CM = []
for i in range(4):
    try:
        row_sum_R = float(np.sum(Right[i]))
        row_sum_AR = float(np.sum(AllRight[i]))
        col_sum_L = float(np.sum(Left[i]))
        col_sum_AL = float(np.sum(AllLeft[i]))
        
        TP = min(row_sum_R, col_sum_L)
        FP = row_sum_R - TP
        FN = col_sum_L - TP
        TN = row_sum_AR - FP
        
        CM.append([[TP, FP], [FN, TN]])
    except Exception as e:
        print(f"Error processing confusion matrix for algorithm {i}: {e}")
        CM.append([[0, 0], [0, 0]])

# 确保 CM 数组的每个元素都是有效的混淆矩阵
for i in range(len(CM)):
    if len(CM[i]) != 2 or len(CM[i][0]) != 2:
        CM[i] = [[0,0],[0,0]]

PCM.plot_confusion_matrix(CM)

# 初始化 data 变量为字典
data = {
    'Accuracy': np.array(Acc),
    'ConfusionMatrix': CM
}

# 计算特征参数
def CalEvaluateIndex(CM):
    Kappa = 0.0
    Sensitivity = 0.0
    Precision = 0.0
    return Kappa, Sensitivity, Precision

Kappa, Sensitivity, Precision = CalEvaluateIndex(CM)
Feature = [Kappa, Sensitivity, Precision]

# 保存结果到本地
sio.savemat('FeatureData1.mat', {'data': data})

# 注意：运行此脚本前请确保已执行 installer.m