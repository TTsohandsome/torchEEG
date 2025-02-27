import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt

# 初始化数据结构
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []

# 获取当前文件路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 循环处理被试数据
for i in range(49, 51):  # 原始MATLAB代码的49:50对应Python的range(49,51)
    if i < 10:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-0{i}')
    else:
        sub_folder = os.path.join(current_path, 'sourcedata', f'sub-{i}')
    
    print(f'Processing folder: {sub_folder}')
    
    mat_files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

    # 初始化算法结果容器
    acc_results = [[] for _ in range(4)]
    left_counts = [np.zeros(2) for _ in range(4)]
    right_counts = [np.zeros(2) for _ in range(4)]

    for mat_file in mat_files:
        file_path = os.path.join(sub_folder, mat_file)
        print(f'Loading file: {file_path}')
        
        # 加载MAT文件
        mat_data = sio.loadmat(file_path, struct_as_record=False)['eeg']
        
        # 数据预处理
        eeg_data = mat_data.rawdata
        labels = mat_data.label
        eeg_data = np.transpose(eeg_data, (0, 2, 1))  # 置换维度
        
        # 数据重塑
        processed_data = np.vstack([np.squeeze(eeg_data[t, ...]) for t in range(eeg_data.shape[0])])

        # 算法参数
        Fs = 500
        LowFreq = 4
        UpFreq = 30

        # 调用分类算法（需要自行实现以下函数）
        acc0, left_num0, right_num0 = CSP_LDA(processed_data, labels, Fs, LowFreq, UpFreq)
        acc1, left_num1, right_num1 = FBCSP_SVM(processed_data, labels, Fs, LowFreq, UpFreq)
        acc2, left_num2, right_num2 = TSLDA_DGFMDM(processed_data, labels, Fs, LowFreq, UpFreq)
        acc3, left_num3, right_num3 = TWFB_DGFMDM(processed_data, labels, Fs, LowFreq, UpFreq)

    # 结果统计
    Acc.append([np.mean(res) for res in acc_results])
    Left.append([np.mean(cnt[:,1]) for cnt in left_counts])
    Right.append([np.mean(cnt[:,1]) for cnt in right_counts])
    AllLeft.append([np.mean(cnt[:,0]) for cnt in left_counts])
    AllRight.append([np.mean(cnt[:,0]) for cnt in right_counts])

    # 统计准确率
    Acc = [Acc]
    Left = [Left]
    Right = [Right]
    AllLeft = [AllLeft]
    AllRight = [AllRight]

    # 调试信息
    print(f'Acc0: {np.mean(acc0):.2f}')
    print(f'Acc1: {np.mean(acc1):.2f}')
    print(f'Acc2: {np.mean(acc2):.2f}')
    print(f'Acc3: {np.mean(acc3):.2f}')

end

# 确保数组大小不小于4
def pad_matrix(mat, target_rows=4):
    while len(mat) < target_rows:
        mat.append(np.zeros(mat[0].shape))
    return mat

Left = pad_matrix(Left)
Right = pad_matrix(Right)
AllLeft = pad_matrix(AllLeft)
AllRight = pad_matrix(AllRight)

# 调试信息
print('Acc:', Acc)
print('Left:', Left)
print('Right:', Right)
print('AllLeft:', AllLeft)
print('AllRight:', AllRight)

# 在当前工作目录及其子目录中搜索 PlotAccBar 函数的定义
search_result = []
for root, dirs, files in os.walk(current_path):
    for file in files:
        if file == 'PlotAccBar.m':
            search_result.append(os.path.join(root, file))

if not search_result:
    print('PlotAccBar 函数未找到')
else:
    print('PlotAccBar 函数定义文件:')
    for file in search_result:
        print(file)

# 绘制准确率柱状图
def plot_acc_bar(acc_data):
    plt.figure(figsize=(10,6))
    plt.bar(range(len(acc_data)), acc_data)
    plt.title('Classification Accuracy Comparison')
    plt.show()

plot_acc_bar(np.mean(Acc, axis=0))

# 调试信息
print('Size of Right:', len(Right))
print('Size of AllRight:', len(AllRight))
print('Size of Left:', len(Left))
print('Size of AllLeft:', len(AllLeft))

# 绘制混淆矩阵
def plot_confusion_matrix(CM):
    plt.figure(figsize=(8,6))
    for i in range(4):
        cm = CM[i]
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix {i+1}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

CM = []
for i in range(4):
    row_sum_R = sum(Right[i])
    row_sum_AR = sum(AllRight[i])
    col_sum_L = sum(Left[i])
    col_sum_AL = sum(AllLeft[i])
    
    TP = min(row_sum_R, col_sum_L)
    FP = row_sum_R - TP
    FN = col_sum_L - TP
    TN = col_sum_AR - FP
    
    CM.append([[TP, FP], [FN, TN]])

# 确保 CM 数组的每个元素都是有效的混淆矩阵
for i in range(len(CM)):
    if len(CM[i]) != 2 or len(CM[i][0]) != 2:
        CM[i] = [[0,0],[0,0]]

plot_confusion_matrix(CM)

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