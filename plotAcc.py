# 获取当前文件的路�
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
fullpath = mfilename('fullpath')
path,name = os.path.split(fullpath)[0],os.path.splitext(os.path.split(fullpath)[1])[0],os.path.splitext(os.path.split(fullpath)[1])[1]
Acc = []
Left = []
Right = []
AllLeft = []
AllRight = []

# ��读取文件�
for i in np.arange(49,50+1).reshape(-1):
    if i < 10:
        SubFolderFiles = np.array([path,'\sourcedata\sub-0',num2str(i),'\'])
    else:
        SubFolderFiles = np.array([path,'\sourcedata\sub-',num2str(i),'\'])
    print(np.array(['Processing folder: ',SubFolderFiles]))
    MatFiles = dir(fullfile(np.array([SubFolderFiles,'*.mat'])))
    acc0 = []
    acc1 = []
    acc2 = []
    acc3 = []
    left_num0 = np.zeros((2,2))
    left_num1 = np.zeros((2,2))
    left_num2 = np.zeros((2,2))
    left_num3 = np.zeros((2,2))
    right_num0 = np.zeros((2,2))
    right_num1 = np.zeros((2,2))
    right_num2 = np.zeros((2,2))
    right_num3 = np.zeros((2,2))
    for j in np.arange(1,len(MatFiles)+1).reshape(-1):
        # 文件�
        FileName = MatFiles(j).name
        print(np.array(['Loading file: ',FileName]))
        # 根据文件名下载数�
        scipy.io.loadmat(np.array([SubFolderFiles,FileName]))
        eeg0 = eeg.rawdata
        label = eeg.label
        eeg0 = permute(eeg0,np.array([1,3,2]))
        data = []
        for t in np.arange(1,eeg0.shape[1-1]+1).reshape(-1):
            data = np.array([[data],[np.squeeze(eeg0(t,:,:))]])
        # 算法分类
        Fs = 500
        LowFreq = 4
        UpFreq = 30
        # CSP+LDA算法
        acc0,left_num0,right_num0 = CSP_LDA(data,label,Fs,LowFreq,UpFreq)
        # FBCSP+SVM算法
        acc1,left_num1,right_num1 = FBCSP_SVM(data,label,Fs,LowFreq,UpFreq)
        # TSLDA+DGFMDM算法
        acc2,left_num2,right_num2 = TSLDA_DGFMDM(data,label,Fs,LowFreq,UpFreq)
        # TWFB+DGFMDM算法
        acc3,left_num3,right_num3 = TWFB_DGFMDM(data,label,Fs,LowFreq,UpFreq)
    A = np.array([[mean(acc0)],[mean(acc1)],[mean(acc2)],[mean(acc3)]])
    B1 = np.array([[mean(left_num0(:,2))],[mean(left_num1(:,2))],[mean(left_num2(:,2))],[mean(left_num3(:,2))]])
    C1 = np.array([[mean(right_num0(:,2))],[mean(right_num1(:,2))],[mean(right_num2(:,2))],[mean(right_num3(:,2))]])
    B2 = np.array([[mean(left_num0(:,1))],[mean(left_num1(:,1))],[mean(left_num2(:,1))],[mean(left_num3(:,1))]])
    C2 = np.array([[mean(right_num0(:,1))],[mean(right_num1(:,1))],[mean(right_num2(:,1))],[mean(right_num3(:,1))]])
    # 统�准��
    Acc = np.array([Acc,A])
    Left = np.array([Left,B1])
    Right = np.array([Right,C1])
    AllLeft = np.array([AllLeft,B2])
    AllRight = np.array([AllRight,C2])
    # 调试信息
    print(np.array(['Acc0: ',num2str(mean(acc0))]))
    print(np.array(['Acc1: ',num2str(mean(acc1))]))
    print(np.array(['Acc2: ',num2str(mean(acc2))]))
    print(np.array(['Acc3: ',num2str(mean(acc3))]))

# �保数组大小不小于4
while Right.shape[1-1] < 4:

    Right = np.array([[Right],[np.zeros((1,Right.shape[2-1]))]])


while AllRight.shape[1-1] < 4:

    AllRight = np.array([[AllRight],[np.zeros((1,AllRight.shape[2-1]))]])


while Left.shape[1-1] < 4:

    Left = np.array([[Left],[np.zeros((1,Left.shape[2-1]))]])


while AllLeft.shape[1-1] < 4:

    AllLeft = np.array([[AllLeft],[np.zeros((1,AllLeft.shape[2-1]))]])


# 调试信息
print('Acc:')
print(Acc)
# disp('Left:');
# disp(Left);
# disp('Right:');
# disp(Right);
# disp('AllLeft:');
# disp(AllLeft);
# disp('AllRight:');
# disp(AllRight);

# 在当前工作目录及其子�录中搜索 PlotAccBar 函数的定�
search_result = dir(fullfile(pwd,'**','PlotAccBar.m'))
# 显示搜索结果
if len(search_result)==0:
    print('PlotAccBar 函数�找到')
else:
    print('PlotAccBar 函数定义文件:')
    for i in np.arange(1,len(search_result)+1).reshape(-1):
        print(fullfile(search_result(i).folder,search_result(i).name))

# 绘制准确率柱状图
plt.figure(1)
PlotAccBar(Acc)
# 调试信息
print('Size of Right:')
print(Right.shape)
print('Size of AllRight:')
print(AllRight.shape)
print('Size of Left:')
print(Left.shape)
print('Size of AllLeft:')
print(AllLeft.shape)
# 绘制混淆矩阵
plt.figure(2)
CM = cell(4,1)

for i in np.arange(1,4+1).reshape(-1):
    if i <= Right.shape[1-1] and i <= AllRight.shape[1-1] and i <= Left.shape[1-1] and i <= AllLeft.shape[1-1]:
        CM[i,1] = np.array([[int(np.floor(sum(Right(i,:)))),int(np.floor(sum(AllRight(i,:)))) - int(np.floor(sum(Right(i,:))))],[int(np.floor(sum(AllLeft(i,:)))) - int(np.floor(sum(Left(i,:)))),int(np.floor(sum(Left(i,:))))]])
        # 调试信息
        print(np.array(['CM{',num2str(i),',1}:']))
        print(CM[i,1])
    else:
        warnings.warn(np.array(['Index ',num2str(i),' exceeds array bounds.']))
        CM[i,1] = np.array([[0,0],[0,0]])

# 调试信息
print('Confusion Matrix (CM):')
print(CM)
# �� CM 数组的每�元素都是有效的混淆矩�
for i in np.arange(1,len(CM)+1).reshape(-1):
    if len(CM[i,1])==0 or not True  or CM[i,1].shape[1-1] != 2 or CM[i,1].shape[2-1] != 2:
        raise Exception(np.array(['Invalid confusion matrix at index ',num2str(i)]))

PlotConfusionMatrix(CM)
# 初�化 data 变量为结构体
data = struct()
# 计算特征参数
Kappa,Sensitivity,Precision = CalEvaluateIndex(CM)
Feature = np.array([Kappa,Sensitivity,Precision])
# 将�算出来的准�率结果存到本�
data.Accuracy = Acc
data.ConfusionMatrix = CM
save('FeatureData1.mat','data')
# 注意run这个之前�定�先运�installer.m