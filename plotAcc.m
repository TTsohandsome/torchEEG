% 获取当前文件的路径
fullpath = mfilename('fullpath');
[path,name]=fileparts(fullpath);
Acc = [];
Left = [];
Right = [];
AllLeft = [];
AllRight = []; % 确保 AllRight 变量已初始化

% 循环读取文件夹
for i = 49:50     % 依次读取50个文件
    if i<10
        SubFolderFiles = [path,'\sourcedata\sub-0' num2str(i) '\'];
    else
        SubFolderFiles = [path,'\sourcedata\sub-' num2str(i) '\'];
    end
    disp(['Processing folder: ', SubFolderFiles]);
    MatFiles = dir(fullfile([SubFolderFiles '*.mat'])); % 读取文件夹内的mat格式的文件
    
    acc0 = [];
    acc1 = [];
    acc2 = [];
    acc3 = [];
    left_num0 = zeros(2);
    left_num1 = zeros(2);
    left_num2 = zeros(2);
    left_num3 = zeros(2);
    right_num0 = zeros(2);
    right_num1 = zeros(2);
    right_num2 = zeros(2);
    right_num3 = zeros(2);

    for j = 1:length(MatFiles)
        % 文件名
        FileName = MatFiles(j).name;
        disp(['Loading file: ', FileName]);
        
        % 根据文件名下载数据
        load([SubFolderFiles FileName]);  % 下载后的数据是一个结构体，data.EEG为脑电数据
        eeg0 = eeg.rawdata;   % 脑电数据
        label = eeg.label;   % 标签
        eeg0 = permute(eeg0,[1 3 2]);  
        data = [];
        for t = 1:size(eeg0,1)
            data = [data;squeeze(eeg0(t,:,:))];
        end
        
        % 算法分类
        Fs = 500;
        LowFreq = 4;
        UpFreq = 30;

        % CSP+LDA算法
        [acc0,left_num0,right_num0] = CSP_LDA(data,label,Fs,LowFreq,UpFreq);
        
        % FBCSP+SVM算法
        [acc1,left_num1,right_num1] = FBCSP_SVM(data,label,Fs,LowFreq,UpFreq);
        
        % TSLDA+DGFMDM算法
        [acc2,left_num2,right_num2] = TSLDA_DGFMDM(data,label,Fs,LowFreq,UpFreq);
        
        % TWFB+DGFMDM算法
        [acc3,left_num3,right_num3] = TWFB_DGFMDM(data,label,Fs,LowFreq,UpFreq);
    end
    A = [mean(acc0);mean(acc1);mean(acc2);mean(acc3);];
    B1 = [mean(left_num0(:,2));mean(left_num1(:,2));mean(left_num2(:,2));mean(left_num3(:,2))];
    C1 = [mean(right_num0(:,2));mean(right_num1(:,2));mean(right_num2(:,2));mean(right_num3(:,2))];
    B2 = [mean(left_num0(:,1));mean(left_num1(:,1));mean(left_num2(:,1));mean(left_num3(:,1))];
    C2 = [mean(right_num0(:,1));mean(right_num1(:,1));mean(right_num2(:,1));mean(right_num3(:,1))];
    
    % 统计准确率
    Acc = [Acc,A];
    Left = [Left,B1];
    Right = [Right,C1];
    AllLeft = [AllLeft,B2];
    AllRight = [AllRight,C2];
    
    % 调试信息
    disp(['Acc0: ', num2str(mean(acc0))]);
    disp(['Acc1: ', num2str(mean(acc1))]);
    disp(['Acc2: ', num2str(mean(acc2))]);
    disp(['Acc3: ', num2str(mean(acc3))]);
end

% 确保数组大小不小于4
while size(Right, 1) < 4
    Right = [Right; zeros(1, size(Right, 2))];
end
while size(AllRight, 1) < 4
    AllRight = [AllRight; zeros(1, size(AllRight, 2))];
end
while size(Left, 1) < 4
    Left = [Left; zeros(1, size(Left, 2))];
end
while size(AllLeft, 1) < 4
    AllLeft = [AllLeft; zeros(1, size(AllLeft, 2))];
end

% 调试信息
disp('Acc:');
disp(Acc);
% disp('Left:');
% disp(Left);
% disp('Right:');
% disp(Right);
% disp('AllLeft:');
% disp(AllLeft);
% disp('AllRight:');
% disp(AllRight);

% 在当前工作目录及其子目录中搜索 PlotAccBar 函数的定义
search_result = dir(fullfile(pwd, '**', 'PlotAccBar.m'));

% 显示搜索结果
if isempty(search_result)
    disp('PlotAccBar 函数未找到');
else
    disp('PlotAccBar 函数定义文件:');
    for i = 1:length(search_result)
        disp(fullfile(search_result(i).folder, search_result(i).name));
    end
end

% 绘制准确率柱状图
figure(1)
PlotAccBar(Acc);

% 调试信息
disp('Size of Right:');
disp(size(Right));
disp('Size of AllRight:');
disp(size(AllRight));
disp('Size of Left:');
disp(size(Left));
disp('Size of AllLeft:');
disp(size(AllLeft));

% 绘制混淆矩阵
figure(2)
CM = cell(4,1); % 修改数组大小为 4x1
for i = 1:4
    if i <= size(Right, 1) && i <= size(AllRight, 1) && i <= size(Left, 1) && i <= size(AllLeft, 1)
        CM{i,1} = [floor(sum(Right(i,:))),floor(sum(AllRight(i,:)))-floor(sum(Right(i,:)));
            floor(sum(AllLeft(i,:)))-floor(sum(Left(i,:))),floor(sum(Left(i,:)))];
        % 调试信息
        disp(['CM{', num2str(i), ',1}:']);
        disp(CM{i,1});
    else
        warning(['Index ', num2str(i), ' exceeds array bounds.']);
        CM{i,1} = [0, 0; 0, 0]; % 初始化为零矩阵以避免错误
    end
end

% 调试信息
disp('Confusion Matrix (CM):');
disp(CM);

% 确保 CM 数组的每个元素都是有效的混淆矩阵
for i = 1:length(CM)
    if isempty(CM{i,1}) || ~ismatrix(CM{i,1}) || size(CM{i,1}, 1) ~= 2 || size(CM{i,1}, 2) ~= 2
        error(['Invalid confusion matrix at index ', num2str(i)]);
    end
end

PlotConfusionMatrix(CM);

% 初始化 data 变量为结构体
data = struct();

% 计算特征参数
[Kappa,Sensitivity,Precision] = CalEvaluateIndex(CM);
Feature = [Kappa,Sensitivity,Precision];

% 将计算出来的准确率结果存到本地
data.Accuracy = Acc;
data.ConfusionMatrix = CM;
save('FeatureData1.mat','data');

% 注意run这个之前一定要先运行installer.m