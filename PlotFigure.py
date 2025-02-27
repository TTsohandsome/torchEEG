import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import create_info
from mne.io import RawArray
from mne.time_frequency import tfr_morlet
from mne.viz import plot_topomap
import scipy.io as sio
from scipy.signal import butter, filtfilt

def load_and_process_eeg(subject_range=(50,50), root_path="."):
    # 初始化参数，与 MATLAB 代码一致
    sample_rate = 500  # 采样率
    lower_freq = 8     # 频带下界
    upper_freq = 30    # 频带上界
    butter_order = 2   # 滤波器阶数
    smooth_window = 200 # 光滑系数

    # 确保 Figure_png 文件夹存在, 对应 MATLAB 代码中的 mkdir
    os.makedirs("Figure_png", exist_ok=True)
    # 确保 Figure_eps 文件夹存在, 对应 MATLAB 代码中的 mkdir
    os.makedirs("Figure_eps", exist_ok=True)

    # 循环读取文件夹, 对应 MATLAB 代码中的 for 循环
    for subj in range(subject_range[0], subject_range[1]+1):
        if subj < 10:
            SubFolderFiles = os.path.join(root_path, 'sourcedata', f'sub-0{subj}') # 对应 MATLAB 代码中的路径构建
        else:
            SubFolderFiles = os.path.join(root_path, 'sourcedata', f'sub-{subj}') # 对应 MATLAB 代码中的路径构建

        MatFiles = [f for f in os.listdir(SubFolderFiles) if f.endswith(".mat")] # 读取文件夹内的mat格式的文件, 对应 MATLAB 代码中的 dir
        # 循环读取 mat 文件, 对应 MATLAB 代码中的 for 循环
        for FileName in MatFiles:
            # 根据文件名下载数据, 对应 MATLAB 代码中的 load
            mat_data = sio.loadmat(os.path.join(SubFolderFiles, FileName))
            # 下载后的数据是一个结构体，data.EEG为脑电数据
            eeg0 = mat_data['eeg']['rawdata'] # 脑电数据, 对应 MATLAB 代码中的 eeg.rawdata
            label = mat_data['eeg']['label'] # 标签, 对应 MATLAB 代码中的 eeg.label
            eeg0 = np.transpose(eeg0, (0, 2, 1)) # 对应 MATLAB 代码中的 permute(eeg0,[1 3 2])
            data_trials = []
            for t in range(eeg0.shape[0]): # 对应 MATLAB 代码中的 for 循环
                data_trials.append(eeg0[t, :, :]) # 对应 MATLAB 代码中的 data = [data;squeeze(eeg0(t,:,:))];
            data = np.array(data_trials)

            # 绘图参数, 与前面定义的参数一致
            sampleRate = sample_rate;  # 采样率
            lower = lower_freq;   # 频带下界
            higher = upper_freq;  # 频带上界
            butterOrder = butter_order;  # 滤波器阶数
            smoothPara = smooth_window;   # 光滑系数

            # 绘制ERP图, 对应 MATLAB 代码中的 PlotERP
            plot_erp(data, label, lower, higher, sampleRate, butterOrder, smoothPara, subj, "Figure_png", "Figure_eps")
            # 绘制时频图, 对应 MATLAB 代码中的 PlotTimeFrequency
            plot_time_frequency(data, label, sampleRate, lower, higher, subj, "Figure_png", "Figure_eps")
            # 绘制脑地形图, 对应 MATLAB 代码中的 PlotBEAM
            plot_topography(data, label, sampleRate, lower, higher, subj, "Figure_png", "Figure_eps")

def plot_erp(data, label, lower, higher, sampleRate, butterOrder, smoothPara, subj, output_dir_png, output_dir_eps):
    """绘制ERP图，对应 MATLAB 代码中的 PlotERP 函数"""
    ch_names = [str(ch[0]) for ch in label[0,:]] if label.ndim > 1 else [str(ch) for ch in label] # 获取通道名
    info = create_info(ch_names=ch_names, sfreq=sampleRate, ch_types='eeg')
    epochs_data = data.mean(axis=0) # Trial averaging for ERP - adjust if needed
    if epochs_data.ndim == 2: # if averaging resulted in 2D data (channels x time)
        evoked = mne.EvokedArray(epochs_data, info)
    else: # if averaging resulted in 3D data (trials x channels x time) - unlikely after mean(axis=0) but for robustness
        evoked = mne.EvokedArray(epochs_data.mean(axis=0), info) # Average again if needed

    # 平滑处理 (简单移动平均)
    smoothed_data = np.convolve(evoked.data.mean(axis=0), np.ones(smoothPara)/smoothPara, mode='same')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    times = np.linspace(0, evoked.times[-1], num=len(smoothed_data)) # Adjust time vector if needed
    ax.plot(times, smoothed_data, linewidth=2)
    ax.set_title(f"Subject {subj} ERP")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)") # Adjust Y label as needed

    plt.savefig(os.path.join(output_dir_png, f"ERP{subj}.png")) # 对应 MATLAB 代码中的 saveas
    plt.savefig(os.path.join(output_dir_eps, f"ERP{subj}.eps")) # 对应 MATLAB 代码中的 saveas
    plt.close(fig) # 关闭图形

def plot_time_frequency(data, label, sampleRate, lower, higher, subj, output_dir_png, output_dir_eps):
    """绘制时频图，对应 MATLAB 代码中的 PlotTimeFrequency 函数"""
    ch_names = [str(ch[0]) for ch in label[0,:]] if label.ndim > 1 else [str(ch) for ch in label] # 获取通道名
    info = create_info(ch_names=ch_names, sfreq=sampleRate, ch_types='eeg')
    epochs_data = data # Use trial data directly for TF - adjust if you want average TF
    if epochs_data.ndim == 3:
        epochs = mne.EpochsArray(epochs_data, info)
    elif epochs_data.ndim == 2: # if data is already averaged or single trial 2D
        epochs = mne.EpochsArray(epochs_data[np.newaxis, :, :], info) # make it 3D

    freqs = np.logspace(np.log10(lower), np.log10(higher), num=20)
    n_cycles = freqs / 2.

    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True) # average=True for average power over epochs

    fig = power.plot([0], baseline=(-0.5, 0), mode='logratio', title=f"Subject {subj} Time-Frequency", show=False) # Plotting the first channel's TF map
    plt.savefig(os.path.join(output_dir_png, f"TF{subj}.png")) # 对应 MATLAB 代码中的 saveas
    plt.savefig(os.path.join(output_dir_eps, f"TF{subj}.eps")) # 对应 MATLAB 代码中的 saveas
    plt.close(fig.figure) # 关闭图形

def plot_topography(data, label, sampleRate, lower, higher, subj, output_dir_png, output_dir_eps):
    """绘制脑地形图，对应 MATLAB 代码中的 PlotBEAM 函数"""
    ch_names = [str(ch[0]) for ch in label[0,:]] if label.ndim > 1 else [str(ch) for ch in label] # 获取通道名
    info = create_info(ch_names=ch_names, sfreq=sampleRate, ch_types='eeg')
    avg_data = data.mean(axis=(0, 2)) # Average across trials and time for topography

    fig, ax = plt.subplots()
    plot_topomap(avg_data, info, axes=ax, show=False) # 使用 MNE 的 plot_topomap 绘制地形图
    ax.set_title(f"Subject {subj} Topography")
    plt.savefig(os.path.join(output_dir_png, f"Topo{subj}.png")) # 对应 MATLAB 代码中的 saveas
    plt.savefig(os.path.join(output_dir_eps, f"Topo{subj}.eps")) # 对应 MATLAB 代码中的 saveas
    plt.close(fig) # 关闭图形


if __name__ == "__main__":
    """
    主程序入口，与 MATLAB 代码的脚本功能类似
    文件结构需要满足:
    ├── current_folder (当前脚本所在文件夹)
    │   ├── sourcedata
    │   │   ├── sub-50 (例如 sub-50)
    │   │   │   └── your_eeg_data.mat (你的 .mat 数据文件)
    │   ├── PlotFigure.py (修改后的 Python 脚本)
    │   ├── Figure_png (用于保存 PNG 图片的文件夹，脚本会自动创建)
    │   └── Figure_eps (用于保存 EPS 图片的文件夹，脚本会自动创建)
    """
    load_and_process_eeg(subject_range=(50,50), # 加载和处理 EEG 数据，指定 subject 范围
                        root_path=os.path.dirname(os.path.abspath(__file__))) # 设置根路径为当前脚本所在目录
