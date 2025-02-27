import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(confusion_matrices):
    """
    绘制混淆矩阵
    Args:
        confusion_matrices: list of 2x2 confusion matrices
    """
    # 确保输入数据是numpy数组格式
    matrices = [np.array(cm) for cm in confusion_matrices]
    
    # 初始化参数
    titles = ['CSP+LDA', 'FBCSP+SVM', 'TSLDA+DGFMDM', 'TWFB+DGFMDM']
    labels = ['Right Hand', 'Left Hand']
    
    # 定义颜色映射
    mincolor = np.array([217, 83, 25]) / 255.0
    maxcolor = np.array([50, 114, 189]) / 255.0
    colors = np.vstack([np.linspace(mincolor, maxcolor, 33)])
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    # 创建画布和子图布局 - 减小图像大小
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.5)

    # 遍历四个子图
    for idx, mat in enumerate(matrices):
        ax = fig.add_subplot(gs[idx])
        
        # 绘制热力图
        sns.heatmap(mat, annot=False, cmap=cmap, cbar=False, square=True,
                    linewidths=0.5, linecolor='gray', ax=ax)
        
        # 添加数值标签 - 减小字体大小
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, f"{int(mat[i, j])}",
                        ha='center', va='center',
                        fontsize=12, fontfamily='Times New Roman')
        
        # 设置坐标轴 - 减小字体大小
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold', rotation=90, va='center')
        ax.set_xlabel('Predict class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual class', fontsize=12, fontweight='bold')
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        
        # 添加颜色条
        vmin = np.min(mat)
        vmax = np.max(mat)
        cax = fig.add_axes([ax.get_position().x1+0.02,
                           ax.get_position().y0,
                           0.015,  # 减小颜色条宽度
                           ax.get_position().height])
        
        # 设置颜色条刻度 - 减小字体大小
        ticks = np.linspace(vmin, vmax, 5)
        cb = plt.colorbar(ax.collections[0], cax=cax, ticks=ticks)
        cb.ax.tick_params(labelsize=10)
        cb.outline.set_linewidth(0.5)  # 减小边框宽度

    # 整体布局调整
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

# 测试代码（可选）
if __name__ == "__main__":
    # 示例数据
    test_matrices = [
        np.array([[250, 50], [60, 240]]),
        np.array([[220, 80], [70, 230]]),
        np.array([[200, 100], [90, 210]]),
        np.array([[180, 120], [110, 190]])
    ]
    plot_confusion_matrix(test_matrices)