from torcheeg.datasets import BaseDataset
import os
import numpy as np

class StrokeEEGDataset(BaseDataset):
    def __init__(self, root_path='sourcedata', chunk_size=512, overlap=0.2, transform=None):
        super().__init__()
        self.root_path = root_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.transform = transform
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        self.data = []
        self.labels = []
        
        # 遍历数据文件夹
        for label_dir in os.listdir(self.root_path):
            label_path = os.path.join(self.root_path, label_dir)
            if not os.path.isdir(label_path):
                continue
                
            label = 1 if 'stroke' in label_dir.lower() else 0
            
            for file in os.listdir(label_path):
                if file.endswith('.npy'):  # 假设数据以numpy格式存储
                    eeg_data = np.load(os.path.join(label_path, file))
                    # 分割数据为固定大小的片段
                    segments = self._segment_signal(eeg_data)
                    self.data.extend(segments)
                    self.labels.extend([label] * len(segments))
    
    def _segment_signal(self, signal):
        segments = []
        step = int(self.chunk_size * (1 - self.overlap))
        for i in range(0, len(signal) - self.chunk_size + 1, step):
            segment = signal[i:i + self.chunk_size]
            segments.append(segment)
        return segments
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)
