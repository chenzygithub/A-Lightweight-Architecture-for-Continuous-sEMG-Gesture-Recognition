import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== 辅助约束函数 ====================
def _inv_softplus(x):
    return np.log(np.exp(x) - 1.0)

def _inv_sigmoid(x):
    return np.log(x / (1.0 - x))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==================== 数据集定义 ====================
def segment_stream(signal, labels, window_len=150, stride=20):
    T = signal.shape[0]
    n = (T - window_len) // stride + 1
    windows = np.stack(
        [signal[i * stride : i * stride + window_len] for i in range(n)], axis=0
    )
    window_labels = np.stack(
        [labels[i * stride : i * stride + window_len] for i in range(n)], axis=0
    )
    return windows, window_labels

class SEMGSequenceDataset(Dataset):
    def __init__(self, windows, window_labels, seq_len=9, anchor_point=149, seq_stride=1):
        self.x = torch.tensor(windows, dtype=torch.float32)
        self.y = torch.tensor(window_labels[:, anchor_point], dtype=torch.long)
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.n_seq = (len(self.x) - seq_len) // seq_stride + 1

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        s = idx * self.seq_stride
        x_seq = self.x[s : s + self.seq_len].permute(0, 2, 1)  # (Seq_Len, 8, 150)
        y_seq = self.y[s : s + self.seq_len]                  # (Seq_Len,)
        return x_seq, y_seq

# ==================== 模型结构 ====================
class StatefulConvGLUBlock(nn.Module):
    def __init__(self, d_model, num_classes, expand=2,
                 alpha_init=10.0, theta_init=0.6, beta_init=10.0, lam_init=0.5):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=2, groups=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.probe_head = nn.Linear(d_model, num_classes)
        self.memory = None

        self.alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))
        self.beta_raw  = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self.theta_raw = nn.Parameter(torch.tensor(_inv_sigmoid(theta_init)))
        self.lam_raw   = nn.Parameter(torch.tensor(_inv_sigmoid(lam_init)))

    def reset_memory(self):
        self.memory = None

    def forward(self, x):
        residual = x
        _, seq_len, _ = x.shape
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = F.silu(x_conv.transpose(1, 2))

        y = x_conv * F.silu(z_branch)
        y = self.out_proj(y) + residual

        pooled = y.mean(dim=1)
        if self.memory is None:
            self.memory = torch.zeros_like(pooled)
        mem = self.memory

        logits_probe = self.probe_head(pooled)
        conf = torch.softmax(logits_probe, dim=-1).max(dim=-1).values
        sim = F.cosine_similarity(mem, pooled, dim=-1, eps=1e-8)

        alpha = F.softplus(self.alpha_raw)
        beta  = F.softplus(self.beta_raw)
        theta = torch.sigmoid(self.theta_raw)
        lam   = torch.sigmoid(self.lam_raw)

        gate = torch.sigmoid(alpha * (conf - theta)) * torch.sigmoid(beta * (1.0 - sim))
        gate = gate.unsqueeze(-1)
        new_mem = (1.0 - gate) * mem + gate * pooled
        self.memory = new_mem
        out = (1.0 - lam) * y + lam * new_mem.unsqueeze(1)
        return out


class StatefulConvGLUBaseline(nn.Module):
    def __init__(self, num_classes, in_channels=8, d_model=64, num_layers=2, seq_len=9, **mem_kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            StatefulConvGLUBlock(d_model=d_model, num_classes=num_classes, **mem_kwargs)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def reset_memory(self):
        for blk in self.blocks:
            blk.reset_memory()

    def forward(self, x_seq, reset=True):
        if reset:
            self.reset_memory()
        logits_list = []
        for t in range(x_seq.size(1)):
            xt = x_seq[:, t]
            h = self.input_proj(xt).transpose(1, 2)
            h = self.input_norm(h)
            for blk in self.blocks:
                h = blk(h)
            feat = h.mean(dim=1)
            logits_list.append(self.classifier(feat))
        return torch.stack(logits_list, dim=1)


# ==================== 训练入口 ====================
def train_model(data_file="emg_data_labeled.npy", model_save_path="convglu_stateful_best.pth",
                epochs=20, batch_size=128, lr=1e-3, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练硬件设备: {device}")

    # 读取原始数据
    raw_data = np.load(data_file, allow_pickle=True)
    raw_x = raw_data[:, :8].astype(np.float32)
    raw_y = raw_data[:, 8].astype(np.int64)

    # 规范化标签映射
    unique_labels = sorted(list(np.unique(raw_y)))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    raw_y = np.vectorize(lambda x: label_map[x])(raw_y)
    num_classes = len(unique_labels)
    print(f"🔄 标签连续映射: {label_map} | 类别数: {num_classes}")

    windows, window_labels = segment_stream(raw_x, raw_y, window_len=150, stride=20)
    dataset = SEMGSequenceDataset(windows, window_labels, seq_len=9, anchor_point=149, seq_stride=1)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = StatefulConvGLUBaseline(num_classes=num_classes, in_channels=8, d_model=64, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n🔥 开始训练 Stateful ConvGLU 模型...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total_samples = 0.0, 0, 0

        for x_seq, y_seq in train_loader:
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)

            optimizer.zero_grad()
            logits = model(x_seq, reset=True)
            loss = criterion(logits.reshape(-1, num_classes), y_seq.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_seq.numel()
            preds = logits.argmax(dim=-1)
            correct += (preds == y_seq).sum().item()
            total_samples += y_seq.numel()

        print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] Loss: {total_loss / total_samples:.4f} | Acc: {(correct / total_samples) * 100:.2f}%")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ 权重已保存至: '{model_save_path}'")
    return num_classes

if __name__ == "__main__":
    train_model()


# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
#
#
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# class SEMGDatasetFormatA(Dataset):
#     def __init__(self, train_data, label_data, anchor_point=149):
#         self.x_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 2, 1)
#         self.y_data = torch.tensor(label_data[:, anchor_point], dtype=torch.long)
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         return self.x_data[idx], self.y_data[idx]
#
#
# class SelectiveSSMBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, expand=2):
#         super().__init__()
#         self.d_inner = int(expand * d_model)
#         self.in_proj = nn.Linear(d_model, self.d_inner * 2)
#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             kernel_size=3,
#             padding=2,
#             groups=self.d_inner
#         )
#         self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
#         self.dt_proj = nn.Linear(1, self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, d_model)
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, x):
#         residual = x
#         batch, seq_len, _ = x.shape
#         x_norm = self.norm(x)
#         xz = self.in_proj(x_norm)
#         x_branch, z_branch = xz.chunk(2, dim=-1)
#
#         x_conv = x_branch.transpose(1, 2)
#         x_conv = self.conv1d(x_conv)[:, :, :seq_len]
#         x_conv = F.silu(x_conv.transpose(1, 2))
#
#         y = x_conv * F.silu(z_branch)
#         out = self.out_proj(y) + residual
#         return out
#
#
# class VanillaMambaBaseline(nn.Module):
#     def __init__(self, num_classes, in_channels=8, d_model=64, num_layers=2):
#         super().__init__()
#
#         self.input_proj = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=d_model,
#             kernel_size=1
#         )
#         self.input_norm = nn.LayerNorm(d_model)
#
#         self.mamba_layers = nn.ModuleList([
#             SelectiveSSMBlock(d_model=d_model) for _ in range(num_layers)
#         ])
#
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Linear(d_model // 2, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.input_proj(x).transpose(1, 2)
#         x = self.input_norm(x)
#
#         for layer in self.mamba_layers:
#             x = layer(x)
#
#         global_feat = torch.mean(x, dim=1)
#         logits = self.classifier(global_feat)
#         return logits
#
#
# def load_and_slice_emg_data(file_path="emg_data_labeled.npy", win_len=150, stride=20):
#     raw_data = np.load(file_path, allow_pickle=True)
#     total_len = raw_data.shape[0]
#
#     data_list = []
#     label_list = []
#
#     for start in range(0, total_len - win_len + 1, stride):
#         end = start + win_len
#         data_sample = raw_data[start:end, :8]
#         label_sample = raw_data[start:end, 8]
#
#         data_list.append(data_sample)
#         label_list.append(label_sample)
#
#     train_data = np.array(data_list, dtype=np.float32)
#     train_lable = np.array(label_list, dtype=np.int64)
#
#     # 自动重映射标签防止越界
#     unique_labels = sorted(list(np.unique(train_lable)))
#     label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
#     vectorized_map = np.vectorize(lambda x: label_map[x])
#     train_lable = vectorized_map(train_lable)
#
#     print(f"🔄 标签连续映射关系: {label_map}")
#     return train_data, train_lable, len(unique_labels)
#
#
# def train_mamba_model(data_file="emg_data_labeled.npy", model_save_path="mamba_baseline_150ms_best.pth", epochs=50, seed=42):
#     """供 main.py 调用的模型训练接口"""
#     set_seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 训练硬件设备: {device}")
#
#     print(f"📦 正在载入采集文件 '{data_file}' 并切分为 150ms 滑动窗口...")
#     train_data, train_lable, num_classes = load_and_slice_emg_data(data_file, win_len=150, stride=20)
#
#     print(f"📊 构建总样本数: {train_data.shape[0]}, 规范化类别数: {num_classes}")
#
#     dataset = SEMGDatasetFormatA(train_data, train_lable, anchor_point=149)
#     train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
#
#     model = VanillaMambaBaseline(num_classes=num_classes, in_channels=8, d_model=64, num_layers=2).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     criterion = nn.CrossEntropyLoss()
#
#     print("\n🔥 开始训练 Mamba 基线模型...")
#     model.train()
#     for epoch in range(epochs):
#         total_loss, correct, total_samples = 0.0, 0, 0
#
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#
#             optimizer.zero_grad()
#             logits = model(x_batch)
#             loss = criterion(logits, y_batch)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item() * x_batch.size(0)
#             preds = torch.argmax(logits, dim=1)
#             correct += (preds == y_batch).sum().item()
#             total_samples += y_batch.size(0)
#
#         epoch_loss = total_loss / total_samples
#         acc = (correct / total_samples) * 100
#         print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")
#
#     torch.save(model.state_dict(), model_save_path)
#     print(f"\n✅ 训练完成！模型权重已保存至 '{model_save_path}'")
#     return num_classes
#
#
# if __name__ == "__main__":
#     train_mamba_model()