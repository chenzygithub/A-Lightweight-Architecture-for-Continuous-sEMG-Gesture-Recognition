import asyncio
import struct
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, sosfilt_zi
from bleak import BleakClient, BleakScanner

from hand_control import ROHandController


# ==================== 模型与组件定义 ====================
def _inv_softplus(x):
    return np.log(np.exp(x) - 1.0)

def _inv_sigmoid(x):
    return np.log(x / (1.0 - x))

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
        new_mem = (1.0 - gate.unsqueeze(-1)) * mem + gate.unsqueeze(-1) * pooled
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

    def forward(self, x_seq, reset=False):
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


class RealtimeBandpassFilter:
    def __init__(self, lowcut=20.0, highcut=350.0, fs=1000, num_channels=8):
        self.sos = butter(4, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, np.newaxis], num_channels, axis=2)

    def filter_chunk(self, chunk):
        filtered_chunk = np.zeros_like(chunk)
        for ch in range(chunk.shape[1]):
            filtered_chunk[:, ch], self.zi[:, :, ch] = sosfilt(
                self.sos, chunk[:, ch], zi=self.zi[:, :, ch]
            )
        return filtered_chunk


class DualWindowMemoryFilter:
    def __init__(self, window_size=9, lock_threshold=9):
        self.window_size = window_size
        self.lock_threshold = lock_threshold
        self.buffer = []
        self.memory_state = 0

    def update(self, raw_pred: int) -> int:
        self.buffer.append(raw_pred)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < self.window_size:
            return raw_pred

        counts = np.bincount(self.buffer)
        proposal = int(np.argmax(counts))
        count = counts[proposal]

        if proposal != self.memory_state and count >= self.lock_threshold:
            self.memory_state = proposal

        return self.memory_state


# ==================== 蓝牙通信 ====================
SERVICE_GUID = "0000ffd0-0000-1000-8000-00805f9b34fb"
CMD_NOTIFY_CHAR_UUID = "f000ffe1-0451-4000-b000-000000000000"
DATA_NOTIFY_CHAR_UUID = "f000ffe2-0451-4000-b000-000000000000"

CMD_SET_EMG_RAWDATA_CONFIG = 0x3F
CMD_SET_DATA_NOTIF_SWITCH = 0x4F
DNF_EMG_RAW = 0x00000080
NTF_EMG_ADC_DATA = 0x08


class GForceProfile:
    def __init__(self):
        self.device = None
        self.onData = None

    async def connect_by_rssi(self, timeout=3.0, name_prefix="gForce", max_retries=3):
        for attempt in range(1, max_retries + 1):
            print(f"🔍 正在扫描 gForce 肌电手环... (第 {attempt}/{max_retries} 次尝试)")
            scanner = BleakScanner(service_uuids=[SERVICE_GUID])
            await scanner.start()
            await asyncio.sleep(timeout)
            await scanner.stop()

            target_dev = max(
                [
                    dev for dev, _ in scanner.discovered_devices_and_advertisement_data.values()
                    if dev.name and dev.name.startswith(name_prefix)
                ],
                key=lambda d: scanner.discovered_devices_and_advertisement_data[d.address][1].rssi,
                default=None
            )

            if target_dev:
                print(f"🔗 成功连接设备: {target_dev.name} ({target_dev.address})")
                self.device = BleakClient(target_dev.address)
                await self.device.connect()
                await self.device.start_notify(CMD_NOTIFY_CHAR_UUID, lambda c, d: None)
                return
            else:
                if attempt < max_retries:
                    print(f"⚠️ 未扫描到手环，等待 2 秒后重试...")
                    await asyncio.sleep(2.0)

        raise RuntimeError("❌ 未找到 gForce 手环设备！")

    async def start_emg_stream(self, on_data_cb, samp_rate=1000):
        self.onData = on_data_cb
        cfg_cmd = struct.pack("<BHHBB", CMD_SET_EMG_RAWDATA_CONFIG, samp_rate, 0xFF, 128, 8)
        await self.device.write_gatt_char(CMD_NOTIFY_CHAR_UUID, cfg_cmd)
        await asyncio.sleep(0.1)

        sw_cmd = struct.pack("<BI", CMD_SET_DATA_NOTIF_SWITCH, DNF_EMG_RAW)
        await self.device.write_gatt_char(CMD_NOTIFY_CHAR_UUID, sw_cmd)
        await asyncio.sleep(0.1)
        await self.device.start_notify(DATA_NOTIFY_CHAR_UUID, lambda c, d: self.onData(d) if self.onData else None)

    async def disconnect(self):
        if self.device and self.device.is_connected:
            try:
                await self.device.stop_notify(DATA_NOTIFY_CHAR_UUID)
            except Exception:
                pass
            await self.device.disconnect()
            print("🔌 gForce 蓝牙连接已关闭。")


# ==================== 实时推理引擎 ====================
class RealtimeInferenceEngine:
    def __init__(self, model_path="convglu_stateful_best.pth", num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 推理硬件平台: {self.device}")

        self.model = StatefulConvGLUBaseline(num_classes=num_classes, in_channels=8, d_model=64, num_layers=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.reset_memory()
        print(f"✅ 成功载入 ConvGLU 状态模型权重: '{model_path}'")

        self.filter = RealtimeBandpassFilter(lowcut=20.0, highcut=350.0, fs=1000, num_channels=8)
        self.dw_filter = DualWindowMemoryFilter(window_size=9, lock_threshold=9)
        self.hand_controller = ROHandController(port_keyword="CH340")

        self.win_len = 150
        self.lock = threading.Lock()
        self.emg_ring_buffer = deque([np.zeros(8, dtype=np.float32) for _ in range(self.win_len)], maxlen=self.win_len)

        self.profile = GForceProfile()
        self.running = True

    def process_incoming_data(self, raw_bytes: bytearray):
        if len(raw_bytes) >= 2 and raw_bytes[0] == NTF_EMG_ADC_DATA:
            payload = np.frombuffer(raw_bytes[1:], dtype=np.uint8)
            num_samples = len(payload) // 8
            if num_samples > 0:
                raw_samples = payload[:num_samples * 8].reshape(-1, 8).astype(np.float32) - 128.0
                filtered_samples = self.filter.filter_chunk(raw_samples)

                with self.lock:
                    self.emg_ring_buffer.extend(filtered_samples)
                    data_snapshot = np.array(self.emg_ring_buffer, dtype=np.float32)

                # 转换形状: (Batch=1, Seq_Len=1, Channels=8, Win_Len=150)
                x_tensor = torch.from_numpy(data_snapshot.T).float().unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(x_tensor, reset=False)  # reset=False 维持时序状态传递
                    raw_pred = int(logits[0, 0].argmax(dim=-1).item())

                filtered_pred = self.dw_filter.update(raw_pred)
                self.hand_controller.send_gesture(filtered_pred)

                print(f"\r⚡ [时序推断] 原始预测: {raw_pred}  |  🎯 记忆锁滤波后: {filtered_pred}   ", end="")

    def _async_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            try:
                await self.profile.connect_by_rssi()
                await self.profile.start_emg_stream(self.process_incoming_data, samp_rate=1000)
                print("\n🟢 连续肌电流式识别与仿生手控制已启动！\n")
                while self.running:
                    await asyncio.sleep(0.1)
            finally:
                await self.profile.disconnect()

        try:
            loop.run_until_complete(main())
        except Exception as e:
            print(f"\n❌ 推理服务中断: {e}")

    def start(self):
        threading.Thread(target=self._async_worker, daemon=True).start()

    def stop(self):
        self.running = False
        if hasattr(self, 'hand_controller'):
            self.hand_controller.close()


if __name__ == "__main__":
    engine = RealtimeInferenceEngine(model_path="convglu_stateful_best.pth", num_classes=3)
    engine.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
        print("\n\n👋 程序已安全终止。")


# import asyncio
# import struct
# import threading
# import time
# from collections import deque
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.signal import butter, sosfilt, sosfilt_zi
# from bleak import BleakClient, BleakScanner
#
# from hand_control import ROHandController
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
# class RealtimeBandpassFilter:
#     def __init__(self, lowcut=20.0, highcut=350.0, fs=1000, num_channels=8):
#         self.sos = butter(4, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
#         self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, np.newaxis], num_channels, axis=2)
#
#     def filter_chunk(self, chunk):
#         filtered_chunk = np.zeros_like(chunk)
#         for ch in range(chunk.shape[1]):
#             filtered_chunk[:, ch], self.zi[:, :, ch] = sosfilt(
#                 self.sos, chunk[:, ch], zi=self.zi[:, :, ch]
#             )
#         return filtered_chunk
#
#
# class RealtimeDualWindowFilter:
#     def __init__(self, window_size=9, lock_threshold=9):
#         self.window_size = window_size
#         self.lock_threshold = lock_threshold
#         self.buffer = []
#         self.memory_state = 0
#
#     def update(self, raw_pred: int) -> int:
#         self.buffer.append(raw_pred)
#         if len(self.buffer) > self.window_size:
#             self.buffer.pop(0)
#
#         if len(self.buffer) < self.window_size:
#             return raw_pred
#
#         counts = np.bincount(self.buffer)
#         proposal = np.argmax(counts)
#         count = counts[proposal]
#
#         if proposal != self.memory_state and count >= self.lock_threshold:
#             self.memory_state = proposal
#
#         return self.memory_state
#
#
# SERVICE_GUID = "0000ffd0-0000-1000-8000-00805f9b34fb"
# CMD_NOTIFY_CHAR_UUID = "f000ffe1-0451-4000-b000-000000000000"
# DATA_NOTIFY_CHAR_UUID = "f000ffe2-0451-4000-b000-000000000000"
#
# CMD_SET_EMG_RAWDATA_CONFIG = 0x3F
# CMD_SET_DATA_NOTIF_SWITCH = 0x4F
# DNF_EMG_RAW = 0x00000080
# NTF_EMG_ADC_DATA = 0x08
#
#
# class GForceProfile:
#     def __init__(self):
#         self.device = None
#         self.onData = None
#
#     async def connect_by_rssi(self, timeout=3.0, name_prefix="gForce", max_retries=3):
#         """带有自动重试机制的 BLE 扫描连接"""
#         for attempt in range(1, max_retries + 1):
#             print(f"🔍 正在扫描 gForce 肌电手环... (第 {attempt}/{max_retries} 次尝试)")
#             scanner = BleakScanner(service_uuids=[SERVICE_GUID])
#             await scanner.start()
#             await asyncio.sleep(timeout)
#             await scanner.stop()
#
#             target_dev = max(
#                 [
#                     dev for dev, _ in scanner.discovered_devices_and_advertisement_data.values()
#                     if dev.name and dev.name.startswith(name_prefix)
#                 ],
#                 key=lambda d: scanner.discovered_devices_and_advertisement_data[d.address][1].rssi,
#                 default=None
#             )
#
#             if target_dev:
#                 print(f"🔗 成功连接设备: {target_dev.name} ({target_dev.address})")
#                 self.device = BleakClient(target_dev.address)
#                 await self.device.connect()
#                 await self.device.start_notify(CMD_NOTIFY_CHAR_UUID, lambda c, d: None)
#                 return
#             else:
#                 if attempt < max_retries:
#                     print(f"⚠️ 未扫描到手环，可能蓝牙正在释放，等待 2 秒后重试...")
#                     await asyncio.sleep(2.0)
#
#         raise RuntimeError("❌ 未找到 gForce 手环设备！请检查手环是否开机或处于广播状态。")
#
#     async def start_emg_stream(self, on_data_cb, samp_rate=1000):
#         self.onData = on_data_cb
#
#         cfg_cmd = struct.pack("<BHHBB", CMD_SET_EMG_RAWDATA_CONFIG, samp_rate, 0xFF, 128, 8)
#         await self.device.write_gatt_char(CMD_NOTIFY_CHAR_UUID, cfg_cmd)
#         await asyncio.sleep(0.1)
#
#         sw_cmd = struct.pack("<BI", CMD_SET_DATA_NOTIF_SWITCH, DNF_EMG_RAW)
#         await self.device.write_gatt_char(CMD_NOTIFY_CHAR_UUID, sw_cmd)
#         await asyncio.sleep(0.1)
#
#         await self.device.start_notify(DATA_NOTIFY_CHAR_UUID, lambda c, d: self.onData(d) if self.onData else None)
#
#     async def disconnect(self):
#         if self.device and self.device.is_connected:
#             try:
#                 await self.device.stop_notify(DATA_NOTIFY_CHAR_UUID)
#             except Exception:
#                 pass
#             await self.device.disconnect()
#             print("🔌 [推理阶段] gForce 蓝牙连接已关闭。")
#
#
# class RealtimeInferenceEngine:
#     def __init__(self, model_path="mamba_baseline_150ms_best.pth", num_classes=7):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🚀 推理硬件平台: {self.device}")
#
#         self.model = VanillaMambaBaseline(num_classes=num_classes, in_channels=8, d_model=64, num_layers=2).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#         print(f"✅ 成功加载 Mamba 权重文件: '{model_path}'")
#
#         self.filter = RealtimeBandpassFilter(lowcut=20.0, highcut=350.0, fs=1000, num_channels=8)
#         self.dw_filter = RealtimeDualWindowFilter(window_size=10, lock_threshold=10)
#         self.hand_controller = ROHandController(port_keyword="CH340")
#
#         self.win_len = 150
#         self.lock = threading.Lock()
#         self.emg_ring_buffer = deque([np.zeros(8, dtype=np.float32) for _ in range(self.win_len)], maxlen=self.win_len)
#
#         self.profile = GForceProfile()
#         self.running = True
#
#     def process_incoming_data(self, raw_bytes: bytearray):
#         if len(raw_bytes) >= 2 and raw_bytes[0] == NTF_EMG_ADC_DATA:
#             payload = np.frombuffer(raw_bytes[1:], dtype=np.uint8)
#             num_samples = len(payload) // 8
#             if num_samples > 0:
#                 raw_samples = payload[:num_samples * 8].reshape(-1, 8).astype(np.float32) - 128.0
#                 filtered_samples = self.filter.filter_chunk(raw_samples)
#
#                 with self.lock:
#                     self.emg_ring_buffer.extend(filtered_samples)
#                     data_snapshot = np.array(self.emg_ring_buffer, dtype=np.float32)
#
#                 x_tensor = torch.tensor(data_snapshot.T, dtype=torch.float32, device=self.device).unsqueeze(0)
#
#                 with torch.no_grad():
#                     logits = self.model(x_tensor)
#                     raw_pred = torch.argmax(logits, dim=1).item()
#
#                 filtered_pred = self.dw_filter.update(raw_pred)
#                 self.hand_controller.send_gesture(filtered_pred)
#
#                 print(f"\r⚡ [实时推断] 原始预测标签: {raw_pred}  |  🎯 滤波后标签 (控制仿生手): {filtered_pred}   ", end="")
#
#     def _async_worker(self):
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#
#         async def main():
#             try:
#                 await self.profile.connect_by_rssi()
#                 await self.profile.start_emg_stream(self.process_incoming_data, samp_rate=1000)
#                 print("\n🟢 连续肌电流式实时识别与仿生手控制系统已全面启动！\n")
#                 while self.running:
#                     await asyncio.sleep(0.1)
#             finally:
#                 await self.profile.disconnect()
#
#         try:
#             loop.run_until_complete(main())
#         except Exception as e:
#             print(f"\n❌ 推理服务中断: {e}")
#
#     def start(self):
#         threading.Thread(target=self._async_worker, daemon=True).start()
#
#     def stop(self):
#         self.running = False
#         if hasattr(self, 'hand_controller'):
#             self.hand_controller.close()
#
#
# if __name__ == "__main__":
#     engine = RealtimeInferenceEngine(model_path="mamba_baseline_150ms_best.pth", num_classes=3)
#     engine.start()
#
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         engine.stop()
#         print("\n\n👋 在线推理程序已安全终止。")