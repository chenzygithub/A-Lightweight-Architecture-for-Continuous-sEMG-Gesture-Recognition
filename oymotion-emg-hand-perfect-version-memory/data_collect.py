# -*- coding: utf-8 -*-
import asyncio
import struct
import threading
import time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from bleak import BleakClient, BleakScanner
from pynput import keyboard

# ==========================================
# 1. 常量与 gForce BLE 通信类
# ==========================================
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

    async def connect_by_rssi(self, timeout=3.0, name_prefix="gForce"):
        print("🔍 正在扫描 gForce 手环...")
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

        if not target_dev:
            raise RuntimeError("❌ 未找到 gForce 手环设备！")

        print(f"🔗 连接设备: {target_dev.name} ({target_dev.address})")
        self.device = BleakClient(target_dev.address)
        await self.device.connect()
        print("✅ 蓝牙连接成功！")

        await self.device.start_notify(CMD_NOTIFY_CHAR_UUID, lambda c, d: None)

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
        """显式断开蓝牙连接，释放给下个阶段使用"""
        if self.device and self.device.is_connected:
            try:
                await self.device.stop_notify(DATA_NOTIFY_CHAR_UUID)
            except Exception:
                pass
            await self.device.disconnect()
            print("🔌 [采集阶段] gForce 蓝牙连接已成功断开释放！")


# ==========================================
# 2. 实时带通滤波器 (20Hz - 350Hz)
# ==========================================
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


# ==========================================
# 3. 实时采集与打标系统
# ==========================================
class EMGDataCollector:
    def __init__(self, save_filename="emg_data_labeled.npy"):
        self.filter = RealtimeBandpassFilter(lowcut=20.0, highcut=350.0, fs=1000, num_channels=8)
        self.profile = GForceProfile()
        self.save_filename = save_filename

        self.is_recording = False
        self.current_label = -1
        self.pressed_keys = set()
        self.running = True

        self.segment_counts = {}
        self.saved_data_list = []
        self.lock = threading.Lock()
        self.listener = None

    def parse_and_process_bytes(self, raw_bytes: bytearray):
        if len(raw_bytes) >= 2 and raw_bytes[0] == NTF_EMG_ADC_DATA:
            payload = np.frombuffer(raw_bytes[1:], dtype=np.uint8)
            num_samples = len(payload) // 8
            if num_samples > 0:
                raw_samples = payload[:num_samples * 8].reshape(-1, 8).astype(np.float32) - 128.0
                filtered_samples = self.filter.filter_chunk(raw_samples)

                if self.is_recording:
                    labels = np.full((num_samples, 1), self.current_label, dtype=np.float32)
                    labeled_chunk = np.hstack((filtered_samples, labels))

                    with self.lock:
                        self.saved_data_list.append(labeled_chunk)

    def on_key_press(self, key):
        try:
            char = key.char
            if char in [str(i) for i in range(10)]:
                if char not in self.pressed_keys:
                    self.pressed_keys.add(char)
                    self.current_label = int(char)
                    self.is_recording = True
                    print(f"\n🔴 [开始录制] 标签 {self.current_label} ...")
            elif char == 'q':
                self.save_and_exit()
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            char = key.char
            if char in [str(i) for i in range(10)]:
                if char in self.pressed_keys:
                    self.pressed_keys.remove(char)

                if len(self.pressed_keys) == 0 and self.is_recording:
                    self.is_recording = False
                    lbl = self.current_label
                    self.segment_counts[lbl] = self.segment_counts.get(lbl, 0) + 1

                    summary_list = [f"手势 [{k}]: {v}段" for k, v in sorted(self.segment_counts.items())]
                    summary_str = " | ".join(summary_list)

                    print(f"⏹️ [停止录制] 手势 [{lbl}] 录制 1 段 (当前累计 {self.segment_counts[lbl]} 段)")
                    print(f"📊 【进度统计】已采集 {len(self.segment_counts)} 种手势 -> {summary_str}\n")
        except AttributeError:
            pass

    def _async_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            try:
                await self.profile.connect_by_rssi()
                await self.profile.start_emg_stream(self.parse_and_process_bytes, samp_rate=1000)
                print("\n🚀 [采集模式] 长按数字键 0-9 采集打标，松开停止。按 'q' 结束采集并保存。")
                while self.running:
                    await asyncio.sleep(0.1)
            finally:
                # 🚩 退出前显式断开蓝牙，避免阻塞阶段 3 的重新连接
                await self.profile.disconnect()

        try:
            loop.run_until_complete(main())
        except Exception as e:
            print(f"\n❌ 采集中断: {e}")

    def start(self):
        threading.Thread(target=self._async_worker, daemon=True).start()

        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        self.listener.join()

    def save_and_exit(self):
        print("\n💾 正在处理并保存采集数据...")
        self.running = False

        with self.lock:
            if len(self.saved_data_list) == 0:
                print("⚠️ 未采集到任何数据。")
            else:
                final_data = np.vstack(self.saved_data_list)
                np.save(self.save_filename, final_data)

                print("=" * 60)
                print(f"✅ 数据成功保存至: '{self.save_filename}'")
                print(f"📊 最终数据形状 (N, 9): {final_data.shape}")
                print("=" * 60)

        if self.listener:
            self.listener.stop()


if __name__ == "__main__":
    collector = EMGDataCollector(save_filename="emg_data_labeled.npy")
    collector.start()