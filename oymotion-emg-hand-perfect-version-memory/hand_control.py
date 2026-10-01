# -*- coding:utf-8 -*-
import sys
import time
from pymodbus.framer import FramerType
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
from serial.tools import list_ports


# 尝试导入官方寄存器定义
USING_OFFICIAL_LIB = False
try:
    from common.roh_registers_v2 import ROH_FINGER_POS_TARGET0

    USING_OFFICIAL_LIB = True
except ImportError:
    # 若无法导入，默认使用 common.roh_registers_v2 中的标准基地址
    ROH_FINGER_POS_TARGET0 = 0x0120

NODE_ID = 2

# ==========================================
# 标签到 ROHand 6自由度姿态的映射字典 (0 - 9)
# ==========================================
GESTURE_POSITIONS = {
    0: [0000, 0000, 0000, 0000, 0000, 0],  # REST / 休息
    1: [25550, 0, 0, 0, 0, 0],  # SPREAD / 张手
    2: [000, 25555, 0, 0, 0, 0],  # FIST / 握拳
    3: [0, 0, 25550, 0, 0, 0],  # VICTORY / 剪刀手
    4: [0, 0, 0, 25550, 0, 0],  # SIX / 666
    5: [0, 0, 0, 0, 25550, 0],  # POINT / 食指点按
    6: [0, 55550, 55550, 0, 0, 0],  # ROCK / 摇滚
    7: [45000, 45000, 45000, 45000, 45000, 0],  # PINCH / 捏合
    8: [0, 65535, 65535, 65535, 65535, 0],  # THUMBUP / 点赞
    9: [27525, 29491, 32768, 27525, 24903, 65535],  # GRASP / 抓握
}


class ROHandController:
    def __init__(self, port_keyword="CH340", baudrate=115200):
        self.client = None
        self.prev_label = None
        self.target_address = ROH_FINGER_POS_TARGET0

        print("\n" + "=" * 50)
        if USING_OFFICIAL_LIB:
            print(f"⚙️ [ROHand] 成功加载官方 roh_registers_v2, 基地址: {hex(self.target_address)}")
        else:
            print(f"⚠️ [ROHand] 未找到 common.roh_registers_v2，使用后备基地址: {hex(self.target_address)}")
        print("=" * 50)

        comport = self._find_comport(port_keyword) or self._find_comport("USB")
        if not comport:
            print("⚠️ [ROHand Warning] 未找到 CH340/USB 串口设备，仿生手控制将处于离线模式！")
            return

        self.client = ModbusSerialClient(comport, framer=FramerType.RTU, baudrate=baudrate)
        if self.client.connect():
            print(f"✅ [ROHand Success] 成功连接 ROHand 机械手 (串口: {comport})")
        else:
            print(f"❌ [ROHand Error] 无法连接到串口设备: {comport}")
            self.client = None

    def _find_comport(self, port_name):
        ports = list_ports.comports()
        for port in ports:
            if port_name in port.description:
                return port.device
        return None

    def send_gesture(self, label: int):
        if self.client is None or label == self.prev_label:
            return

        if label not in GESTURE_POSITIONS:
            return

        target_pos = GESTURE_POSITIONS[label]

        try:
            # 尝试 1: 批量写入 6 个寄存器
            resp = self.client.write_registers(self.target_address, target_pos, slave=NODE_ID)

            # 如果批量写入成功
            if not resp.isError():
                print(f" 🤖 -> [仿生手] 成功切换至标签 {label} 动作")
                self.prev_label = label
                return

            # 如果批量写入报 IllegalAddress，尝试 2: 逐个寄存器单独写入
            print(f"\n⚠️ 批量写入地址 {hex(self.target_address)} 失败，尝试逐通道写入...")
            success_count = 0
            for idx, val in enumerate(target_pos):
                single_addr = self.target_address + idx
                res = self.client.write_register(single_addr, val, slave=NODE_ID)
                if not res.isError():
                    success_count += 1

            if success_count == len(target_pos):
                print(f" 🤖 -> [仿生手] 逐通道写入成功，切换至标签 {label}")
                self.prev_label = label
            else:
                print(f"❌ [ROHand Error] 写入失败，请检查寄存器基地址 {hex(self.target_address)} 是否正确。")

        except ModbusException as e:
            print(f"❌ [ROHand Exception]: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("🔌 [ROHand] 串口连接已关闭。")

# # -*- coding:utf-8 -*-
# import time
# from pymodbus.framer import FramerType
# from pymodbus.client import ModbusSerialClient
# from pymodbus.exceptions import ModbusException
# from serial.tools import list_ports
#
# # 尝试导入寄存器地址定义，若路径不同请自行修改导入路径
# try:
#     from common.roh_registers_v2 import ROH_FINGER_POS_TARGET0
# except ImportError:
#     # 若无法导入，默认使用标准手部位置目标寄存器基地址
#     ROH_FINGER_POS_TARGET0 = 0x0120
#
# NODE_ID = 2
#
# # ==========================================
# # 标签到 ROHand 6自由度姿态的映射字典 (0 - 9)
# # 6个数值分别代表：[大拇指旋转, 大拇指弯曲, 食指, 中指, 无名指, 小指]
# # 范围: 0 (完全伸展/张开) ~ 65535 (完全弯曲/握紧)
# # ==========================================
# GESTURE_POSITIONS = {
#     0: [20000, 10000, 10000, 10000, 10000, 0],  # REST / 休息
#     1: [0, 0, 0, 0, 0, 0],  # SPREAD / 张手
#     2: [45000, 65535, 65535, 65535, 65535, 0],  # FIST / 握拳
#     3: [45000, 0, 0, 65535, 65535, 0],  # VICTORY / 胜利(剪刀手)
#     4: [0, 65535, 65535, 65535, 0, 0],  # SIX / 666
#     5: [45000, 0, 65535, 65535, 65535, 0],  # POINT / 食指点按
#     6: [0, 0, 65535, 65535, 0, 0],  # ROCK / 摇滚
#     7: [45000, 45000, 45000, 45000, 45000, 0],  # PINCH / 捏合
#     8: [0, 65535, 65535, 65535, 65535, 0],  # THUMBUP / 点赞
#     9: [27525, 29491, 32768, 27525, 24903, 65535],  # GRASP / 抓握
# }
#
#
# class ROHandController:
#     def __init__(self, port_keyword="CH340", baudrate=115200):
#         """初始化 Modbus 串口并连接 ROHand 机械手"""
#         self.client = None
#         self.prev_label = None  # 记录上一次发给机械手的标签，用于去重
#
#         comport = self._find_comport(port_keyword) or self._find_comport("USB")
#         if not comport:
#             print("⚠️ [ROHand Warning] 未找到 CH340/USB 串口设备，仿生手控制将处于离线模式！")
#             return
#
#         self.client = ModbusSerialClient(comport, framer=FramerType.RTU, baudrate=baudrate)
#         if self.client.connect():
#             print(f"✅ [ROHand Success] 成功连接 ROHand 机械手 (串口: {comport})")
#         else:
#             print(f"❌ [ROHand Error] 无法连接到串口设备: {comport}")
#             self.client = None
#
#     def _find_comport(self, port_name):
#         """自动扫描串口设备"""
#         ports = list_ports.comports()
#         for port in ports:
#             if port_name in port.description:
#                 return port.device
#         return None
#
#     def send_gesture(self, label: int):
#         """
#         实时接收预测标签并控制机械手
#         :param label: 经过后处理的整数手势标签 (0 ~ 9)
#         """
#         if self.client is None:
#             return  # 未成功连接设备时跳过
#
#         # 若预测标签与上一次相同，则直接跳过，避免无意义的串口写操作阻塞主循环
#         if label == self.prev_label:
#             return
#
#         if label not in GESTURE_POSITIONS:
#             print(f"\n⚠️ [ROHand Warning] 未知的标签ID: {label}，无法执行控制指令")
#             return
#
#         target_pos = GESTURE_POSITIONS[label]
#
#         try:
#             # 向机械手写入目标姿态坐标
#             resp = self.client.write_registers(ROH_FINGER_POS_TARGET0, target_pos, slave=NODE_ID)
#             if not resp.isError():
#                 print(f" 🤖 -> [仿生手响应] 成功切换至标签 {label} 的动作指令")
#                 self.prev_label = label
#             else:
#                 print(f"❌ [ROHand Error] 发送指令失败, 响应错误: {resp}")
#         except ModbusException as e:
#             print(f"❌ [ROHand ModbusException]: {e}")
#
#     def close(self):
#         """关闭串口通信"""
#         if self.client:
#             self.client.close()
#             print("🔌 [ROHand] 仿生手串口连接已断开。")