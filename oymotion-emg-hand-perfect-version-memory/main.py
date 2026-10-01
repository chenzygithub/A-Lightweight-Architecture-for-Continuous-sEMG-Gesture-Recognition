# -*- coding: utf-8 -*-
import os
import time
import torch
from pynput import keyboard

# 导入各模块组件
from data_collect import EMGDataCollector
from model_train import train_model                      # 1. 修改为新训练函数名
from model_predict import RealtimeInferenceEngine


def wait_for_choice(prompt_text, allowed_keys):
    """通用按键监听：打印提示语并等待用户按下 allowed_keys 中的任意一个按键"""
    print(prompt_text)
    selected_key = None

    def on_press(key):
        nonlocal selected_key
        try:
            ch = key.char.lower()
            if ch in [k.lower() for k in allowed_keys]:
                selected_key = ch
                return False  # 停止监听
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    return selected_key


def get_num_classes_from_model(model_path):
    """当直接按 'e' 跳过训练时，从已有 .pth 模型权重文件中自动解析 num_classes"""
    if not os.path.exists(model_path):
        return None
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        # 根据 StatefulConvGLUBaseline 分类头最后一层的偏置维度得出类别数
        if "classifier.2.bias" in state_dict:
            return state_dict["classifier.2.bias"].shape[0]
    except Exception as e:
        print(f"⚠️ 解析权重文件类别数失败: {e}")
    return None


def main():
    DATA_FILE = "emg_data_labeled.npy"
    MODEL_FILE = "convglu_stateful_best.pth"             # 2. 更新为新模型权重文件名

    print("==================================================")
    print("🚀 连续肌电手势识别与仿生手控制系统 (主控程序)")
    print("==================================================")

    # ----------------------------------------------------
    # 1. 自动启动数据采集阶段
    # ----------------------------------------------------
    print("\n【阶段 1】自动启动实时肌电采集程序...")
    print("👉 提示：按长按 0-9 采集打标；若无需采集新数据，可直接按 ['q'] 键跳过/完成采集。")

    collector = EMGDataCollector(save_filename=DATA_FILE)
    collector.start()  # 用户按 'q' 保存/结束，返回主流程

    num_classes = None

    # ----------------------------------------------------
    # 2. 交互选择：重新训练 ('w') 或 直接预测 ('e')
    # ----------------------------------------------------
    prompt_str = (
        "\n【选择下一步操作】\n"
        "👉 按 ['w'] 键：使用最新采集的数据开始模型训练\n"
        "👉 按 ['e'] 键：跳过训练，直接使用已有模型开始实时预测"
    )
    choice = wait_for_choice(prompt_str, allowed_keys=['w', 'e'])

    # --- 情况 A：用户选择训练模型 ('w') ---
    if choice == 'w':
        if not os.path.exists(DATA_FILE):
            print(f"\n❌ 未检测到数据集文件 '{DATA_FILE}'，无法训练！请重新运行程序并采集数据。")
            return

        print("\n【阶段 2】启动模型训练程序...")
        # 3. 调用新模型的训练接口
        num_classes = train_model(
            data_file=DATA_FILE,
            model_save_path=MODEL_FILE,
            epochs=20,
            batch_size=128,
            lr=1e-3,
            seed=42
        )

        # 训练完成后，提示按 'e' 启动预测
        prompt_after_train = "\n👉 模型训练完成！按 ['e'] 键启动实时预测与仿生手控制..."
        wait_for_choice(prompt_after_train, allowed_keys=['e'])

    # ----------------------------------------------------
    # 3. 启动实时预测阶段 ('e')
    # ----------------------------------------------------
    print("\n【阶段 3】启动实时预测与仿生手控制程序...")

    if not os.path.exists(MODEL_FILE):
        print(f"\n❌ 未找到模型权重文件 '{MODEL_FILE}'！请先按 'w' 训练模型。")
        return

    # 若直接选择 'e' 跳过训练，则自动从 .pth 文件中解析 num_classes
    if num_classes is None:
        num_classes = get_num_classes_from_model(MODEL_FILE)
        if num_classes is None:
            print(f"\n❌ 无法解析权重文件 '{MODEL_FILE}'，请重新训练。")
            return
        print(f"ℹ️ 检测到历史权重文件 '{MODEL_FILE}'，自动匹配类别数: num_classes = {num_classes}")

    # 实例化推理引擎并启动
    engine = RealtimeInferenceEngine(
        model_path=MODEL_FILE,
        num_classes=num_classes
    )
    engine.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
        print("\n\n👋 实时控制系统已安全终止。")


if __name__ == "__main__":
    main()


# # -*- coding: utf-8 -*-
# import os
# import time
# import torch
# from pynput import keyboard
#
# # 导入各模块组件
# from data_collect import EMGDataCollector
# from model_train import train_mamba_model
# from model_predict import RealtimeInferenceEngine
#
#
# def wait_for_choice(prompt_text, allowed_keys):
#     """通用按键监听：打印提示语并等待用户按下 allowed_keys 中的任意一个按键"""
#     print(prompt_text)
#     selected_key = None
#
#     def on_press(key):
#         nonlocal selected_key
#         try:
#             ch = key.char.lower()
#             if ch in [k.lower() for k in allowed_keys]:
#                 selected_key = ch
#                 return False  # 停止监听
#         except AttributeError:
#             pass
#
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()
#     return selected_key
#
#
# def get_num_classes_from_model(model_path):
#     """当直接按 'e' 跳过训练时，从已有 .pth 模型权重文件中自动解析 num_classes"""
#     if not os.path.exists(model_path):
#         return None
#     try:
#         state_dict = torch.load(model_path, map_location="cpu")
#         # 根据 VanillaMambaBaseline 分类头最后一层的偏置维度得出类别数
#         if "classifier.2.bias" in state_dict:
#             return state_dict["classifier.2.bias"].shape[0]
#     except Exception as e:
#         print(f"⚠️ 解析权重文件类别数失败: {e}")
#     return None
#
#
# def main():
#     DATA_FILE = "emg_data_labeled.npy"
#     MODEL_FILE = "mamba_baseline_150ms_best.pth"
#
#     print("==================================================")
#     print("🚀 连续肌电手势识别与仿生手控制系统 (主控程序)")
#     print("==================================================")
#
#     # ----------------------------------------------------
#     # 1. 自动启动数据采集阶段
#     # ----------------------------------------------------
#     print("\n【阶段 1】自动启动实时肌电采集程序...")
#     print("👉 提示：按长按 0-9 采集打标；若无需采集新数据，可直接按 ['q'] 键跳过/完成采集。")
#
#     collector = EMGDataCollector(save_filename=DATA_FILE)
#     collector.start()  # 用户按 'q' 保存/结束，返回主流程
#
#     num_classes = None
#
#     # ----------------------------------------------------
#     # 2. 交互选择：重新训练 ('w') 或 直接预测 ('e')
#     # ----------------------------------------------------
#     prompt_str = (
#         "\n【选择下一步操作】\n"
#         "👉 按 ['w'] 键：使用最新采集的数据开始模型训练\n"
#         "👉 按 ['e'] 键：跳过训练，直接使用已有模型开始实时预测"
#     )
#     choice = wait_for_choice(prompt_str, allowed_keys=['w', 'e'])
#
#     # --- 情况 A：用户选择训练模型 ('w') ---
#     if choice == 'w':
#         if not os.path.exists(DATA_FILE):
#             print(f"\n❌ 未检测到数据集文件 '{DATA_FILE}'，无法训练！请重新运行程序并采集数据。")
#             return
#
#         print("\n【阶段 2】启动模型训练程序...")
#         # 训练模型并返回本次数据的实际 num_classes
#         num_classes = train_mamba_model(
#             data_file=DATA_FILE,
#             model_save_path=MODEL_FILE,
#             epochs=50,
#             seed=42
#         )
#
#         # 训练完成后，提示按 'e' 启动预测
#         prompt_after_train = "\n👉 模型训练完成！按 ['e'] 键启动实时预测与仿生手控制..."
#         wait_for_choice(prompt_after_train, allowed_keys=['e'])
#
#     # ----------------------------------------------------
#     # 3. 启动实时预测阶段 ('e')
#     # ----------------------------------------------------
#     print("\n【阶段 3】启动实时预测与仿生手控制程序...")
#
#     if not os.path.exists(MODEL_FILE):
#         print(f"\n❌ 未找到模型权重文件 '{MODEL_FILE}'！请先按 'w' 训练模型。")
#         return
#
#     # 若直接选择 'e' 跳过训练，则自动从 .pth 文件中解析 num_classes
#     if num_classes is None:
#         num_classes = get_num_classes_from_model(MODEL_FILE)
#         if num_classes is None:
#             print(f"\n❌ 无法解析权重文件 '{MODEL_FILE}'，请重新训练。")
#             return
#         print(f"ℹ️ 检测到历史权重文件 '{MODEL_FILE}'，自动匹配类别数: num_classes = {num_classes}")
#
#     # 实例化推理引擎并启动
#     engine = RealtimeInferenceEngine(
#         model_path=MODEL_FILE,
#         num_classes=num_classes
#     )
#     engine.start()
#
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         engine.stop()
#         print("\n\n👋 实时控制系统已安全终止。")
#
#
# if __name__ == "__main__":
#     main()



#
# 第一版
# -*- coding: utf-8 -*-
# import os
# import time
# from pynput import keyboard
#
# # 1. 导入各模块组件
# from data_collect import EMGDataCollector
# from model_train import train_mamba_model
# from model_predict import RealtimeInferenceEngine
#
#
# def wait_for_key(target_char):
#     """阻塞等待指定键盘字符按下"""
#     print(f"\n👉 请按键盘 ['{target_char}'] 键继续下个流程...")
#     pressed = False
#
#     def on_press(key):
#         nonlocal pressed
#         try:
#             if key.char.lower() == target_char.lower():
#                 pressed = True
#                 return False  # 停止 Listener
#         except AttributeError:
#             pass
#
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()
#
#
# def main():
#     DATA_FILE = "emg_data_labeled.npy"
#     MODEL_FILE = "mamba_baseline_150ms_best.pth"
#
#     print("==================================================")
#     print("🚀 连续肌电手势识别与仿生手控制系统 (主控程序)")
#     print("==================================================")
#
#     # ----------------------------------------------------
#     # 1. 自动启动数据采集阶段
#     # ----------------------------------------------------
#     print("\n【阶段 1】自动启动实时肌电采集程序...")
#     collector = EMGDataCollector(save_filename=DATA_FILE)
#
#     # collector.start() 将会开始采集，并在用户按下 'q' 保存数据后返回
#     collector.start()
#
#     if not os.path.exists(DATA_FILE):
#         print(f"\n❌ 未检测到保存的 '{DATA_FILE}' 文件，程序提前退出。")
#         return
#
#     # ----------------------------------------------------
#     # 2. 等待 'w' 键启动训练阶段
#     # ----------------------------------------------------
#     wait_for_key('w')
#     print("\n【阶段 2】启动模型训练程序...")
#
#     # 执行训练，并获取动态生成的 num_classes
#     num_classes = train_mamba_model(
#         data_file=DATA_FILE,
#         model_save_path=MODEL_FILE,
#         epochs=50,
#         seed=42
#     )
#
#     # ----------------------------------------------------
#     # 3. 等待 'e' 键启动预测控制阶段
#     # ----------------------------------------------------
#     wait_for_key('e')
#     print("\n【阶段 3】启动实时预测与仿生手控制程序...")
#
#     # 传入与训练集精准匹配的 num_classes，彻底解决 size mismatch 报错
#     engine = RealtimeInferenceEngine(
#         model_path=MODEL_FILE,
#         num_classes=num_classes
#     )
#     engine.start()
#
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         engine.stop()
#         print("\n\n👋 主系统已终止运行。")
#
#
# if __name__ == "__main__":
#     main()