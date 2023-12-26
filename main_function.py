# ########################
# # 函数版 ###############

# import numpy as np
# from utils import ESA, MA, Origin, RoCoM, rotate_coordinate
# from tqdm import tqdm

# def main(u2_file, v2_file, w2_file, theta_file, Fs, Fw1, Fw2): # beta_file, 
#     # 加载数据
#     print("正在加载数据...")
#     u2 = np.loadtxt(u2_file) # 经坐标转换和去峰值处理后的数据
#     v2 = np.loadtxt(v2_file)
#     w2 = np.loadtxt(w2_file)
#     theta = np.loadtxt(theta_file)
#     # beta = np.loadtxt(beta_file)
#     n1, n2 = u2.shape # n1: 每个突发的样本数; n2: 突发的数量

#     # 确定波浪方向
#     print("正在确定波浪方向...")
#     bata_max1 = np.zeros(n2)
#     uf, vf, wf = np.zeros_like(u2), np.zeros_like(v2), np.zeros_like(w2)

#     for i in tqdm(range(n2), desc="波浪方向处理"):
#         bata_max1[i], uf[:, i], vf[:, i], wf[:, i] = rotate_coordinate(u2, v2, w2, theta, i, n1, Fs)

#     # 波-湍流分解
#     print("正在进行波-湍流分解...")
#     TKE = np.zeros((n2, 4))

#     for i in tqdm(range(n2), desc="波-湍流分解"):
#         # 原始方法
#         _, _, Ac_psd = Origin(uf[:, i], vf[:, i], wf[:, i], Fs) #freq, psd, Ac_psd
#         TKE[i, 0] = np.sum(Ac_psd[-1, :]) # 这一步是对的

#         # 移动平均法
#         _, Ac_psd_MA = MA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw2) #psd_MA, Ac_psd_MA
#         TKE[i, 1] = np.sum(Ac_psd_MA[-1, :])

#         # 能谱分析法
#         _, Ac_psd_ESA = ESA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2) #psd_ESA, Ac_psd_ESA
#         TKE[i, 2] = np.sum(Ac_psd_ESA[-1, :])

#         # 旋转坐标法
#         _, Ac_psd_RoCoM = RoCoM(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2) #psd_RoCoM, Ac_psd_RoCoM
#         TKE[i, 3] = np.sum(Ac_psd_RoCoM[-1, :])

#     # 保存数据
#     print("正在保存数据...")
#     np.savetxt('TKE.txt', TKE, fmt='%.7e')
#     print("数据处理完成，结果已保存到 'TKE.txt'")

# # 主程序入口
# if __name__ == "__main__":
#     # 定义文件名和参数
#     u2_file = 'u2.txt'
#     v2_file = 'v2.txt'
#     w2_file = 'w2.txt'
#     theta_file = 'theta.txt'
#     # beta_file = 'beta.txt'
#     Fs = 16
#     Fw1 = 0.05
#     Fw2 = 0.5

#     # 调用主函数
#     main(u2_file, v2_file, w2_file, theta_file, Fs, Fw1, Fw2)# beta_file, 

########################################
###     百分比进度条        ##############

# import numpy as np
# from utils import ESA, MA, Origin, RoCoM, rotate_coordinate
# 移除 tqdm，因为我们将使用 WebSocket 发送进度更新

# async def main_function(websocket, u2_file, v2_file, w2_file, theta_file, Fs, Fw1, Fw2):
#     # 加载数据
#     await websocket.send_text("正在加载数据...")
#     u2 = np.loadtxt(u2_file)
#     v2 = np.loadtxt(v2_file)
#     w2 = np.loadtxt(w2_file)
#     theta = np.loadtxt(theta_file)
#     n1, n2 = u2.shape

#     # 确定波浪方向
#     await websocket.send_text("正在确定波浪方向...")
#     bata_max1 = np.zeros(n2)
#     uf, vf, wf = np.zeros_like(u2), np.zeros_like(v2), np.zeros_like(w2)

#     for i in range(n2):
#         bata_max1[i], uf[:, i], vf[:, i], wf[:, i] = rotate_coordinate(u2, v2, w2, theta, i, n1, Fs)
#         progress = int((i + 1) / n2 * 100)  # 计算进度百分比
#         await websocket.send_text(f"波浪方向处理进度: {progress}%")

#     # 波-湍流分解
#     await websocket.send_text("正在进行波-湍流分解...")
#     TKE = np.zeros((n2, 4))

#     for i in range(n2):
#         _, _, Ac_psd = Origin(uf[:, i], vf[:, i], wf[:, i], Fs)
#         TKE[i, 0] = np.sum(Ac_psd[-1, :])
#         _, Ac_psd_MA = MA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw2)
#         TKE[i, 1] = np.sum(Ac_psd_MA[-1, :])
#         _, Ac_psd_ESA = ESA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2)
#         TKE[i, 2] = np.sum(Ac_psd_ESA[-1, :])
#         _, Ac_psd_RoCoM = RoCoM(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2)
#         TKE[i, 3] = np.sum(Ac_psd_RoCoM[-1, :])
#         progress = int((i + 1) / n2 * 100)
#         await websocket.send_text(f"波-湍流分解进度: {progress}%")

#     # 保存数据
#     await websocket.send_text("正在保存数据...")
#     np.savetxt('TKE.txt', TKE, fmt='%.7e')
#     await websocket.send_text("数据处理完成，结果已保存到 'TKE.txt'")


import numpy as np
from utils import ESA, MA, Origin, RoCoM, rotate_coordinate

async def main_function(websocket, u2_file, v2_file, w2_file, theta_file, Fs, Fw1, Fw2):
    # 加载数据
    await websocket.send_text("正在加载数据...")
    u2 = np.loadtxt(u2_file)
    v2 = np.loadtxt(v2_file)
    w2 = np.loadtxt(w2_file)
    theta = np.loadtxt(theta_file)
    n1, n2 = u2.shape

    # 确定波浪方向
    await websocket.send_text("正在确定波浪方向...")
    bata_max1 = np.zeros(n2)
    uf, vf, wf = np.zeros_like(u2), np.zeros_like(v2), np.zeros_like(w2)

    for i in range(n2):
        bata_max1[i], uf[:, i], vf[:, i], wf[:, i] = rotate_coordinate(u2, v2, w2, theta, i, n1, Fs)
        # 发送详细的进度信息
        await websocket.send_text(f"波浪方向处理: {i+1}/{n2} 完成")

    # 波-湍流分解
    await websocket.send_text("正在进行波-湍流分解...")
    TKE = np.zeros((n2, 4))

    for i in range(n2):
        # 原始方法
        _, _, Ac_psd = Origin(uf[:, i], vf[:, i], wf[:, i], Fs)
        TKE[i, 0] = np.sum(Ac_psd[-1, :])
        # 发送详细的进度信息
        await websocket.send_text(f"波-湍流分解 (原始方法): {i+1}/{n2} 完成")

        # 移动平均法
        _, Ac_psd_MA = MA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw2)
        TKE[i, 1] = np.sum(Ac_psd_MA[-1, :])
        # 发送详细的进度信息
        await websocket.send_text(f"波-湍流分解 (移动平均法): {i+1}/{n2} 完成")

        # 能谱分析法
        _, Ac_psd_ESA = ESA(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2)
        TKE[i, 2] = np.sum(Ac_psd_ESA[-1, :])
        # 发送详细的进度信息
        await websocket.send_text(f"波-湍流分解 (能谱分析法): {i+1}/{n2} 完成")

        # 旋转坐标法
        _, Ac_psd_RoCoM = RoCoM(uf[:, i], vf[:, i], wf[:, i], Fs, Fw1, Fw2)
        TKE[i, 3] = np.sum(Ac_psd_RoCoM[-1, :])
        # 发送详细的进度信息
        await websocket.send_text(f"波-湍流分解 (旋转坐标法): {i+1}/{n2} 完成")

    # 保存数据
    await websocket.send_text("正在保存数据...")
    np.savetxt('TKE.txt', TKE, fmt='%.7e')
    await websocket.send_text("数据处理完成，结果已保存到 'TKE.txt'")
