# -*- coding:utf-8 -*-
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math
from scipy.fftpack import fft, ifft


#
# # 码元数,横坐标为1000个点
# size = 24
# sampling_t = 0.01
# t = np.arange(0, size, sampling_t)
#
# # 随机生成信号序列
# # 随机生成二进制数，一共10个bit，由size决定
#
# logon_content = "000000010000000100101001000111110110111010001101101000100011011100001000010001110100110111000110010110111011100101101100100101100110111000101110001110110010111100011011001001100011100100000101110111000010110100001011110110110011100100010111001001111001000001110000111110001110110000110101111001111010001101011111001001010001001001000111111001000110001110101010101010110100001111000011101011100101111110110000100111101101001001011110001100101110101111101101011110101000000000011100110011111111110010000000010100010110001001101110111111100110111011001010110110101101010111110100001010101000010010111000011110100010100111011011000010000011100101110110101101001011101000110100111001100111011000001001100000100001111110101010100100111101011101100111101111101010001111101101100100100111001010111001001110110000001010100010100010111011011001001110111101111111110000101110 "
#
# # 双极性码，生成两个元素的列表，每个符号为24个元素的列表
# a = 2 * np.random.randint(0, 2, (2, 24)) - 1
# b = 2 * np.random.randint(0, 2, (2, 24)) - 1
# # a[0]对应I路信号，a[1]对应Q路信号
# # m为定义的长度为1000个0列表，类型为float
# I = np.zeros(len(t), dtype=np.float32)
# Q = np.zeros(len(t), dtype=np.float32)
# # 循环给m赋值
# for i in range(len(t)):
#     I[i] = a[0][math.floor(t[i])]
#     Q[i] = a[1][math.floor(t[i])]
# fig = plt.figure()
# ax1 = fig.add_subplot(4, 3, 1)
# # 画出I路信号
# zhfont1 = matplotlib.font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
# ax1.set_title('I路信号', fontproperties=zhfont1, fontsize=15)
# # 定义图的横纵坐标范围
# plt.axis([0, size, -2, 2])
# plt.plot(t, I, 'b')
# # 画出Q路信号
# ax2 = fig.add_subplot(4, 3, 4)
# # 解决set_title中文乱码
# zhfont1 = matplotlib.font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
# ax2.set_title('Q路信号', fontproperties=zhfont1, fontsize=15)
# # 定义图的横纵坐标范围
# plt.axis([0, size, -2, 2])
# plt.plot(t, Q, 'b')
#
# # # 定义两个载频
# # 定义载波频率为1024hz
# fc1 = 2048
# # fc2 = 10000
# # # 定义采样频率，采样频率要大于信号带宽的两倍
# fs = 50000
# # # 100 为0.01相当于1个符号采样100个点
# ts = np.arange(0, (100 * size) / fs, 1 / fs)
# # # np.dot返回两个数组的点积，ts为一个数组，返回前一个数与后一个数组的积，为一个数组
# coherent_carrier = np.cos(np.dot(2 * pi * fc1, ts))
# coherent_carrier1 = np.sin(np.dot(2 * pi * fc1, ts))
# # # 两个数组相乘，得到的也是一个数组，两个数组对应下标的乘积，同时两个数组的长度要一致
# qpsk = I * coherent_carrier + Q * coherent_carrier1
# #
# # # BPSK调制信号波形
# ax3 = fig.add_subplot(4, 3, 7)
# ax3.set_title('QPSK调制信号', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, qpsk, 'r')
#
#
# #
# #
# 定义加性高斯白噪声 y为原始信号，snr为信噪比
def awgn(y, snr):
    # 根据db信噪比计算信号与造成的功率比
    snr = 10 ** (snr / 10.0)
    # 原始信号的的归一化功率 值的平方除以抽样点的数量
    xpower = np.sum(y ** 2) / len(y)
    # 噪声的平均功率
    npower = xpower / snr
    # randn为标准正态分布 ，在与原始噪声相加，加性噪声
    return np.random.randn(len(y)) * np.sqrt(npower) + y
#
#
# #
# #
# # # 加AWGN噪声
# noise_qpsk = awgn(qpsk, -5)
# #
# # BPSK调制信号叠加噪声波形
# ax3 = fig.add_subplot(4, 3, 10)
# ax3.set_title('调制信号叠加噪声波形', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, noise_qpsk, 'r')
# #
# # 带通butter滤波器设计，通带为[1500，2500]
# b, a = signal.butter(4, [1500 * 2 / fs, 2500 * 2 / fs], 'bandpass')
# # # 低通滤波器设计，通带截止频率为2000Hz，截至频率计算方法，截止数值 * 2 / 抽样点数
# bl, al = signal.butter(4, 1500 * 2 / fs, 'lowpass')
# #
# # # 通过带通滤波器滤除带外噪声
# bandpass_out1 = signal.filtfilt(b, a, noise_qpsk)
# #
# # # 相干解调,乘以同频同相的相干载波
# coherent_demodI = bandpass_out1 * coherent_carrier
# coherent_demodQ = bandpass_out1 * coherent_carrier1
#
# # 通过低通滤波器
# lowpass_outI = signal.filtfilt(bl, al, coherent_demodI)
# lowpass_outQ = signal.filtfilt(bl, al, coherent_demodQ)
#
# bx1 = fig.add_subplot(4, 3, 8)
# bx1.set_title('I路信号相干解调后的信号', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, lowpass_outI, 'r')
#
# bx2 = fig.add_subplot(4, 3, 11)
# bx2.set_title('Q路信号相干解调后的信号', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, lowpass_outQ, 'r')
#
# # 抽样判决
# # 抽样的数据为1000个，类型为float
# detection_I = np.zeros(len(t), dtype=np.float32)
# detection_Q = np.zeros(len(t), dtype=np.float32)
# # 译码的数据为10个，类型为float
# flagI = np.zeros(size, dtype=np.float32)
# flagQ = np.zeros(size, dtype=np.float32)

# 循环从1000个数据中取值，如果100此求和，如果最终的结果大于50，则为1
# for i in range(size):
#     tempI = 0
#     tempQ = 0
#     for j in range(100):
#         tempI = tempI + lowpass_outI[i * 100 + j]
#         tempQ = tempQ + lowpass_outQ[i * 100 + j]
#
#     flagI[i] = np.sign(tempI)
#     flagQ[i] = np.sign(tempQ)
# #
# # 将抽样数据规整，100个值要求一致
# for i in range(size):
#     if flagI[i] == 1:
#         for j in range(100):
#             detection_I[i * 100 + j] = 1
#     else:
#         for j in range(100):
#             detection_I[i * 100 + j] = -1
#     if flagQ[i] == 1:
#         for j in range(100):
#             detection_Q[i * 100 + j] = 1
#     else:
#         for j in range(100):
#             detection_Q[i * 100 + j] = -1
#
#
# # 定义一个fft函数
# def signal_fft(signal):
#     fy = fft(signal)
#     fy_real = fy.real
#     fy_imag = fy.imag
#     fy_abs = abs(fy)
#     fy1 = fy_abs / len(signal)
#     fy2 = fy1[range(int(len(signal) / 2))]
#     xf = np.arange(len(signal))
#     xf1 = xf
#     xf2 = xf[range(int(len(signal) / 2))]
#     return fy_abs
#     # plt.subplot(311)
#     # plt.plot(xf, fy_abs, 'r')
#     # plt.title("FFT of Mixed wave(two sides)")
#     # plt.subplot(312)
#     # plt.plot(xf1, fy1, 'g')
#     # plt.title("FFT of Mixed wave(normalization)")
#     # plt.subplot(313)
#     # plt.plot(xf2, fy2, 'b')
#     # plt.title("FFT of Mixed wave")
#     # plt.show()
#

# originalI_fft = signal_fft(I)
# originalQ_fft = signal_fft(Q)
# module_fft = signal_fft(qpsk)
# noise_fft = signal_fft(noise_qpsk)
# x = np.arange(len(I))
# fp1 = fig.add_subplot(4, 3, 3)
# fp1.set_title('I路频谱', fontproperties=zhfont1, fontsize=15)
# # plt.axis([0, len(I), 0, 2])
# plt.plot(x[0:500], originalI_fft[0:500], 'r')
# fp1 = fig.add_subplot(4, 3, 6)
# fp1.set_title('Q路频谱', fontproperties=zhfont1, fontsize=15)
# # plt.axis([0, len(I), 0, 2])
# plt.plot(x[0:500], originalQ_fft[0:500], 'r')
# fp1 = fig.add_subplot(4, 3, 9)
# fp1.set_title('QPSK频谱', fontproperties=zhfont1, fontsize=15)
# # plt.axis([0, len(I), 0, 2])
# plt.plot(x[0:500], module_fft[0:500], 'r')
# fp1 = fig.add_subplot(4, 3, 12)
# fp1.set_title('加噪频谱', fontproperties=zhfont1, fontsize=15)
# # plt.axis([0, len(I), 0, 2])
# plt.plot(x[0:500], noise_fft[0:500], 'r')
#
# bx2 = fig.add_subplot(4, 3, 2)
# bx2.set_title('I路抽样判决后的信号', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, detection_I, 'r')
# bx2 = fig.add_subplot(4, 3, 5)
# bx2.set_title('Q路抽样判决后的信号', fontproperties=zhfont1, fontsize=15)
# plt.axis([0, size, -2, 2])
# plt.plot(t, detection_Q, 'r')
# plt.show()

# 1.进行帧数据统计，每载波中时隙数据长度以最长数据为准进行扩展，其中_代表不发送数据为空信号，其余为双极性编码

def flame_format_data(flame_data):
    # 1找到时隙中最长的比特数
    max = 0
    data = []
    r_data = []
    for i in range(len(flame_data)):
        sub_max_length = 0
        split_data = flame_data[i].split(":")
        data.extend(split_data)
        for j in range(len(split_data)):
            if len(split_data[j]) > sub_max_length:
                sub_max_length = len(split_data[j])
        if sub_max_length >= max:
            max = sub_max_length
    # 2对字符串进行扩充
    if max > 1:
        for j in range(len(data)):
            print(len(data))
            if data[j] == "_":
                data[j] = "@" * max
            else:
                # 将1变成2,，在调制时直接减1操作形成双极性码
                data[j] = data[j].ljust(max, "@").replace("1", "2")
    else:
        # 如果所有元素为空，则创建一个10* 20 的单符号的空列表2bit 为后续串并转换
        for j in range(len(data)):
            data[j] = "@@"
    # 3对data中的数据进行20个元素分割
    r_data = [data[i:i + 20] for i in range(0, len(data), 20)]
    return r_data


def format_data(filename):
    f = open(filename, "r")
    line = f.readline().strip(":\n")
    datas = []
    frame_data = []
    while line:
        flame_index = 0
        line = line.strip(":\n")
        temp = line.split(":")
        if temp[0] == "frame":
            if temp[1] != "0":
                data = flame_format_data(frame_data)
                print("data")
                # for i in range(len(data)):
                #     for j in range(len(data[i])):
                #         print(data[i][j])
                datas.append(data)
            frame_data = []
        else:
            frame_data.append(line)

        print(line)
        print("frame_data")

        # print(frame_data)
        line = f.readline()
    f.close()
    return datas


# 返回的数据为三位数组，其中第一维为帧数，第二维为子载波，第三位为时隙，时隙中的元素为规整后的比特数据流
# data = format_data()
# print(data[2][0][17])

# 2.根据数据进行I，Q两路串并转换，操作对象为帧数据，返回为帧数据，格式为子载波数，I路数据，Q路数据
def IQ_split(data):
    for i in range(len(data)):
        # 10个子载波
        I = []
        Q = []
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if k % 2 == 0:
                    I.append(data[i][j][k])
                else:
                    Q.append(data[i][j][k])


# 3.对每子帧组数据选择抽样点和载波抽样控制，将数据和载波进行乘 和操作，生成最终的MF-TDMA操作
# 每个比特进行100个采样，符号个数为串并转换后的长度，载波频率为fc1 = 2048 倍数，采样频率为50000，对每个时隙的比特流进行调制
def modulation_QPSK(data, fc):
    # 符号采样书
    sample_b = 0.01
    size = int(len(data) / 2)
    t = np.arange(0, size, sample_b)
    # 对符号进行串并转换,并且进行采样扩充
    temp_I = []
    temp_Q = []
    I = np.zeros(len(t), dtype=np.float32)
    Q = np.zeros(len(t), dtype=np.float32)
    # 判断该时隙是否为全0，否则进行双极性码转换
    for i in range(len(data)):
        if i % 2 == 0:
            if data[i] == "@":
                temp_I.append(0)
            elif data[i] == "0":
                temp_I.append(-1)
            else:
                temp_I.append(1)
        else:
            if data[i] == "@":
                temp_Q.append(0)
            elif data[i] == "0":
                temp_Q.append(-1)
            else:
                temp_Q.append(1)
    # print(size)
    # 对I路和Q路进行扩充
    for i in range(len(t)):
        I[i] = temp_I[math.floor(t[i])]
        Q[i] = temp_Q[math.floor(t[i])]

    # 定义载波和采样频率
    fc = fc
    fs = 50000
    ts = np.arange(0, (size / sample_b) / fs, 1 / fs)
    coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
    coherent_carrier1 = np.sin(np.dot(2 * pi * fc, ts))
    # 两个数组相乘，得到的也是一个数组，两个数组对应下标的乘积，同时两个数组的长度要一致
    qpsk = I * coherent_carrier + Q * coherent_carrier1
    return qpsk


# print(data[0][2][3])
# qpsk = modulation_QPSK(data[0][2][3], 2048)
# print(len(data[0][2][3]))
# print(len(qpsk))
# print(qpsk.tolist())

# 循环对data数据进行时隙数据调制，最终生成二维的加载波的调制信号
# print(qpsk)
# 最终生成三维数据，第一位为帧数据，第二维为载波数据，第三位为时隙数据
def flame_modulation(data):
    qpsk_signal = []
    print("test1")
    print(len(data[5][3]))
    for i in range(len(data)):
        # 获取帧数据，是二维信号
        flame_data = data[i]
        fc = 2048
        qpsk_fc = []
        for j in range(len(flame_data)):
            # 定义载波的载波频率
            fc_j = fc * (j + 1)
            # 获取当前载波的帧数据
            fc_data = flame_data[j]
            fc_data_qpsk = []
            for k in range(len(fc_data)):
                # 获取时隙数据
                slot_data = fc_data[k]
                qpsk = modulation_QPSK(slot_data, fc_j)
                fc_data_qpsk.append(qpsk)
                # print(qpsk)
            qpsk_fc.append(fc_data_qpsk)
        qpsk_signal.append(qpsk_fc)
    return qpsk_signal


# data = format_data()
# qpsk_Data = flame_modulation(data)
# print(len(qpsk_Data))

# 重放攻击为截取一段调制数据，在剩余帧中进行重放，重放攻击要保证两方面，1，同一子载波的帧之间的数据重放，2，截取的数据要满足所重放时隙的调制长度，进行扩充和删减。
# 定义一个写文件函数，原始正确数据，以及重放攻击数据的函数
def write_qpsk_data(data, filename):
    f = open("./attack/" + filename, "w")
    content = ""
    for i in range(len(data)):
        f.write("frame sequence is " + str(i) + "\n")
        flame_data = data[i][0]
        for k in range(len(flame_data)):
            # print(str(k) + ":" + str(sub_data[k].tolist()))
            # 只对0载波进行重放，因此只记录0载波的发送数据
            f.write(str(k) + ":" + str(flame_data[k].tolist()) + "\n")
    f.close()


# 定义一个记录完整帧数据的写入函数
def write_all_data(data, filename):
    f = open("./attack/" + filename, "w")
    content = ""
    for i in range(len(data)):
        f.write("frame sequence is " + str(i) + "\n")
        flame_data = data[i]
        for j in range(len(flame_data)):
            sub_data = flame_data[j]
            f.write("the sub sequence is " + str(j) + "\n")
            for k in range(len(sub_data)):
                # print(str(k) + ":" + str(sub_data[k].tolist()))
                # 只对0载波进行重放，因此只记录0载波的发送数据
                f.write(str(k) + ":" + str(sub_data[k].tolist()) + "\n")
    f.close()


# 重放攻击可以截取一段数据
# 单次重放，重放主要是对0载波数据进行重放，截取0载波中某一时隙的数据，在其余帧中进行重放
# 多次重放，截取0载波中某一时隙的数据，在其余帧中随机多次重放
# 定义一个重放函数，入参包含原始数据和重放次数
def replay_attack(data, num):
    # 先获取包含数据和不包含数据的列表，随机挑选出包含数据的数组，在不包含数据的时隙中进行重放
    empty_array = []
    data_array = []
    for i in range(len(data)):
        sub_data = data[i][0]
        for j in range(len(sub_data)):
            if np.all(sub_data[j] == 0):
                empty_array.append((i, j))
            else:
                data_array.append((i, j))
    # print(empty_array)
    # print(data_array)
    # 进行重放，先从包含数据的列表中挑选一组重放数据
    index = np.random.randint(0, len(data_array))
    replay_data = data[data_array[index][0]][0][data_array[index][1]]
    # 取得重放数据长度
    replay_data_length = len(replay_data)
    # print(replay_data.tolist())
    # 与num比较修正重放次数，重放数据必须在重放数据帧之后的帧中进行重放
    if data_array[index][0] == len(data) - 1:
        print("the replay data is last frame, can not replay")
    elif data_array[index][0] + num > len(data) - 1:
        num = len(data) - 1 - data_array[index][0]
        print("replay num must be less total frame num , recite the num is " + str(num))
    else:
        print("replay num is " + str(num))
    # 对数据进行重放，在空时隙中进行重放，首先找到重放帧序号，重放帧在数据所在帧之后
    replay_frame = data_array[index][0]
    start_index = 0
    for i in range(len(empty_array)):
        if empty_array[i][0] > replay_frame:
            start_index = i
            break
    replay_empty_index = np.random.randint(start_index, len(empty_array), num)
    # 对空时隙长度进行统计

    for i in range(len(replay_empty_index)):
        print("replay attack " + str(i + 1) + ":" + str(data_array[index]) + "->" + str(
            empty_array[replay_empty_index[i]]))
        empty_frame = empty_array[replay_empty_index[i]][0]
        empty_slot = empty_array[replay_empty_index[i]][1]
        empty_data = data[empty_frame][0][empty_slot]
        if replay_data_length <= len(empty_data):
            # 如果重放数据长度小于等于空时隙长度，则重放数据扩展为空时隙长度
            data[empty_frame][0][empty_slot][0:replay_data_length] = replay_data
        else:
            # 如果重放数据长度大于空时隙长度，按照空时隙长度对重放数据进行截取
            data[empty_frame][0][empty_slot] = replay_data[0:len(empty_data)]
            # print(data[empty_frame][0][empty_slot].tolist())
    return data


# 增加时隙碰撞函数，将调制后的信号不同时隙下的信号进行线性叠加，逻辑与占用类似，从数据中选取时隙数据在另一有数据的时隙进行线性叠加，
# 仍满足碰撞时隙在窃听时隙之后，包含碰撞个数，多次碰撞或单词碰撞
def collision_attack(data, num):
    # 先获取包含数据和不包含数据的列表，随机挑选出包含数据的数组，在不包含数据的时隙中进行重放
    empty_array = []
    data_array = []
    for i in range(len(data)):
        sub_data = data[i][0]
        for j in range(len(sub_data)):
            if np.all(sub_data[j] == 0):
                empty_array.append((i, j))
            else:
                data_array.append((i, j))
    # print(empty_array)
    # print(data_array)
    # 进行重放，先从包含数据的列表中挑选一组重放数据
    index = np.random.randint(0, len(data_array))
    collision_data = data[data_array[index][0]][0][data_array[index][1]]

    # 取得重放数据长度
    replay_data_length = len(collision_data)
    # print(replay_data.tolist())
    # 与num比较修正重放次数，重放数据必须在重放数据帧之后的帧中进行重放
    if data_array[index][0] == len(data) - 1:
        print("the collision data is last frame, can not replay")
    elif data_array[index][0] + num > len(data) - 1:
        num = len(data) - 1 - data_array[index][0]
        print("collision num must be less total frame num , recite the num is " + str(num))
    else:
        print("collision num is " + str(num))
    # 对数据进行重放，在空时隙中进行重放，首先找到重放帧序号，重放帧在数据所在帧之后
    replay_frame = data_array[index][0]
    start_index = 0
    for i in range(len(data_array)):
        if data_array[i][0] > replay_frame:
            start_index = i
            break
    replay_empty_index = np.random.randint(start_index, len(data_array), num)
    # 对空时隙长度进行统计
    for i in range(len(replay_empty_index)):
        print("collision attack " + str(i + 1) + ":" + str(data_array[index]) + "->" + str(
            data_array[replay_empty_index[i]]))
        empty_frame = data_array[replay_empty_index[i]][0]
        empty_slot = data_array[replay_empty_index[i]][1]
        empty_data = data[empty_frame][0][empty_slot]
        # 这边要对数据进行叠加
        if replay_data_length <= len(empty_data):
            # 如果重放数据长度小于等于空时隙长度，则重放数据扩展为空时隙长度
            data[empty_frame][0][empty_slot][0:replay_data_length] += collision_data
        else:
            # 如果重放数据长度大于空时隙长度，按照空时隙长度对重放数据进行截取
            data[empty_frame][0][empty_slot] += collision_data[0:len(empty_data)]
            # print(data[empty_frame][0][empty_slot].tolist())
    return data



# 增加解调函数，通过相干解调将调制后的信号进行恢复，最终输出成文件格式进行记录，实现相干解调，输入为一个载波中的时隙信号 np.array的以为信号，
def demodulation_QPSK(data,fc):
    # 1。定义带通滤波器，信号经过带通滤波器后消除频带外噪音
    # 定义采样频率，与调制采样一致
    fs = 50000
    fc = fc
    # 带通butter滤波器设计，通带为[fc - 500，fc + 500]
    b, a = signal.butter(4, [(fc - 500) * 2 / fs, (fc + 500) * 2 / fs], 'bandpass')
    # # 低通滤波器设计，通带截止频率为2000Hz，截至频率计算方法，截止数值 * 2 / 抽样点数
    bl, al = signal.butter(4, (fc - 500) * 2 / fs, 'lowpass')
    #
    # # 通过带通滤波器滤除带外噪声
    bandpass_out1 = signal.filtfilt(b, a, data)
    # 根据信号对能量进行判断，如果能量超过阈值，则进行解调，否则直接返回"_"，代表当前时隙不存在数据
    # print(bandpass_out1.shape)
    #print(type(bandpass_out1))
    # print(bandpass_out1.T.shape)
    # print(np.dot(bandpass_out1, bandpass_out1.T))
    if np.dot(bandpass_out1, bandpass_out1.T)/len(bandpass_out1) < 0.01:
        # print("_")
        return "_"
    else:
        # 在有数据的地方进行解调
        # FFT_signal(data)
        # FFT_signal(bandpass_out1)
        # 通过带通滤波器后需要根据能量进行判断是否存在有效信号
        # 2。定义相干解调载波
        # # 相干解调,乘以同频同相的相干载波
        sample_b = 0.01
        size = int(len(data) * sample_b)
        t = np.arange(0, size, sample_b)
        ts = np.arange(0, (size / sample_b) / fs, 1 / fs)
        coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
        coherent_carrier1 = np.sin(np.dot(2 * pi * fc, ts))
        # print(len(bandpass_out1))
        # print(len(coherent_carrier))
        coherent_demodI = bandpass_out1 * coherent_carrier
        coherent_demodQ = bandpass_out1 * coherent_carrier1


        # 3。定义低通滤波器，信号经过低通滤波器进行解调，求和判决解调
        # 通过低通滤波器
        lowpass_outI = signal.filtfilt(bl, al, coherent_demodI)
        lowpass_outQ = signal.filtfilt(bl, al, coherent_demodQ)

        flagI = np.zeros(size, dtype=np.float32)
        flagQ = np.zeros(size, dtype=np.float32)

        # 对数据进行判决采样
        # 循环从1000个数据中取值，如果100此求和，如果最终的结果大于50，则为1
        for i in range(size):
            tempI = 0
            tempQ = 0
            for j in range(100):
                tempI = tempI + lowpass_outI[i * 100 + j]
                tempQ = tempQ + lowpass_outQ[i * 100 + j]
            # 定长情况下不需要进行判定
            if int(tempI) == 0:
                flagI[i] = 3
            elif int(tempQ) == 0:
                flagQ[i] = 3
            else:
                flagI[i] = np.sign(tempI)
                flagQ[i] = np.sign(tempQ)

        # print(len(flagI))
        # print(flagI)
        # print(len(flagQ))
        # print(flagQ)
        # 对IQ数据进行并串转换
        content = ""
        for i in range(size):
            content += str(int(flagI[i]+1))
            content += str(int(flagQ[i]+1))
        content = content.replace("4", "@").replace("1", "@")
        # print(content.replace("4", "@").replace("1", "@") + "\n")
        return content


# 对帧信号进行相干解调，其输入为汇总后一帧的信号，注意帧信号是0-8载波，9载波数据不全存在问题，输入为一帧的汇总数据
def frame_demodulation(signals):
    # 获取一帧帧信号
    signal = signals
    # 对一帧信号进行时隙划分，定义每帧的数据长度
    slot_len = len(signal) // 20
    slot_datas = []
    array = []
    for j in range(20):
        slot_datas.append(signal[j * slot_len: (j + 1) * slot_len])
        # FFT_signal(signal[j * slot_len: (j + 1) * slot_len])
    for k in range(len(slot_datas)):
        slot_data = slot_datas[k]
        for l in range(9):
            # 定义相干载波
            fc = (l+1) * 2048
            # 定义的十个载波依次进行解调
            bi_data = demodulation_QPSK(slot_data, fc)
            if bi_data == "_":
                array.append(bi_data)
            else:
                # 将占位符号去掉，并且将2 换成1
                bi_data = bi_data.replace("@","").replace("2", "1")
                # 去掉最后两个比特
                bi_data = bi_data[0:len(bi_data)-2]
                array.append(bi_data)
            # 存在bug通过相干解调，多解析出来两个bit，
            # array.append(bi_data)
            #print("frq-" + str(l) + "-slot-" + str(k) + ":" + bi_data + "\n")
    # print(array)

    #数据进行汇总写成文件
    contents = ""
    for i in range(10):
        if i == 9:
            contents += "_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:\n"
        else:
            # 定义一帧的数据
            frq_data = []
            for j in range(20):
                frq_data.append(array[i + j * 9])
            contents += ":".join(frq_data) + "\n"
    f = open("./NCC/demodulation.txt", "w")
    f.write(contents)
    f.close()





# 定义一个快速傅立叶FFT变化，求取调制后信号的频率特征，可以对数据进行时频域图示,输入为数组。
def FFT_signal(data):
    fy = fft(data)
    fy_real = fy.real
    fy_imag = fy.imag
    fy_abs = abs(fy)
    fy1 = fy_abs / len(data)
    fy2 = fy1[range(int(len(data) / 2))]
    xf = np.arange(len(data))
    xf1 = xf
    xf2 = xf[range(int(len(data) / 2))]
    # return fy_abs
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(xf, data, 'r')
    plt.title("original time wave")
    # plt.title("FFT of Mixed wave(two sides)")
    plt.subplot(212)
    plt.plot(xf1, fy1, 'g')
    plt.title("FFT of Mixed wave(normalization)")
    # plt.subplot(313)
    # plt.plot(xf2, fy2, 'b')
    # plt.title("FFT of Mixed wave")
    plt.show()


# 对一帧的数据进行同一时隙的相加，再进行时隙串联，最后得到以为一维数组，输入为一帧数据的二维矩阵，输出为一帧的汇总数据
def frame_con(frame_data):
    # 用np转换为矩阵进行转置，在进行相加，再进行串接
    # 先把每一载波数据转换为array
    # frame_datas = []
    # for i in range(len(frame_data)):
    #     frame_datas.append(np.array(frame_data[i]))
    lists = np.array(frame_data).transpose(1,0,2)
    datas = []
    for i in range(len(lists)):
        temp_data = np.zeros(len(lists[i][0]))
        # print(temp_data.dtype)
        for j in range(len(lists[i])):
            # print(data[i][j].dtype)
            temp_data += lists[i][j]
        datas.append(temp_data)
    return np.array(datas).flatten()


# 时频域转换函数
def show_charater(data):
    for i in range(len(data)):
        frame5 = frame_con(data[i][0:9])
        # print(type(qpsk_data[5]))
        FFT_signal(frame5)


# 读取帧文件汇总数据
data = format_data("./NCC/Frame_data.txt")
# print(data[5][6][4])
# print(len(data[5][0:10]))
# print(data[5][10])
# shape_array = []
# for i in range(len(data[5][0:10])):
#     datas = data[5][i]
#
#     for j in range(len(datas)):
#         if "0" in datas[j]:
#             shape_array.append(1)
#         else:
#             shape_array.append(0)
# # print(len(shape_array))
# print(np.array(shape_array).reshape(10,20))
# print(data[5][0:10])
qpsk_data = flame_modulation(data)
# print(data[1][0][17])
# print(len(qpsk_data[1][0][17]))
# demodulation_QPSK(qpsk_data[1][0][17], 2048)
# print(data[5])
# 对一帧信号进行相加
# 把调制之后的各个子载波在时域上进行相加
frame5 = frame_con(qpsk_data[5][0:9])
# FFT_signal(frame5)
# print(frame5)
frame_demodulation(frame5)
# print(np.array(aaa).reshape(20, 9).transpose())
# print(len(aaa))
# show_charater(qpsk_data)
# ------------------------------对帧数据进行时频域转换-------------------------------
# # 先对0-8载波进行相加，9载波时隙数有bug，不影响结论，后续解决
# for i in range(len(qpsk_data)):
#     frame5 = frame_con(qpsk_data[i][0:9])
#     # print(type(qpsk_data[5]))
#     FFT_signal(frame5)
# -----------------------------对重放攻击数据进行时频域转换---------------------------
# replay_data = replay_attack(qpsk_data, 1)
#
# for i in range(len(replay_data)):
#     frame5 = frame_con(replay_data[i][0:9])
#     # print(type(qpsk_data[5]))
#     FFT_signal(frame5)
#-----------------------------对数据碰撞情况进行时频域转换---------------------------
# collision_data = collision_attack(qpsk_data, 1)
# show_charater(collision_data)




# -----------------------读取单用户文件汇总数据------------------------
# data = format_data("./NCC/Frame_UE1.txt")
# ue_data = flame_modulation(data)
# 时频域转换
# show_charater(ue_data)











# write_all_data(qpsk_data, "original.txt")
# write_qpsk_data(qpsk_data, "original-signal.txt")
# write_qpsk_data(replay_data, "replay-attack-once.txt")
# replay num is 1
# replay attack 1:(7, 19)->(8, 4)

# replay time is 3
# replay attack 1:(4, 17)->(7, 11)
# replay attack 2:(4, 17)->(6, 1)
# replay attack 3:(4, 17)->(6, 3)
# print(np.all(qpsk_Data[0][0][0] == 0))

# empty_array:[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 12), (1, 16), (1, 17), (1, 18), (1, 19), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 12), (2, 15), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (3, 11), (3, 12), (3, 14), (3, 15), (3, 16), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 11), (5, 12), (5, 13), (5, 14), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 15), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 8), (9, 9), (9, 11), (9, 12), (9, 13)]
# data_array = [(1, 11), (1, 13), (1, 14), (1, 15), (2, 11), (2, 13), (2, 14), (2, 16), (2, 17), (2, 18), (2, 19), (3, 7),
#               (3, 10), (3, 13), (3, 17), (3, 18), (3, 19), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (5, 10),
#               (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19),
#               (7, 17), (7, 18), (7, 19), (8, 14), (8, 16), (8, 17), (8, 18), (8, 19), (9, 7), (9, 10), (9, 14), (9, 15),
#               (9, 16), (9, 17), (9, 18), (9, 19)]
# index = np.random.randint(0, len(data_array))
# print(index)
# print(type(data_array[index][1]))

# print(np.random.randint(0,9))

# print(qpsk_Data[1][0][17].tolist())
# print(type(data))

# 对调制后信号相加生成MF-TDMA信号，画图并且进行时频域调制
