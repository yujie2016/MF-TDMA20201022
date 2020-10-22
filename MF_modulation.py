import os
import random
import time
from math import pi
import matplotlib
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
import numpy as np
import math
import scipy.signal as signal

size = 24
sampling_t = 0.01
# # 定义两个载频
# 定义载波频率为1024hz
fc1 = 2048
# fc2 = 10000
# # 定义采样频率，采样频率要大于信号带宽的两倍
fs = 50000
# # 100 为0.01相当于1个符号采样100个点
ts = np.arange(0, (100 * size) / fs, 1 / fs)
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
t = np.arange(0, size, sampling_t)


def tto(I, Q):
    data1 = []
    data = []
    for i in range(len(I)):
        data.append(I[i])
        data.append(Q[i])
    for j in range(len(data)):
        data1.append(int((data[j]+1)/2))
    return data1


def module_signal(signal):
    I = np.zeros(len(t), dtype=np.float32)
    Q = np.zeros(len(t), dtype=np.float32)
    # 循环给m赋值
    for i in range(len(t)):
        I[i] = signal[0][math.floor(t[i])]
        Q[i] = signal[1][math.floor(t[i])]
    # fig = plt.figure()
    # ax1 = fig.add_subplot(4, 3, 1)
    # # 画出I路信号
    # ax1.set_title('I路信号', fontproperties=zhfont1, fontsize=15)
    # # 定义图的横纵坐标范围
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, I, 'b')
    # # 画出Q路信号
    # ax2 = fig.add_subplot(4, 3, 4)
    # # 解决set_title中文乱码
    # ax2.set_title('Q路信号', fontproperties=zhfont1, fontsize=15)
    # # 定义图的横纵坐标范围
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, Q, 'b')
    # # np.dot返回两个数组的点积，ts为一个数组，返回前一个数与后一个数组的积，为一个数组
    coherent_carrier = np.cos(np.dot(2 * pi * fc1, ts))
    coherent_carrier1 = np.sin(np.dot(2 * pi * fc1, ts))
    # # 两个数组相乘，得到的也是一个数组，两个数组对应下标的乘积，同时两个数组的长度要一致
    qpsk = I * coherent_carrier + Q * coherent_carrier1
    #
    # # BPSK调制信号波形
    # ax3 = fig.add_subplot(4, 3, 7)
    # ax3.set_title('QPSK调制信号', fontproperties=zhfont1, fontsize=15)
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, qpsk, 'r')
    return qpsk


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


def demodule_signal(signal1):
    # fig = plt.figure()
    # 带通buffer滤波器设计，通带为[1500，2500]
    b, a = signal.butter(4, [1500 * 2 / fs, 2500 * 2 / fs], 'bandpass')
    # # 低通滤波器设计，通带截止频率为2000Hz，截至频率计算方法，截止数值 * 2 / 抽样点数
    bl, al = signal.butter(4, 1500 * 2 / fs, 'lowpass')
    #
    # # 通过带通滤波器滤除带外噪声
    bandpass_out1 = signal.filtfilt(b, a, signal1)
    #
    # # 相干解调,乘以同频同相的相干载波
    coherent_carrier = np.cos(np.dot(2 * pi * fc1, ts))
    coherent_carrier1 = np.sin(np.dot(2 * pi * fc1, ts))
    coherent_demodI = bandpass_out1 * coherent_carrier
    coherent_demodQ = bandpass_out1 * coherent_carrier1

    # 通过低通滤波器
    lowpass_outI = signal.filtfilt(bl, al, coherent_demodI)
    lowpass_outQ = signal.filtfilt(bl, al, coherent_demodQ)

    # bx1 = fig.add_subplot(4, 3, 8)
    # bx1.set_title('I路信号相干解调后的信号', fontproperties=zhfont1, fontsize=15)
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, lowpass_outI, 'r')
    #
    # bx2 = fig.add_subplot(4, 3, 11)
    # bx2.set_title('Q路信号相干解调后的信号', fontproperties=zhfont1, fontsize=15)
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, lowpass_outQ, 'r')

    # 抽样判决
    # 抽样的数据为1000个，类型为float
    detection_I = np.zeros(len(t), dtype=np.float32)
    detection_Q = np.zeros(len(t), dtype=np.float32)
    # 译码的数据为10个，类型为float
    flagI = np.zeros(size, dtype=np.float32)
    flagQ = np.zeros(size, dtype=np.float32)

    # 循环从1000个数据中取值，如果100此求和，如果最终的结果大于50，则为1
    for i in range(size):
        tempI = 0
        tempQ = 0
        for j in range(100):
            tempI = tempI + lowpass_outI[i * 100 + j]
            tempQ = tempQ + lowpass_outQ[i * 100 + j]

        flagI[i] = np.sign(tempI)
        flagQ[i] = np.sign(tempQ)
    # 将抽样数据规整，100个值要求一致
    for i in range(size):
        if flagI[i] == 1:
            for j in range(100):
                detection_I[i * 100 + j] = 1
        else:
            for j in range(100):
                detection_I[i * 100 + j] = -1
        if flagQ[i] == 1:
            for j in range(100):
                detection_Q[i * 100 + j] = 1
        else:
            for j in range(100):
                detection_Q[i * 100 + j] = -1
    # bx2 = fig.add_subplot(4, 3, 2)
    # bx2.set_title('I路抽样判决后的信号', fontproperties=zhfont1, fontsize=15)
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, detection_I, 'r')
    # bx2 = fig.add_subplot(4, 3, 5)
    # bx2.set_title('Q路抽样判决后的信号', fontproperties=zhfont1, fontsize=15)
    # plt.axis([0, size, -2, 2])
    # plt.plot(t, detection_Q, 'r')
    return tto(flagI, flagQ)


# 场景一，固定模式，所有时隙发送，一个malignant substation，5个normal substation，分为全干扰和部分干扰
m_signal = 2 * np.random.randint(0, 2, (2, 24)) - 1
# print(len(m_signal[0]))
m_s = tto(m_signal[0], m_signal[1])
n_signal1 = 2 * np.random.randint(0, 2, (2, 24)) - 1
n_s = tto(n_signal1[0], n_signal1[1])
n_signal2 = 2 * np.random.randint(0, 2, (2, 24)) - 1
n_s2 = tto(n_signal2[0], n_signal2[1])
n_signal3 = 2 * np.random.randint(0, 2, (2, 24)) - 1
n_s3 = tto(n_signal3[0], n_signal3[1])
n_signal4 = 2 * np.random.randint(0, 2, (2, 24)) - 1
n_s4 = tto(n_signal4[0], n_signal4[1])
n_signal5 = 2 * np.random.randint(0, 2, (2, 24)) - 1
n_s5 = tto(n_signal5[0], n_signal5[1])

# 场景二，干扰序列不固定，有特征，特殊字段固定，其余字段不固定，分为全干扰和部分干扰
m_signal1 = 2 * np.random.randint(0, 2, (2, 24)) - 1
m_signal2 = 2 * np.random.randint(0, 2, (2, 24)) - 1
m_signal3 = 2 * np.random.randint(0, 2, (2, 24)) - 1
m_signal4 = 2 * np.random.randint(0, 2, (2, 24)) - 1

# 每一个恶意信号将10个bit值固定
RN = 12
index = np.random.randint(0, 24, RN)
I_fix_signal = np.random.randint(0, 2, RN) * 2 - 1
Q_fix_signal = np.random.randint(0, 2, RN) * 2 - 1
print(index)

for i in range(RN):
    m_signal[0][index[i]] = I_fix_signal[i]
    m_signal[1][index[i]] = Q_fix_signal[i]
    m_signal1[0][index[i]] = I_fix_signal[i]
    m_signal1[1][index[i]] = Q_fix_signal[i]
    m_signal2[0][index[i]] = I_fix_signal[i]
    m_signal2[1][index[i]] = Q_fix_signal[i]
    m_signal3[0][index[i]] = I_fix_signal[i]
    m_signal3[1][index[i]] = Q_fix_signal[i]
    m_signal4[0][index[i]] = I_fix_signal[i]
    m_signal4[1][index[i]] = Q_fix_signal[i]



qpskm = module_signal(m_signal)
qpskm1 = module_signal(m_signal1)
qpskm2 = module_signal(m_signal2)
qpskm3 = module_signal(m_signal3)
qpskm4 = module_signal(m_signal4)
qpskn1 = module_signal(n_signal1)
qpskn2 = module_signal(n_signal2)
qpskn3 = module_signal(n_signal3)
qpskn4 = module_signal(n_signal4)
qpskn5 = module_signal(n_signal5)

sum_qpsk1 = qpskm + qpskn1
sum_qpsk2 = qpskn2
sum_qpsk3 = qpskm2 + qpskn3
sum_qpsk4 = qpskn4
sum_qpsk5 = qpskm4 + qpskn5


d_s1 = demodule_signal(sum_qpsk1)
d_s2 = demodule_signal(sum_qpsk2)
d_s3 = demodule_signal(sum_qpsk3)
d_s4 = demodule_signal(sum_qpsk4)
d_s5 = demodule_signal(sum_qpsk5)

# 场景二图示
x = np.arange(48)
plt.figure()
plt.subplot(531)
plt.step(x, tto(m_signal[0], m_signal[1]))
plt.subplot(534)
# plt.step(x, tto(m_signal1[0], m_signal1[1]))
plt.subplot(537)
plt.step(x, tto(m_signal2[0], m_signal2[1]))
plt.subplot(5, 3, 10)
# plt.step(x, tto(m_signal3[0], m_signal3[1]))
plt.subplot(5, 3, 13)
plt.step(x, tto(m_signal4[0], m_signal4[1]))
plt.subplot(532)
plt.step(x, n_s)
plt.subplot(535)
plt.step(x, n_s2)
plt.subplot(538)
plt.step(x, n_s3)
plt.subplot(5, 3, 11)
plt.step(x, n_s4)
plt.subplot(5, 3, 14)
plt.step(x, n_s5)
plt.subplot(533)
plt.step(x, d_s1)
plt.subplot(536)
plt.step(x, d_s2)
plt.subplot(539)
plt.step(x, d_s3)
plt.subplot(5, 3, 12)
plt.step(x, d_s4)
plt.subplot(5, 3, 15)
plt.step(x, d_s5)

# 场景一图示
# plt.figure()
# # x = np.arange(48)
# plt.subplot(621)
# # plt.step(x, m_s)
# plt.subplot(623)
# plt.step(x, n_s)
# plt.subplot(624)
# plt.step(x, d_s1)
# plt.subplot(625)
# plt.step(x, n_s2)
# plt.subplot(626)
# plt.step(x, d_s2)
# plt.subplot(627)
# plt.step(x, n_s3)
# plt.subplot(628)
# plt.step(x, d_s3)
# plt.subplot(629)
# plt.step(x, n_s4)
# plt.subplot(6, 2, 10)
# plt.step(x, d_s4)
# plt.subplot(6, 2, 11)
# plt.step(x, n_s5)
# plt.subplot(6, 2, 12)
# plt.step(x, d_s5)

plt.show()
