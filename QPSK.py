import numpy as np
from math import pi
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib
import scipy.signal as signal
import math
from scipy.fftpack import fft, ifft
# matplotlib.use('Agg')


def FFT_signal(data, filename):
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
    # plt.show()
    plt.savefig("./Images/" + filename + ".png")
    plt.close()


def flame_format_data(flame_data):
    # 1找到时隙中最长的比特数
    max = 1504
    data = []
    r_data = []
    for i in range(len(flame_data)):
        split_data = flame_data[i].split(":")
        data.extend(split_data)
    # 2对字符串进行扩充
    if max > 1:
        for j in range(len(data)):
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
    frame_data = []
    while line:
        frame_data.append(line)
        line = f.readline().strip(":\n")
    f.close()
    data = flame_format_data(frame_data)
    return data


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


def flame_modulation(data):
    flame_data = data
    fc = 2048
    qpsk_signal = []
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
            # print(slot_data)
            fc_data_qpsk.append(qpsk)
        qpsk_signal.append(fc_data_qpsk)
        # print("next")
    return qpsk_signal


def frame_con(frame_data):
    # 用np转换为矩阵进行转置，在进行相加，再进行串接
    # 先把每一载波数据转换为array
    # frame_datas = []
    # for i in range(len(frame_data)):
    #     frame_datas.append(np.array(frame_data[i]))
    lists = np.array(frame_data).transpose(1, 0, 2)
    datas = []
    for i in range(len(lists)):
        temp_data = np.zeros(len(lists[i][0]))
        # print(temp_data.dtype)
        for j in range(len(lists[i])):
            # print(data[i][j].dtype)
            temp_data += lists[i][j]
        datas.append(temp_data)
    return np.array(datas).flatten()


def modulate(batch):
    data = format_data("./frame/frame-" + str(batch) + ".txt")
    qpsk_data = flame_modulation(data)
    # frame = frame_con(qpsk_data[0:9])
    frame = frame_con(qpsk_data)
    FFT_signal(frame, "frame" + str(batch) + "-modulate")
    file_name = ("./modulated/frame-%s.npy" % batch)
    # f = open("./modulated/frame-" + str(batch) + ".txt", "a+")
    np.save(file_name, frame)


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


def frame_demodulation(signals, batch):
    # 获取一帧帧信号
    signal = signals
    # print(len(signal))
    # 对一帧信号进行时隙划分，定义每帧的数据长度
    slot_len = len(signal) // 20
    slot_datas = []
    array = []
    for j in range(20):
        slot_datas.append(signal[j * slot_len: (j + 1) * slot_len])
        # FFT_signal(signal[j * slot_len: (j + 1) * slot_len])
    for k in range(len(slot_datas)):
        slot_data = slot_datas[k]
        for l in range(10):
            # 定义相干载波
            fc = (l + 1) * 2048
            # 定义的十个载波依次进行解调
            bi_data = demodulation_QPSK(slot_data, fc)
            if bi_data == "_":
                array.append(bi_data)
            else:
                # 将占位符号去掉，并且将2 换成1
                bi_data = bi_data.replace("@", "").replace("2", "1")
                # 去掉最后两个比特
                bi_data = bi_data[0:len(bi_data) - 2]
                array.append(bi_data)
    # print(len(array))
            # 存在bug通过相干解调，多解析出来两个bit，
            # array.append(bi_data)
            # print("frq-" + str(l) + "-slot-" + str(k) + ":" + bi_data + "\n")
    # print(array)

    # 数据进行汇总写成文件
    contents = ""
    for i in range(10):
        if i == 10:
            contents += "_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:\n"
        else:
            # 定义一帧的数据
            frq_data = []
            for j in range(20):
                # print(i + j * 10)
                frq_data.append(array[i + j * 10])
            contents += ":".join(frq_data) + "\n"
    f = open("./trash/frame-" + str(batch) + ".txt", "w")
    f.write(contents)
    f.close()
    data_f = open("./trash/frame-" + str(batch) + ".txt", "r")
    data = ""
    line = data_f.readline().strip()
    while line:
        freq = line.split(":")
        for i in range(len(freq)):
            data += freq[i] + ":" + '\n'
        line = data_f.readline().strip()
    data_f.close()
    f_temp = open("./demodulated/frame-" + str(batch) + ".txt", "w")
    f_temp.write(data)
    f_temp.close()


def response_frame_demodulation(signals, batch):
    # 获取一帧帧信号
    signal = signals
    # print(len(signal))
    # 对一帧信号进行时隙划分，定义每帧的数据长度
    slot_len = len(signal) // 20
    slot_datas = []
    array = []
    for j in range(20):
        slot_datas.append(signal[j * slot_len: (j + 1) * slot_len])
        # FFT_signal(signal[j * slot_len: (j + 1) * slot_len])
    for k in range(len(slot_datas)):
        slot_data = slot_datas[k]
        for l in range(10):
            # 定义相干载波
            fc = (l + 1) * 2048
            # 定义的十个载波依次进行解调
            bi_data = demodulation_QPSK(slot_data, fc)
            if bi_data == "_":
                array.append(bi_data)
            else:
                # 将占位符号去掉，并且将2 换成1
                bi_data = bi_data.replace("@", "").replace("2", "1")
                # 去掉最后两个比特
                bi_data = bi_data[0:len(bi_data) - 2]
                array.append(bi_data)

    contents = ""
    for i in range(10):
        if i == 10:
            contents += "_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:_:\n"
        else:
            # 定义一帧的数据
            frq_data = []
            for j in range(20):
                # print(i + j * 10)
                frq_data.append(array[i + j * 10])
            contents += ":".join(frq_data) + "\n"
    f = open("./trash/response_frame-" + str(batch) + ".txt", "w")
    f.write(contents)
    f.close()
    data_f = open("./trash/response_frame-" + str(batch) + ".txt", "r")
    data = ""
    line = data_f.readline().strip()
    while line:
        freq = line.split(":")
        for i in range(len(freq)):
            data += freq[i] + ":" + '\n'
        line = data_f.readline().strip()
    data_f.close()
    f_temp = open("./demodulated/response_frame-" + str(batch) + ".txt", "w")
    f_temp.write(data)
    f_temp.close()


def demodulate(batch):
    file_name = ("./modulated/frame-%s.npy" % batch)
    data = np.load(file_name)
    frame_demodulation(data, batch)



def response_modulate(batch):
    o_file = open("./frame/response_frame-" + str(batch) + ".txt", "r")
    line = o_file.readline().strip()
    content = ""
    index = 0
    while line:
        index += 1
        if index == 20:
            content += line + '\n'
            index = 0
        else:
            content += line
        line = o_file.readline().strip()

    o_file.close()
    n_file = open("./response_frame/frame-" + str(batch) + ".txt", "w")
    n_file.write(content)
    n_file.close()
    data = format_data("./response_frame/frame-" + str(batch) + ".txt")
    qpsk_data = flame_modulation(data)
    frame = frame_con(qpsk_data)
    FFT_signal(frame, "frame" + str(batch) + "-demodulate")
    file_name = ("./modulated/response_frame-%s.npy" % batch)
    # f = open("./modulated/frame-" + str(batch) + ".txt", "a+")
    np.save(file_name, frame)


def response_demodulate(batch):
    file_name = ("./modulated/response_frame-%s.npy" % batch)
    data = np.load(file_name)
    response_frame_demodulation(data, batch)


# modulate(6)

# for i in range(10):
#     response_modulate(i+1)
# response_modulate(5)
# demodulate(6)
# for i in range(10):
#     modulate(i + 1)
#     demodulate(i + 1)

# response_modulate(3)
# response_demodulate(3)

