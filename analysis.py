import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.fftpack import fft, ifft
import struct
import time

def static_len(file):
    f = open(file, "rb")
    line = f.readline()
    index = 0
    # 求取每行数据的的比特数大小
    t = []
    i = 0
    len_array = []
    while line:
        t.append(i)
        i += 1
        len_array.append(len(line))
        print(line)
        line = f.readline()
    data = [t, len_array]
    f.close()
    return data


def signal_fft(lens):
    fy = fft(lens)
    fy_real = fy.real
    fy_imag = fy.imag
    fy_abs = abs(fy)
    fy1 = fy_abs / len(lens)
    fy2 = fy1[range(int(len(lens) / 2))]
    xf = np.arange(len(lens))
    xf1 = xf
    xf2 = xf[range(int(len(lens) / 2))]
    return fy1


filelist = os.listdir("./substation")
# print(filelist)
index = np.random.randint(0, len(filelist), 5)


for i in range(10):
    name_count = i * 2 + 200620
    file_name = "./substation/20200312_" + str(name_count) + "_1289745_1250000_TDMA_8PSK_afterFEC.dat"
    # 子站根据210字节进行数据分割
    f = open(file_name, "rb")
    section = f.read()
    # print(bin(section[210]))
    # 每210个字符生成一个TS帧
    flames = []
    flame = []
    for i in range(len(section)):
        if i % 210 == 0:
            if i != 0:
                flames.append(flame)
            flame = []
            # 二进制
            # flame.append(bin(section[i])[2:].zfill(8))
            # 十六进制
            flame.append(hex(section[i])[2:].zfill(2))
        else:
            # 二进制
            # flame.append(bin(section[i])[2:].zfill(8))
            # 十六进制
            flame.append(hex(section[i])[2:].zfill(2))

    f.close()

    strs = []
    for j in range(len(flames)):
        str_bit = "|".join(flames[j])
        strs.append(str_bit)
    f_s = open("./mainstation/substation.txt", "a+")
    for i in range(len(strs)):
        f_s.write(strs[i] + "\n")
    f_s.close()




# 主站根据42 00 00 00进行数据分割
def split_Str(temp_str):
    temp_str = "xx" + temp_str + "xx"
    str_list = temp_str.split("42")
    # print(str_list)
    contents = []
    content = ""
    # 判断字符串以42结尾
    last_idx = len(str_list) - 1
    if str_list[last_idx] == "xx":
        str_list.pop(last_idx)
        str_list[last_idx - 1] = str_list[last_idx - 1] + "42"
    else:
        str_list[last_idx] = str_list[last_idx][:-2]
    # 判断字符串以42开头
    if str_list[0] == "xx":
        str_list[1] = "42" + str_list[1]
        str_list.pop(0)
    else:
        str_list[0] = str_list[0][2:]
    # print(str_list)

    for i in range(len(str_list)):
        # 如果字符串长度是偶数，则将字符串赋值到content中
        # if i == 0:
        #     content = str_list[0]
        if len(str_list[i]) % 2 == 0:
            if str_list[i][0:6] == "000000":
                contents.append(content)
                content = "42" + str_list[i]
            else:
                pre_str = ""
                if i != 0:
                    pre_str = "42"
                content = content + pre_str + str_list[i]
        else:
            pre_str = ""
            if i != 0:
                pre_str = "42"
            content = content + pre_str + str_list[i]
    contents.append(content)
    return contents


def write_file(final_contents):
    f_w = open("./mainstation/split.txt", "a+")
    # f_w.write()
    for i in range(len(final_contents)):
        content = []
        for j in range(len(final_contents[i])):
            if j % 2 == 0:
                content.append(final_contents[i][j:j + 2])
        if i == 0 and final_contents[0][0:2] != "42":
            f_w.write("|".join(content))
        elif i == 0 and final_contents[0][0:2] == "42":
            f_w.write("\n" + "|".join(content) + "\n")
        elif i == len(final_contents) - 1:
            f_w.write("|".join(content))
        else:
            f_w.write("|".join(content) + "\n")
    f_w.close()


def unit_content(main_contents, start_pos, end_pos):
    t_str = ""
    for i in range(end_pos - start_pos):
        t_str = t_str + main_contents[start_pos + i]
    return t_str


def count_byte(main_contents):
    count = 0
    for i in range(len(main_contents)):
        count += len(main_contents[i])
    print(count)


f = open("./mainstation/20200312_200212_1343336_6781081_gs_IDirect Evolution_X7_FL.dat", "rb")
size = 1024
index = 0
end_line_num = 10
main_contents = []
while True:
    if index % end_line_num == 0 and index != 0:
        # 每读取10行生成一个main_contents
        index_count = []
        for i in range(len(main_contents)):
            if i == 0:
                continue
            else:
                if main_contents[i][0:6] == "000000" and main_contents[i - 1][:-2] == "42":
                    main_contents[i - 1] = main_contents[i - 1][:-2]
                    main_contents[i] = "42" + main_contents[i]
                    index_count.append(i)
                elif main_contents[i][0:8] == "42000000":
                    index_count.append(i)

        if len(index_count) > 0 and index_count[len(index_count) - 1] != len(main_contents):
            index_count.append(len(main_contents))
        # 根据索引值合并索引间的元素
        final_contents = []
        index_sum = len(index_count)
        for i in range(index_sum):
            if i == 0:
                start_pos = 0
            else:
                start_pos = index_count[i - 1]
            end_pos = index_count[i]
            final_contents.append(unit_content(main_contents, start_pos, end_pos))
        count_byte(main_contents)
        count_byte(final_contents)
        write_file(final_contents)
        # 清空main_conetents
        index += 1
        time.sleep(0.5)
        main_contents = []
    else:
        content = f.read(size)
        str1 = ""
        for i in range(size):
            str1 = str1 + str(hex(content[i])[2:].zfill(2))
        index += 1
        # print(str1)
        main_content = split_Str(str1)
        main_contents.extend(main_content)
f.close()

