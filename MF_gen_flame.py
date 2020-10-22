from MF_mainstation_class import *
from MF_substation_class import *
import time
from MF_gen_flame import *
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# 根据子站生成的数据形成每帧的比特流文件

def generate_frame():
    f = open("./NCC/NCC.txt", "r")
    data_f = open("./NCC/Frame_data.txt", "w")
    line = f.readline()
    index = 0
    while line:
        # print(line)
        eles = line.split(":")
        # 确定 帧 的下标
        if eles[0] == "init":
            index = 0
        else:
            index = eles[0]
        rule = eval(eles[1])
        # 循环读取计划表中的值，再将生成的二进制数据写入文件中
        content = ""
        for i in range(len(rule)):
            for j in range(len(rule[i])):
                if isinstance(rule[i][j], int):
                    content += "_:"
                else:
                    file_name = ("./RCST/RCST-%s.txt" % rule[i][j])
                    f_temp = open(file_name, "r")
                    line_temp = f_temp.readline()
                    while line_temp:
                        sub_eles = line_temp.split("|")
                        sub_frame = sub_eles[0].split(":")[1]
                        sub_fr = sub_eles[2]
                        sub_slot = sub_eles[3]
                        sub_data = sub_eles[4]
                        if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                            content += (str(sub_data) + ":")
                        line_temp = f_temp.readline()
                        line_temp = line_temp.strip()####
                    f_temp.close()
            content += "\n"
        data_f.write("frame:" + str(index) + "\n" + content)

        line = f.readline()
    f.close()
    data_f.close()

# 生成一帧的数据
def one_frame(batch):
    f = open("./NCC/NCC.txt", "r")
    batch = str(batch)
    data_f = open("./frame/frame-%s.txt" % batch, "w")
    line = f.readline()
    index = 0
    while line:
        eles = line.split(":")
        # 确定 帧 的下标
        if eles[0] == batch:
            index = batch
            rule = eval(eles[1])
            # 循环读取计划表中的值，再将生成的二进制数据写入文件中
            content = ""
            for i in range(len(rule)):
                for j in range(len(rule[i])):
                    if isinstance(rule[i][j], int):
                        content += "_:"
                        # content += '\n'
                    else:
                        file_name = ("./RCST/RCST-%s.txt" % rule[i][j])
                        f_temp = open(file_name, "r")
                        line_temp = f_temp.readline()
                        while line_temp:
                            sub_eles = line_temp.split("|")
                            sub_frame = sub_eles[0].split(":")[1]
                            sub_fr = sub_eles[2]
                            sub_slot = sub_eles[3]
                            sub_data = sub_eles[4].strip()
                            if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                            # if int(sub_frame) == int(index):
                                content += (str(sub_data) + ":")
                                # content += (str(sub_data) + ":" + '\n')
                            line_temp = f_temp.readline()
                            line_temp = line_temp.strip()
                        f_temp.close()
                content += '\n'
            data_f.write(content)
        line = f.readline()
    f.close()
    data_f.close()

def new_generate_frame():
    f = open("./NCC/NCC.txt", "r")
    data_f = open("./NCC/Frame_data.txt", "w")
    line = f.readline()
    index = 0
    while line:
        # print(line)
        eles = line.split(":")
        # 确定 帧 的下标
        if eles[0] == "init":
            index = 0
        else:
            index = eles[0]
        rule = eval(eles[1])
        # 循环读取计划表中的值，再将生成的二进制数据写入文件中
        content = ""
        for i in range(len(rule)):
            for j in range(len(rule[i])):
                if isinstance(rule[i][j], int):
                    content += "_:"
                else:
                    if rule[i][j].isalnum():
                        file_name = ("./RCST/RCST-%s.txt" % rule[i][j])
                        f_temp = open(file_name, "r")
                        line_temp = f_temp.readline()
                        while line_temp:
                            sub_eles = line_temp.split("|")
                            sub_frame = sub_eles[0].split(":")[1]
                            sub_fr = sub_eles[2]
                            sub_slot = sub_eles[3]
                            sub_data = sub_eles[4]
                            if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                                content += (str(sub_data) + ":")
                            line_temp = f_temp.readline()
                            line_temp = line_temp.strip()
                        f_temp.close()
                    else:
                        user = rule[i][j].split("+")
                        user1 = user[0]
                        user2 = user[1]
                        print(user1 + " " + user2)
                        # 对合法子站进行统计
                        file_name1 = ("./RCST/RCST-%s.txt" % user1)
                        f_temp1 = open(file_name1, "r")
                        line_temp1 = f_temp1.readline()
                        while line_temp1:
                            sub_eles = line_temp1.split("|")
                            sub_frame = sub_eles[0].split(":")[1]
                            sub_fr = sub_eles[2]
                            sub_slot = sub_eles[3]
                            sub_data = sub_eles[4]
                            if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                                content += (str(sub_data) + ":")
                            line_temp1 = f_temp1.readline()
                            line_temp1 = line_temp1.strip()
                        f_temp1.close()
                        # 对恶意子站进行统计
                        file_name2 = ("./RCST/RCST-%s.txt" % user2)
                        f_temp2 = open(file_name2, "r")
                        line_temp2 = f_temp2.readline()
                        while line_temp2:
                            sub_eles = line_temp2.split("|")
                            sub_frame = sub_eles[0].split(":")[1]
                            sub_fr = sub_eles[2]
                            sub_slot = sub_eles[3]
                            sub_data = sub_eles[4]
                            if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                                content += (str(sub_data) + ":")
                            line_temp2 = f_temp2.readline()
                            line_temp2 = line_temp2.strip()
                        f_temp2.close()
            content += "\n"
        data_f.write("frame:" + str(index) + "\n" + content)

        line = f.readline()
    f.close()
    data_f.close()

def one_response_frame(batch):
    f = open("./NCC/receipt.txt", "r")
    batch = str(batch)
    data_f = open("./frame/response_frame-%s.txt" % batch, "w")
    line = f.readline()
    index = 0
    while line:
        eles = line.split(":")
        # 确定 帧 的下标
        if eles[0] == batch:
            index = batch
            rule = eval(eles[1])
            # 循环读取计划表中的值，再将生成的二进制数据写入文件中
            content = ""
            for i in range(len(rule)):
                for j in range(len(rule[i])):
                    if isinstance(rule[i][j], int):
                        content += "_:"
                    else:
                        file_name = ("./response/RCST-%s.txt" % rule[i][j])
                        f_temp = open(file_name, "r")
                        line_temp = f_temp.readline()
                        while line_temp:
                            sub_frame = line_temp.split("|")[0].split(":")[1]
                            sub_data = line_temp.split("|")[4].strip()
                            sub_freq = line_temp.split("|")[2]
                            sub_slot = line_temp.split("|")[3]
                            if int(sub_frame) == int(index) and int(sub_freq) == i and int(sub_slot) == j:
                                content += (str(sub_data) + ":")
                            line_temp = f_temp.readline()
                            line_temp = line_temp.strip()
                        f_temp.close()
                content += '\n'
            data_f.write(content)
        line = f.readline()
    f.close()
    data_f.close()

def response_frame():
    data_f = open("./NCC/response_frame_data.txt", "w")
    for i in range(10):
        content = ""
        file_name = ("./frame/response_frame-%s.txt" % (i+1))
        f = open(file_name, "r")
        line = f.readline()
        while line:
            data = line.split(":")[0]
            if data == "_":
                content += "_:"
            else:
                frag = []
                num = int(len(data)/8)
                # print(len(data))
                # print(num)
                for j in range(num):
                    # print(j)
                    frag.append(data[j*8:(j+1)*8])
                for j in range(num):
                    frag[j] = hex(int(frag[j], 2))[2:]
                    if len(frag[j]) == 1:
                        frag[j] = "0" + frag[j]
                    content += frag[j] + '\000'
            line = f.readline().strip()
        f.close()
        data_f.write("frame:" + str(i+1) + "\n" + content + '\n')
    data_f.close()


# response_frame()
# response_frame()
# new_generate_frame()
# one_response_frame(3)
