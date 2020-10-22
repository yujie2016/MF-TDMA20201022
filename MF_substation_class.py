from MF_layer2_class import *
# from MF_module_class import  *
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
import math

# 子站利用layer2生成数据函数，将传输数据存储在文件中
class Substation(object):
    # 需要一个链路层数据源 layer_frame 字典形式（speed=128，time=1） 传递两个参数，一个为速率，一个为时长（时长的一个帧的时长）生成了一个帧长的数据
    # 需要一个调制方式
    # 需要频分复用过程中的载频和时隙信息，数据格式为字典格式，键为时隙信息，值为频率信息，以一个帧为例
    # 输出时域和频域的结果
    # data_length 为数据长度
    # mtu 为物理层最大传输单元
    def __init__(self, name, mtu, data_length):
        self.name = name
        self.frequency = []
        self.slot = []
        self.source_data = Layer2()
        self.signal_frequency = -1
        self.signal_slot = -1
        self.data_frequency = -1
        self.data_slot = -1
        self.data_length = data_length
        self.phy_mtu = mtu
        self.status = 0 # 定义子站的状态，0表示未接入，1表示要发送登陆时隙，2表示控制时隙，3 表示发送数据
        self.rules = [] # 解析规则，二维数组 10 * 20  解析规则中，对于信令分配，0代表不可用，1为登陆时隙，2为控制时隙 ；对于数据分配：0代表不可用 1代表可用时隙
        self.receipt = []
        self.flame = 0
        self.priority = 0
        # 定义一个函数，对类中的变量进行赋值
        #for key, value in fsinfo.items():
        #    self.frequency.append(value)
        #    self.slot.append(key)

    # 定义接收接续规则函数
    def receive_rules(self, rules, receipt):
        self.rules = rules
        self.receipt = receipt


    # 定义入网过程中文件写入函数
    def write_file(self,content):
        f = open("./RCST/RCST-" + self.name + ".txt", "a+")
        f.writelines(content)
        f.close()

    def find_position(self, str, keywords):
        pos = -1
        # print(str)
        for i in range(len(str)):
            if str[i] == keywords:
                # print("deng")
                pos = i
                break
            else:
                continue
        # print(pos)
        return pos

    def new_find_position(self, str, keywords):
        pos = ""
        for i in range(len(str)):
            if not isinstance(str[i], int):
                if str[i].isalnum():
                    if str[i] == keywords:
                        pos = i
                else:
                    s = str[i].split("+")
                    if s[0] == keywords:
                        pos = i
            else:
                continue
        return pos

    def allo_fs(self):
        slot_num = math.ceil(self.data_length / self.phy_mtu)
        m = np.mat(self.rules)
        m = np.transpose(m)[:16]
        pos_nums = 0  # 记录可用时隙数
        rule = self.rules
        if m.dtype == "int32":
            thres = 0
        else:
            thres = "0"
        for i in range(len(m)):
            # 字符与数字不能比较
            if len(np.argwhere(m[i] == thres)) > 0:
                pos_nums += 1
        m = np.transpose(m)
        if slot_num <= pos_nums:
            print(" " + self.name + "占用时隙数：" + str(slot_num))
            # 当需要时隙数小于可用的数据时，进行时隙分配，一个子站在一帧中最多分配16个时隙
            num = []
            slot_label = []
            for j in range(len(m)):
                pos_array = np.argwhere(m[j] == thres)
                num.append(len(pos_array))
            start_mm = num.index(max(num)) - 1
            mm = num.index(max(num))
            while slot_num != 0:
                pos_array = np.argwhere(m[mm] == thres)
                count = 0
                k = 0
                while (count < 2) and (slot_num > 0) and (k < num[mm]):
                    if len(np.argwhere(np.mat(slot_label) == pos_array[k][1])) == 0:
                        self.rules[mm][pos_array[k][1]] = self.name
                        # print(str(mm) + " " + str(pos_array[k][1]))
                        slot_label.append(pos_array[k][1])
                        slot_num = slot_num - 1
                        count += 1
                    k += 1
                mm = mm + 1
                # print(str(slot_num) + " " + "break")
                if mm == 10:
                    break
            while slot_num != 0:
                pos_array = np.argwhere(m[start_mm] == thres)
                count = 0
                k = 0
                while (count < 2) and (slot_num > 0) and (k < num[start_mm]):
                    if len(np.argwhere(np.mat(slot_label) == pos_array[k][1])) == 0:
                        self.rules[start_mm][pos_array[k][1]] = self.name
                        # print(str(start_mm) + " " + str(pos_array[k][1]))
                        slot_label.append(pos_array[k][1])
                        slot_num = slot_num - 1
                        count += 1
                    k += 1
                start_mm = start_mm - 1
                if start_mm < 0:
                    break
        else:
            return

    def find_data_position(self, str, keywords):
        pos = []
        for i in range(len(str)):
            if str[i] == keywords:
                pos.append(i)
            else:
                continue
        return pos

    # 定义解析分配规则函数
    def profile_rule(self, batch):
        id = int(self.name.strip("UE"))
        if self.status == 1:
            self.signal_slot = 19 - (id - 1)//10
            self.signal_frequency = (id - 1) % 10
            self.rules[self.signal_frequency][self.signal_slot] = self.name
            login_fpdu = self.source_data.logon_FPDU(id)
            # print(str(self.name) + str(int(len(login_fpdu)/8)))
            content = "batch" + ":" + str(batch) + "|status:" + str(self.status) + "|" + str(self.signal_frequency) + "|" + str(self.signal_slot) + "|" + login_fpdu + "\n"
            self.write_file(content)
            # self.status = 2
        elif self.status == 2:
            self.signal_slot = 17 - (id - 1) // 10
            self.signal_frequency = (id - 1) % 10
            self.rules[self.signal_frequency][self.signal_slot] = self.name
            control_fpdu = self.source_data.control_FPDU(id)
            # print(str(self.name) + str(int(len(control_fpdu) / 8)))
            content = "batch" + ":" + str(batch) + "|status:" + str(self.status) + "|" + str(self.signal_frequency) + "|" + str(self.signal_slot) + "|" + control_fpdu + "\n"
            self.write_file(content)
            # self.status = 3
        elif self.status == 3:
            # print(self.name)
            self.allo_fs()
            id = int(self.name.strip("UE"))
            data_fpdu = self.source_data.data_FPDU(self.phy_mtu, self.data_length, id)
            data_length = len(data_fpdu)
            print(data_length)
            pos = np.argwhere(np.mat(self.rules) == self.name)
            # print(pos)
            for i in range(len(pos)):
                self.data_frequency = pos[i][0]
                self.data_slot = pos[i][1]
                content = "batch" + ":" + str(batch) + "|status:" + str(self.status) + "|" + str(self.data_frequency) \
                          + "|" + str(self.data_slot) + "|" + data_fpdu[i] + "\n"
                self.write_file(content)
            # self.status = 0
        else:
            return

    # 定义接收回执函数
    def receive_receipt(self, batch):
        # print("./RCST_receipt/RCST-%s.txt" % self.name)
        f = open("./RCST_receipt/RCST-%s.txt" % self.name, "r")
        line = f.readline().strip()
        permission = 0
        count = 0
        while line:
            mess = line.split("|")
            bat = int(mess[0].split(":")[1])
            if bat == batch:
                per = int(mess[4][120:121])
                permission = permission + per
                count += 1
            line = f.readline().strip()
        f.close()
        if permission == count:
            permission = 1
        else:
            permission = 0
        # print("permission: " + str(permission))
        return permission


    # 对于具体的子站来说，采用one-hot编码来确定用户数据的载频和时隙 以10个载频，20个时隙为例 10 * 20 矩阵
    def module_data(self):
        # 计算时隙数
        slot_num = len(self.slot)
        # 计算每个时隙的符号数
        slot_data_num = len(self.source_data.data) // slot_num
        # 取出每个时隙的数据
        slot_datas = [self.source_data.data[i: i+slot_data_num] for i in range(0, len(self.source_data.data), slot_data_num)]
        # 对时隙中的数据按照调制方式以及频点进行跳频调制
        # 每个时隙传输的比特数是一样的
        for i in range(slot_num):
            slot_data = slot_data[i]
            slot_info = self.slot[i]
            frequency_info = self.frequency[i]
            # 进行数据调制
            
        # print(slot_datas)

def write_fileB(name, content):
    f = open("./RCST_receipt/RCST-" + name + ".txt", "a+")
    f.writelines(content)
    f.close()


def response_analysis(batch):
    pos = 0
    data_f = open("./demodulated/response_frame-%s.txt" % batch, "r")
    line = data_f.readline().strip()
    while line:
        # print(line)
        data = line.split(":")[0]
        # print(data)
        if data != "_":
            station_id = data[127:135]
            id = int(station_id, 2)
            station_name = "UE" + str(id)
            # print(station_name)
            station_status = data[123:127]
            # print(station_status)
            station_fre = pos // 20
            station_slot = pos % 20
            station_permission = data[120:121]
            content = "batch:" + str(batch) + "|status:" + station_status + "|" + str(station_fre) + "|" + str(station_slot) + "|" + data + "\n"
            # print(content)
            write_fileB(station_name, content)
        pos += 1
        line = data_f.readline().strip()
    data_f.close()



# dict1 = {"12": "1", "22": "3"}
# layer_frame = {"speed": 128, "time": 1}
# substation = Substation(dict1, layer_frame)
# substation.module_data()
# print(substation.source_data.data)
# print(substation.frequency)
# print(substation.slot)
# response_analysis(3)