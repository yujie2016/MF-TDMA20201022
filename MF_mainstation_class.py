import MF_substation_class
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# warnings.simplefilter(action='ignore', category=FutureWarning)

# 主站用于生成MF-TDMA的分配规则，存储在文件当中。 分配策略 一个信令信道，多个数据信道
class Mainstation(object):
    # 传入一个子站个数ss_num，载频个数frq_num，时隙个数slot_num，根据生成子站计划，依次传入各个子站当中
    # 计算每个载频上时域的信号
    # 总的频域信息
    # 以10个载频 每个载频上20个时隙为里进行分配。 假设一个主站控制20个子站交互，子站接入按照泊松分布进行 0信道默认为信令信道
    def __init__(self, ss_num, frq_num, slot_num):
        self.ss_num = ss_num
        self.frq_num = frq_num
        self.slot_num = slot_num
        self.substations = []
        self.rules = []
        self.receipt = []
        self.init_rules()
        # self.fs_infos = []  # 格式为{频率：时隙} 键值对
        # 调用分配函数
        # self.allocate_fs()
        # for i in range(self.ss_num):
        #     layer_frame = {"speed": 128, "time": 1}
        #     substation = MF_substation_class.Substation(self.fs_infos[i],layer_frame)
        #     self.substations.append(substation)

    # 定义一个初始化的10 * 20 时隙计划安排矩阵以及一个3*20的回执信道
    def init_rules(self):
        for i in range(self.frq_num):
            slot = []
            for j in range(self.slot_num):
                slot.append(0)
            self.rules.append(slot)
        for i in range(10):
            slot = []
            for j in range(20):
                slot.append(0)
            self.receipt.append(slot)
        f = open("./NCC/NCC.txt", "w")
        f.writelines("init:" + str(self.rules) + "\n")
        f.close()
        f = open("./NCC/receipt.txt", "w")
        f.writelines("init:" + str(self.receipt) + "\n")
        f.close()
        # print(self.receipt)

    # 定义一个清除UE状态的函数
    def clear_rules(self):
        m = np.mat(self.rules)
        for i in range(len(self.substations)):
            if m.dtype != "int64":
                new_name = self.substations[i].name + "+UEevil"
                pos2 = np.argwhere(m == "UEevil")
                pos1 = np.argwhere(m == new_name)
                pos = np.argwhere(m == self.substations[i].name)
                if len(pos) > 0:
                    for j in range(len(pos)):
                        if pos[j][0] == 0:
                            self.rules[pos[j][0]][pos[j][1]] = 0
                            # self.rules[pos[j][0]][pos[j][1]] = random.randint(1, 2)
                        else:
                            self.rules[pos[j][0]][pos[j][1]] = 0
                if len(pos1) > 0:
                    for j in range(len(pos1)):
                        if pos1[j][0] == 0:
                            self.rules[pos1[j][0]][pos1[j][1]] = 0
                            # self.rules[pos1[j][0]][pos1[j][1]] = random.randint(1, 2)
                        else:
                            self.rules[pos1[j][0]][pos1[j][1]] = 0
                if len(pos2) > 0:
                    for j in range(len(pos2)):
                        if pos2[j][0] == 0:
                            self.rules[pos2[j][0]][pos2[j][1]] = 0
                            # self.rules[pos2[j][0]][pos2[j][1]] = random.randint(1, 2)
                        else:
                            self.rules[pos2[j][0]][pos2[j][1]] = 0

    def clear_receipt(self):
        for i in range(10):
            for j in range(20):
                self.receipt[i][j] = 0

    # 定义更新函数，根据子站的属性进行更新，需传递具体子站
    def update_rule_by_substation(self, substation):
        if substation.status == 1 or substation.status == 2:
            # print(substation.signal_slot)
            # print (substation.signal_frequency)
            self.rules[substation.signal_frequency][substation.signal_slot] = substation.name
        elif substation.status == 3:
            self.allo_fs(substation)
        else:
            return

    # 定义写文件函数，记录主站的时隙分配
    def write_file(self, content):
        f = open("./NCC/NCC.txt", "a+")
        f.writelines(content)
        f.close()

    def write_receipt_file(self, content):
        f = open("./NCC/receipt.txt", "a+")
        f.writelines(content)
        f.close()

    def write_receive_info(self, content, batch):
        filename = ("./receive_info/info-%s.txt" % batch)
        f = open(filename, "w")
        f.writelines(content)
        f.close()

    def write_response_data(self, name, content):
        f = open("./response/RCST-%s.txt" % name, "a+")
        f.writelines(content)
        f.close()

    def analysis(self, batch):
        data_f = open("./demodulated/frame-%s.txt" % batch, "r")
        line = data_f.readline()
        index = 0
        all_info = ""
        while line:
            # print("test")
            data = line.split(":")[0]
            if data != "_" and len(data) == 1502:
                information = self.analysis_TS(data)
                # print(information)
                identity = information[0:8]
                RCST_status = information[8:12]
                id = int(identity, 2)
                if RCST_status == "0100":
                    status = "控制"
                elif RCST_status == "0010":
                    status = "登陆"
                elif RCST_status == "1000":
                    status = "数据"
                all_info += identity + RCST_status + '\n'
                print("identity:" + str(id) + " status:" + status + " frequency:" + str(index // 20) + "  slot:" + str(index % 20))
            index += 1
            line = data_f.readline()
        data_f.close()
        return all_info

    def analysis_TS(self, TS_packet):
        adaptation_length = TS_packet[32:40]
        adaptation_length = int(adaptation_length, 2)
        header_adapt = (4 + adaptation_length) * 8
        payload = TS_packet[header_adapt:]
        identity = payload[0:8]
        RCST_status = payload[8:12]
        return identity + RCST_status

    def send_receipt(self, batch):
        filename = ("./receive_info/info-%s.txt" % batch)
        f = open(filename, "r")
        line = f.readline()
        data_fpdu = []
        while line:
            identity = line[0:8]
            status = line[8:12]
            id = int(identity, 2)
            if status != "1000":
                content = self.substations[id - 1].source_data.response(identity, status)
                response = self.substations[id - 1].source_data.GSpacket(content)
                name = "UE" + str(int(identity, 2))
                frequency = (id - 1) // 2
                slot = self.substations[id - 1].find_position(self.receipt[frequency], 0)
                self.receipt[frequency][slot] = name
                data = "batch:" + str(batch) + "|" + "status:" + status + "|" + str(frequency) + "|" + str(
                    slot) + "|" + response + '\n'
                self.write_response_data(name, data)
            else:
                data_fpdu.append(identity)
            line = f.readline()
        data_num = set(data_fpdu)
        for k in data_num:
            identity = k
            id = int(identity, 2)
            status = "1000"
            content = self.substations[id - 1].source_data.response(identity, status)
            response = self.substations[id - 1].source_data.GSpacket(content)
            name = "UE" + str(int(identity, 2))
            frequency = (id - 1) // 2
            slot = self.substations[id - 1].find_position(self.receipt[frequency], 0)
            self.receipt[frequency][slot] = name
            data = "batch:" + str(batch) + "|" + "status:" + status + "|" + str(frequency) + "|" + str(
                slot) + "|" + response + '\n'
            self.write_response_data(name, data)
        f.close()

    # 定义一个增加子站函数
    def add_substation(self, substation):
        self.substations.append(substation)

    def get_list(self, slot, frequency):
        n = []
        for i in range(slot):
            fre = []
            for j in range(frequency):
                fre.append(j)
            n.append(fre)
        return n
