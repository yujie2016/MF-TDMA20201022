import os
import random
import time


class Layer2(object):
    # 数据链路层的原始数据函数，定义传输速率speed 默认128 传输时间 time 默认1s
    # 定义生成数据函数，输出为二进制比特流
    # 补充协议相关数据，突发类型，6种
    # 返回一个数据列表
    def __init__(self, speed=128, time=1):
        self.speed = speed
        self.time = time
        self.data = []
        self.source_data()

    def source_data(self):
        # 计算需要产生多少个原始二进制数据
        size = self.speed * self.time
        # 计算每个二进制数据时间间隔
        betimes = 1 / self.speed
        i = 0
        while i < size:
            bit_data = random.randint(0, 100) % 2
            self.data.append(bit_data)
            i += 1
            # time.sleep(betimes)

    # 定义生成随机二进制的函数
    def gen_data(self, size):
        i = 0
        bits_data = ""
        while i < size:
            bits_data += str(random.randint(0, 100) % 2)
            i += 1
        return bits_data

    # 生成登陆FPDF函数
    def logon_FPDU(self, id):
        header = ""
        adaptation = ""
        payload = ""
        # 1.包头，4字节
        # 定义同步字节，8bit，取值位0x47
        sync_byte = "01000111"
        # 定义错误指示信息，1bit
        transport_error_indicator = '0'
        # 负载单元开始标志，1bit
        payload_unit_start_indicator = '0'
        # 传输优先级标志，1bit
        transport_priority = random.randint(0, 1)
        transport_priority = bin(transport_priority)[2:].zfill(1)
        # PID，13bit
        PID = random.randint(0, 8191)
        PID = bin(PID)[2:].zfill(13)
        # 加密标志，2bit，00表示未加密
        transport_scrambling_control = random.randint(1, 3)
        transport_scrambling_control = bin(transport_scrambling_control)[2:].zfill(2)
        # 附加区域控制，2bit。01-无附加区域，只有payload；10-只有附加区域，无payload；11-有附加区域和payload；00-保留
        adaptation_field_control = "11"
        # 包递增计数器，4bit
        continuity_counter = random.randint(0, 15)
        continuity_counter = bin(continuity_counter)[2:].zfill(4)

        header = sync_byte + transport_error_indicator + payload_unit_start_indicator + transport_priority + PID + \
                 transport_scrambling_control + adaptation_field_control + continuity_counter
        # print("header: " + str(len(header)/8))

        # 2.负载字段
        # 定义接入类型，4bit，取值位0 - 6
        identity = bin(id)[2:].zfill(8)
        entry_type = random.randint(0, 6)
        entry_type = bin(entry_type)[2:].zfill(4)
        # 定义接入状态，4bit中只有一个值位1，每个bit代表不同的含义
        # status_position = random.randint(0, 3)
        #         # access_status = ""
        #         # for i in range(4):
        #         #     if i == status_position:
        #         #         access_status += "1"
        #         #     else:
        #         #         access_status += "0"
        access_status = "0010"
        # 接入状态为登陆
        # 定义登陆PDU的具体二进制数据，其中类型为 type-length-value value为随机二进制bit 注：0 ，4，5，6，7，11长度随机生成，满足tlv格式
        content_value = ""
        for i in range(12):
            if i == 0:
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 1:
                # user id，hash算法器为固定长度值sha-1 输出160bits的hash值 输出长度固定 4bit无法表示20字节，取最大值
                type_value = bin(i)[2:].zfill(4)
                length = 15
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 2:
                # signature 相关信息级连后的sha-1的hash值
                type_value = bin(i)[2:].zfill(4)
                length = 15
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 3:
                # low layer，5～6个byte。其余的一个bype由控制位决定
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(5, 6)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 4:
                # high layer
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 5:
                # options requested
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 6:
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 7:
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 8:
                # 定长 EIRP
                type_value = bin(i)[2:].zfill(4)
                length = 2
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 9:
                # 定长 MTU
                type_value = bin(i)[2:].zfill(4)
                length = 3
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 10:
                # 定航 pointing alignment
                type_value = bin(i)[2:].zfill(4)
                length = 3
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 11:
                type_value = bin(i)[2:].zfill(4)
                length = random.randint(0, 15)
                type_length = bin(length)[2:].zfill(4)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content

        payload = identity + access_status + entry_type + content_value
        # print("payload: " + str(len(payload)/8))

        # 3.自适应字段
        # 自适应区域长度,8bit
        adaptation_field_length = int(188 - len(header)/8 - len(payload)/8)
        adaptation_length = (adaptation_field_length - 2) * 8
        adaptation_field_length = bin(adaptation_field_length)[2:].zfill(8)
        # 指示符，8bit，表示后面不包含PCR
        flag = "01000000"
        # 填充字节，TS包长度为188字节，填充内容为0xff
        stuffing_bytes = self.gen_data(adaptation_length)
        adaptation = adaptation_field_length + flag + stuffing_bytes
        # print("TS length: " + str(int(len(header + adaptation + payload))/8))
        return header + adaptation + payload

    # 定义控制二进制数据
    def control_FPDU(self, id):
        header = ""
        adaptation = ""
        payload = ""
        # 1.包头，4字节
        # 定义同步字节，8bit，取值位0x47
        sync_byte = "01000111"
        # 定义错误指示信息，1bit
        transport_error_indicator = '0'
        # 负载单元开始标志，1bit
        payload_unit_start_indicator = '0'
        # 传输优先级标志，1bit
        transport_priority = random.randint(0, 1)
        transport_priority = bin(transport_priority)[2:].zfill(1)
        # PID，13bit
        PID = random.randint(0, 8191)
        PID = bin(PID)[2:].zfill(13)
        # 加密标志，2bit，00表示未加密
        transport_scrambling_control = random.randint(1, 3)
        transport_scrambling_control = bin(transport_scrambling_control)[2:].zfill(2)
        # 附加区域控制，2bit。01-无附加区域，只有payload；10-只有附加区域，无payload；11-有附加区域和payload；00-保留
        adaptation_field_control = "11"
        # 包递增计数器，4bit
        continuity_counter = random.randint(0, 15)
        continuity_counter = bin(continuity_counter)[2:].zfill(4)
        header = sync_byte + transport_error_indicator + payload_unit_start_indicator + transport_priority + PID + \
                 transport_scrambling_control + adaptation_field_control + continuity_counter
        # 2.负载字段
        identity = bin(id)[2:].zfill(8)
        RCST_status = 2
        transmission_content = ""
        if RCST_status < 3:
            transmission_content = self.gen_data(16)
        else:
            transmission_content = self.gen_data(24)
        RCST_status = "0100"
        power_headroom = bin(random.randint(0, 15))[2:].zfill(4)
        content_value = ""
        # 包含两种内容方式，一类是确定类型，一类是用户自定义，当确定类型时，长度字段省略，自定义是1bype定义
        # 在控制字段中把两种格类型的控制突发都生成
        for i in range(13):
            if i == 0:
                type_value = bin(i)[2:].zfill(8)
                length = 1
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 1:
                type_value = bin(i)[2:].zfill(8)
                length = 1
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 2:
                type_value = bin(i)[2:].zfill(8)
                length = 2
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 3:
                # 3是自定义类型，则包含长度字段
                type_value = bin(i)[2:].zfill(8)
                length = random.randint(5, 6)
                type_length = bin(length)[2:].zfill(8)
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_length + type_content
            elif i == 4:
                type_value = bin(i)[2:].zfill(8)
                length = 3
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 5:
                type_value = bin(i)[2:].zfill(8)
                length = 2
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 6:
                type_value = bin(i)[2:].zfill(8)
                length = 2
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 7:
                type_value = bin(i)[2:].zfill(8)
                length = 3
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 8:
                type_value = bin(i)[2:].zfill(8)
                length = 2
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 9:
                type_value = bin(i)[2:].zfill(8)
                length = 4
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 10:
                type_value = bin(i)[2:].zfill(8)
                length = 3
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 11:
                type_value = bin(i)[2:].zfill(8)
                length = 5
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
            elif i == 12:
                type_value = bin(i)[2:].zfill(8)
                length = 4
                type_content = self.gen_data(length * 8)
                content_value += type_value + type_content
        # print(RCST_status + power_headroom + transmission_content + content_value)
        # print("控制： "+str(int(len(RCST_status + power_headroom + transmission_content + content_value)/8)))
        payload = identity + RCST_status + power_headroom + transmission_content + content_value
        # 3.自适应字段
        adaptation_field_length = int(188 - len(header) / 8 - len(payload) / 8)
        adaptation_length = (adaptation_field_length - 2) * 8
        adaptation_field_length = bin(adaptation_field_length)[2:].zfill(8)
        # 指示符，8bit，表示后面不包含PCR
        flag = "01000000"
        # 填充字节，TS包长度为188字节，填充内容为0xff
        stuffing_bytes = self.gen_data(adaptation_length)
        adaptation = adaptation_field_length + flag + stuffing_bytes
        return header + adaptation + payload

    # 定义切割数据的函数
    def cut(self, obj, sec):
        return [obj[i:i + sec] for i in range(0, len(obj), sec)]

    def TSpacket(self, str):
        header = ""
        adaptation = ""
        payload = ""
        # 1.包头，4字节
        # 定义同步字节，8bit，取值位0x47
        sync_byte = "01000111"
        # 定义错误指示信息，1bit
        transport_error_indicator = '0'
        # 负载单元开始标志，1bit
        payload_unit_start_indicator = '0'
        # 传输优先级标志，1bit
        transport_priority = random.randint(0, 1)
        transport_priority = bin(transport_priority)[2:].zfill(1)
        # PID，13bit
        PID = random.randint(0, 8191)
        PID = bin(PID)[2:].zfill(13)
        # 加密标志，2bit，00表示未加密
        transport_scrambling_control = random.randint(1, 3)
        transport_scrambling_control = bin(transport_scrambling_control)[2:].zfill(2)
        # 附加区域控制，2bit。01-无附加区域，只有payload；10-只有附加区域，无payload；11-有附加区域和payload；00-保留
        adaptation_field_control = "11"
        # 包递增计数器，4bit
        continuity_counter = random.randint(0, 15)
        continuity_counter = bin(continuity_counter)[2:].zfill(4)
        header = sync_byte + transport_error_indicator + payload_unit_start_indicator + transport_priority + PID + \
                 transport_scrambling_control + adaptation_field_control + continuity_counter
        payload = str
        adaptation_field_length = int(188 - len(header) / 8 - len(payload) / 8)
        adaptation_length = (adaptation_field_length - 2) * 8
        adaptation_field_length = bin(adaptation_field_length)[2:].zfill(8)
        # 指示符，8bit，表示后面不包含PCR
        flag = "01000000"
        # 填充字节，TS包长度为188字节，填充内容为0xff
        stuffing_bytes = self.gen_data(adaptation_length)
        adaptation = adaptation_field_length + flag + stuffing_bytes
        return header + adaptation + payload

    # 定义数据突发，数据突发为发送的用户数据，数据突发由于物理层MTU限制，因此需要对数据进行分割，并且标明标号顺序，其中tlv格式。
    # max_unit:最大传输单元，data_speed:上层传输单元 单位均为bit，用于用户数据的生成
    def data_FPDU(self, max_unit, data_speed, id):
        max_unit = max_unit
        content = self.gen_data(data_speed)
        # 上取整数确定FPDU数量
        contents = self.cut(content, max_unit)
        if len(contents) == 1:
            # 数组为1 则一个FPDU中可完整存放数据,
            S = "1"
            E = "1"
            length = bin(len(contents[0]))[2:].zfill(11)
            identity = bin(id)[2:].zfill(8)
            RCST_status = "1000"
            frag_id = "1111"
            label_type = bin(random.randint(0, 3))[2:].zfill(2)
            pro_press = "1"
            contents[0] = identity + RCST_status + S + E + length + frag_id + label_type + pro_press + contents[0]
            contents[0] = self.TSpacket(contents[0])
        else:
            for i in range(len(contents)):
                if i == 0:
                    # 切割数据的开始数据
                    S = "1"
                    E = "0"
                    length = bin(len(contents[i]))[2:].zfill(11)
                    identity = bin(id)[2:].zfill(8)
                    RCST_status = "1000"
                    frag_id = bin(i)[2:].zfill(7)
                    CRC = "0"
                    total_length = bin(len(content))[2:].zfill(12)
                    label_type = bin(random.randint(0, 3))[2:].zfill(2)
                    pro_press = "1"
                    contents[0] = identity + RCST_status + S + E + length + frag_id + CRC + total_length + label_type + pro_press + contents[0]
                    contents[0] = self.TSpacket(contents[0])
                elif i == len(contents) - 1:
                    # 切割数据的结尾数据
                    S = "0"
                    E = "1"
                    length = bin(len(contents[i]))[2:].zfill(11)
                    identity = bin(id)[2:].zfill(8)
                    RCST_status = "1000"
                    frag_id = bin(i)[2:].zfill(7)
                    # 低位补0
                    for j in range(8 - len(contents[i])):
                        contents[i] = contents[i] + "0"
                    contents[i] = identity + RCST_status + S + E + length + frag_id + contents[i]
                    contents[i] = self.TSpacket(contents[i])
                else:
                    # 切割数据的中间数据
                    S = "0"
                    E = "0"
                    length = bin(len(contents[i]))[2:].zfill(11)
                    identity = bin(id)[2:].zfill(8)
                    RCST_status = "1000"
                    frag_id = bin(i)[2:].zfill(7)
                    contents[i] = identity + RCST_status + S + E + length + frag_id + contents[i]
                    contents[i] = self.TSpacket(contents[i])
        # print(contents)
        return contents

    def response(self, id, status):
        response_content = ""
        tag = "10111001"
        length = bin(104)[2:].zfill(8)
        prob = random.randint(0,10)
        if prob == 0:
            keep_identifiers_after_logoff = '0'
        else:
            keep_identifiers_after_logoff = '1'
        power_control_mode = bin(random.randint(0, 3))[2:].zfill(2)
        RCST_accsee_status = status
        group_id = id
        logon_id = id
        content = self.gen_data(73) + self.gen_data(random.randint(3, 7) * 8)
        response_content = tag + length + keep_identifiers_after_logoff + power_control_mode + RCST_accsee_status + group_id + logon_id + content
        return response_content

    def GSpacket(self, str):
        S = '1'
        E = '1'
        LT = '00'
        GSElength = bin(len(str))[2:].zfill(12)
        frag_id = bin(random.randint(0, 15))[2:].zfill(8)
        total_length = bin(len(str))[2:].zfill(16)
        protocol_type = self.gen_data(16)
        label = self.gen_data(48)
        return S + E + LT + GSElength + frag_id + total_length + protocol_type + label + str


# layer2 = Layer2()
# layer2.control_FPDU()
# layer2.data_FPDU(8,25)
