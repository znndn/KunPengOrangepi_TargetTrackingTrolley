import struct

# 注意画幅已经被强制
# 注意最大只支持640*480的画幅，如果要修改的更大需要修改位数640*480 = 307200
def SendDataToStm32(x_offset, size,ser):
    x_offset = max(-320, min(320, int(x_offset)))
    size = max(0, min(307200, int(size)))
    packet = struct.pack('<BBhIB', 0xB3, 0x01, x_offset, size, 0x5B)
    # 小端，永远显式地加上 < 或 >
    ser.write(packet)