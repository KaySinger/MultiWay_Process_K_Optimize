# 定义P1参与后续反应微分方程
def equations_process1(p, t, k, k_inv):
    k = k[:40]
    k_inv = k_inv[:39]
    dpdt = [0] * 41
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1] ** 2 + k_inv[0] * p[2]
    for i in range(1, 39):
        dpdt[1] += k_inv[i] * p[i + 2] - k[i + 1] * p[1] * p[i + 1]
    dpdt[2] = k[1] * p[1] ** 2 + k_inv[1] * p[3] - k_inv[0] * p[2] - k[2] * p[1] * p[2]
    for i in range(3, 40):
        dpdt[i] = 2 * k[i - 1] * p[1] * p[i - 1] + k_inv[i - 1] * p[i + 1] - 2 * k_inv[i - 2] * p[i] - k[i] * p[1] * p[i]
    dpdt[40] = 2 * k[39] * p[1] * p[39] - 2 * k_inv[38] * p[40]
    return dpdt

# 定义P2参与后续反应微分方程
def equations_process2(p, t, k, k_inv):
    k = k[40:77]
    k_inv = k_inv[39:76]
    dpdt = [0] * 41
    dpdt[2] = k_inv[0] * p[4] - k[0] * p[2]**2
    for i in range(1, 37):
        dpdt[2] += k_inv[i] * p[i+4] - k[i] * p[2] * p[i+2]
    dpdt[3] = k_inv[1] * p[5] - k[1] * p[2] * p[3]
    dpdt[4] = k[0] * p[2]**2 - k_inv[0] * p[4] + k_inv[2] * p[6] - k[2] * p[2] * p[4]
    for i in range(5, 39):
        dpdt[i] = 2 * k[i-4] * p[2] * p[i-2] + k_inv[i-2] * p[i+2] - 2 * k_inv[i-4] * p[i] - k[i-2] * p[2] * p[i]
    dpdt[39] = 2 * k[35] * p[2] * p[37] - 2 * k_inv[35] * p[39]
    dpdt[40] = 2 * k[36] * p[2] * p[38] - 2 * k_inv[36] * p[40]
    return dpdt

# 定义P3参与后续反应微分方程
def equations_process3(p, t, k, k_inv):
    k = k[77:112]
    k_inv = k_inv[76:111]
    dpdt = [0] * 41
    dpdt[3] = k_inv[0] * p[6] - k[0] * p[3]**2
    for i in range(1, 35):
        dpdt[3] += k_inv[i] * p[i+6] - k[i] * p[3] * p[i+3]
    dpdt[4] = k_inv[1] * p[7] - k[1] * p[3] * p[4]
    dpdt[5] = k_inv[2] * p[8] - k[2] * p[3] * p[5]
    dpdt[6] = k[0] * p[3]**2 + k_inv[3] * p[9] - k_inv[0] * p[6] - k[3] * p[3] * p[6]
    for i in range(7, 38):
        dpdt[i] = 2 * k[i-6] * p[3] * p[i-3] + k_inv[i-3] * p[i+3] - 2 * k_inv[i-6] * p[i] - k[i-3] * p[3] * p[i]
    dpdt[38] = 2 * k[32] * p[3] * p[35] - 2 * k_inv[32] * p[38]
    dpdt[39] = 2 * k[33] * p[3] * p[36] - 2 * k_inv[33] * p[39]
    dpdt[40] = 2 * k[34] * p[3] * p[37] - 2 * k_inv[34] * p[40]
    return dpdt

# 定义P4参与后续反应微分方程
def equations_process4(p, t, k, k_inv):
    k = k[112:145]
    k_inv = k_inv[111:144]
    dpdt = [0] * 41
    dpdt[4] = - k[0] * p[4]**2 + k_inv[0] * p[8]
    for i in range(1, 33):
        dpdt[4] += k_inv[i] * p[i+8] - k[i] * p[4] * p[i+4]
    dpdt[5] = k_inv[1] * p[9] - k[1] * p[4] * p[5]
    dpdt[6] = k_inv[2] * p[10] - k[2] * p[4] * p[6]
    dpdt[7] = k_inv[3] * p[11] - k[3] * p[4] * p[7]
    dpdt[8] = k[0] * p[4]**2 + k_inv[4] * p[12] - k_inv[0] * p[8] + k[4] * p[4] * p[8]
    for i in range(9, 37):
        dpdt[i] = 2 * k[i-8] * p[4] * p[i-4] + k_inv[i-4] * p[i+4] - 2 * k_inv[i-8] * p[i] - k[i-4] * p[4] * p[i]
    dpdt[37] = 2 * k[29] * p[4] * p[33] - 2 * k_inv[29] * p[37]
    dpdt[38] = 2 * k[30] * p[4] * p[34] - 2 * k_inv[30] * p[38]
    dpdt[39] = 2 * k[31] * p[4] * p[35] - 2 * k_inv[31] * p[39]
    dpdt[40] = 2 * k[32] * p[4] * p[36] - 2 * k_inv[32] * p[40]
    return dpdt

# 定义P5参与后续反应微分方程
def equations_process5(p, t, k, k_inv):
    k = k[145:176]
    k_inv = k_inv[144:175]
    dpdt = [0] * 41
    dpdt[5] = -k[0] * p[5]**2 + k_inv[0] * p[10]
    for i in range(1, 31):
        dpdt[5] += k_inv[i] * p[i+10] - k[i] * p[5] * p[i+5]
    dpdt[6] = k_inv[1] * p[11] - k[1] * p[5] * p[6]
    dpdt[7] = k_inv[2] * p[12] - k[2] * p[5] * p[7]
    dpdt[8] = k_inv[3] * p[13] - k[3] * p[5] * p[8]
    dpdt[9] = k_inv[4] * p[14] - k[4] * p[5] * p[9]
    dpdt[10] = k[0] * p[5]**2 + k_inv[5] * p[15] - k_inv[0] * p[10] + k[5] * p[5] * p[10]
    for i in range(11, 36):
        dpdt[i] = 2 * k[i-10] * p[5] * p[i-5] + k_inv[i-5] * p[i+5] - 2 * k_inv[i-10] * p[i] - k[i-5] * p[5] * p[i]
    dpdt[36] = 2 * k[26] * p[5] * p[31] - 2 * k_inv[26] * p[36]
    dpdt[37] = 2 * k[27] * p[5] * p[32] - 2 * k_inv[27] * p[37]
    dpdt[38] = 2 * k[28] * p[5] * p[33] - 2 * k_inv[28] * p[38]
    dpdt[39] = 2 * k[29] * p[5] * p[34] - 2 * k_inv[29] * p[39]
    dpdt[40] = 2 * k[30] * p[5] * p[35] - 2 * k_inv[30] * p[40]
    return dpdt

# 定义P6参与后续反应微分方程
def equations_process6(p, t, k, k_inv):
    k = k[176:205]
    k_inv = k_inv[175:204]
    dpdt = [0] * 41
    dpdt[6] = -k[0] * p[6]**2 + k_inv[0] * p[12]
    for i in range(1, 29):
        dpdt[6] += k_inv[i] * p[i+12] - k[i] * p[6] * p[i+6]
    dpdt[7] = k_inv[1] * p[13] - k[1] * p[6] * p[7]
    dpdt[8] = k_inv[2] * p[14] - k[2] * p[6] * p[8]
    dpdt[9] = k_inv[3] * p[15] - k[3] * p[6] * p[9]
    dpdt[10] = k_inv[4] * p[16] - k[4] * p[6] * p[10]
    dpdt[11] = k_inv[5] * p[17] - k[5] * p[6] * p[11]
    dpdt[12] = k[0] * p[6]**2 + k_inv[6] * p[18] - k_inv[0] * p[12] + k[6] * p[6] * p[12]
    for i in range(13, 35):
        dpdt[i] = 2 * k[i-12] * p[6] * p[i-6] + k_inv[i-6] * p[i+6] - 2 * k_inv[i-12] * p[i] - k[i-6] * p[6] * p[i]
    dpdt[35] = 2 * k[23] * p[6] * p[29] - 2 * k_inv[23] * p[35]
    dpdt[36] = 2 * k[24] * p[6] * p[30] - 2 * k_inv[24] * p[36]
    dpdt[37] = 2 * k[25] * p[6] * p[31] - 2 * k_inv[25] * p[37]
    dpdt[38] = 2 * k[26] * p[6] * p[32] - 2 * k_inv[26] * p[38]
    dpdt[39] = 2 * k[27] * p[6] * p[33] - 2 * k_inv[27] * p[39]
    dpdt[40] = 2 * k[28] * p[6] * p[34] - 2 * k_inv[28] * p[40]
    return dpdt

# 定义P7参与后续反应微分方程
def equations_process7(p, t, k, k_inv):
    k = k[205:232]
    k_inv = k_inv[204:231]
    dpdt = [0] * 41
    dpdt[7] = -k[0] * p[7]**2 + k_inv[0] * p[14]
    for i in range(1, 27):
        dpdt[7] += k_inv[i] * p[i+14] - k[i] * p[7] * p[i+7]
    dpdt[8] = k_inv[1] * p[15] - k[1] * p[7] * p[8]
    dpdt[9] = k_inv[2] * p[16] - k[2] * p[7] * p[9]
    dpdt[10] = k_inv[3] * p[17] - k[3] * p[7] * p[10]
    dpdt[11] = k_inv[4] * p[18] - k[4] * p[7] * p[11]
    dpdt[12] = k_inv[5] * p[19] - k[5] * p[7] * p[12]
    dpdt[13] = k_inv[6] * p[20] - k[6] * p[7] * p[13]
    dpdt[14] = k[0] * p[7]**2 + k_inv[7] * p[21] - k_inv[0] * p[14] - k[7] * p[7] * p[14]
    for i in range(15, 34):
        dpdt[i] = 2 * k[i-14] * p[7] * p[i-7] + k_inv[i-7] * p[i+7] - 2 * k_inv[i-14] * p[i] - k[i-7] * p[7] * p[i]
    dpdt[34] = 2 * k[20] * p[7] * p[27] - 2 * k_inv[20] * p[34]
    dpdt[35] = 2 * k[21] * p[7] * p[28] - 2 * k_inv[21] * p[35]
    dpdt[36] = 2 * k[22] * p[7] * p[29] - 2 * k_inv[22] * p[36]
    dpdt[37] = 2 * k[23] * p[7] * p[30] - 2 * k_inv[23] * p[37]
    dpdt[38] = 2 * k[24] * p[7] * p[31] - 2 * k_inv[24] * p[38]
    dpdt[39] = 2 * k[25] * p[7] * p[32] - 2 * k_inv[25] * p[39]
    dpdt[40] = 2 * k[26] * p[7] * p[33] - 2 * k_inv[26] * p[40]
    return dpdt

# 定义P8参与后续反应微分方程
def equations_process8(p, t, k, k_inv):
    k = k[232:257]
    k_inv = k_inv[231:256]
    dpdt = [0] * 41
    dpdt[8] = -k[0] * p[8]**2 + k_inv[0] * p[16]
    for i in range(1, 25):
        dpdt[8] += k_inv[i] * p[i+16] - k[i] * p[8] * p[i+8]
    for i in range(9, 16):
        dpdt[i] = k_inv[i-8] * p[i+8] - k[i-8] * p[8] * p[i]
    dpdt[16] = k[0] * p[8]**2 + k_inv[8] * p[24] - k_inv[0] * p[16] - k[8] * p[8] * p[16]
    for i in range(17, 33):
        dpdt[i] = 2 * k[i-16] * p[8] * p[i-8] + k_inv[i-8] * p[i+8] - 2 * k_inv[i-16] * p[i] - k[i-8] * p[8] * p[i]
    for i in range(33, 41):
        dpdt[i] = 2 * k[i-16] * p[8] * p[i-8] - 2 * k_inv[i-16] * p[i]
    return dpdt

# 定义P9参与后续反应微分方程
def equations_process9(p, t, k, k_inv):
    k = k[257:280]
    k_inv = k_inv[256:279]
    dpdt = [0] * 41
    dpdt[9] = -k[0] * p[9]**2 + k_inv[0] * p[18]
    for i in range(1, 23):
        dpdt[9] += k_inv[i] * p[i+18] - k[i] * p[9] * p[i+9]
    for i in range(10, 18):
        dpdt[i] = k_inv[i-9] * p[i+9] - k[i-9] * p[9] * p[i]
    dpdt[18] = k[0] * p[9]**2 + k_inv[9] * p[27] - k_inv[0] * p[18] - k[9] * p[9] * p[18]
    for i in range(19, 32):
        dpdt[i] = 2 * k[i-18] * p[9] * p[i-9] + k_inv[i-9] * p[i+9] - 2 * k_inv[i-18] * p[i] - k[i-9] * p[9] * p[i]
    for i in range(32, 41):
        dpdt[i] = 2 * k[i-18] * p[9] * p[i-9] - 2 * k_inv[i-18] * p[i]
    return dpdt

# 定义P10参与后续反应微分方程
def equations_process10(p, t, k, k_inv):
    k = k[280:301]
    k_inv = k_inv[279:300]
    dpdt = [0] * 41
    dpdt[10] = -k[0] * p[10]**2 + k_inv[0] * p[20]
    for i in range(1, 21):
        dpdt[10] += k_inv[i] * p[i+20] - k[i] * p[10] * p[i+10]
    for i in range(11, 20):
        dpdt[i] = k_inv[i-10] * p[i+10] - k[i-10] * p[10] * p[i]
    dpdt[20] = k[0] * p[10]**2 + k_inv[10] * p[30] - k_inv[0] * p[20] - k[10] * p[10] * p[20]
    for i in range(21, 31):
        dpdt[i] = 2 * k[i-20] * p[10] * p[i-10] + k_inv[i-10] * p[i+10] - 2 * k_inv[i-20] * p[i] - k[i-10] * p[10] * p[i]
    for i in range(31, 41):
        dpdt[i] = 2 * k[i-20] * p[10] * p[i-10] - 2 * k_inv[i-20] * p[i]
    return dpdt

# 定义P11参与后续反应微分方程
def equations_process11(p, t, k, k_inv):
    k = k[301:320]
    k_inv = k_inv[300:319]
    dpdt = [0] * 41
    dpdt[11] = -k[0] * p[11]**2 + k_inv[0] * p[22]
    for i in range(1, 19):
        dpdt[11] += k_inv[i] * p[i+22] - k[i] * p[11] * p[i+11]
    for i in range(12, 22):
        dpdt[i] = k_inv[i-11] * p[i+11] - k[i-11] * p[11] * p[i]
    dpdt[22] = k[0] * p[11]**2 + k_inv[11] * p[33] - k_inv[0] * p[22] - k[11] * p[11] * p[22]
    for i in range(23, 30):
        dpdt[i] = 2 * k[i-22] * p[11] * p[i-11] + k_inv[i-11] * p[i+11] - 2 * k_inv[i-22] * p[i] - k[i-11] * p[11] * p[i]
    for i in range(30, 41):
        dpdt[i] = 2 * k[i-22] * p[11] * p[i-11] - 2 * k_inv[i-22] * p[i]
    return dpdt

# 定义P12参与后续反应微分方程
def equations_process12(p, t, k, k_inv):
    k = k[320:337]
    k_inv = k_inv[319:336]
    dpdt = [0] * 41
    dpdt[12] = -k[0] * p[12]**2 + k_inv[0] * p[24]
    for i in range(1, 17):
        dpdt[12] += k_inv[i] * p[i+24] - k[i] * p[12] * p[i+12]
    for i in range(13, 24):
        dpdt[i] = k_inv[i-12] * p[i+12] - k[i-12] * p[12] * p[i]
    dpdt[24] = k[0] * p[12]**2 + k_inv[12] * p[36] - k_inv[0] * p[24] - k[12] * p[12] * p[24]
    for i in range(25, 29):
        dpdt[i] = 2 * k[i-24] * p[12] * p[i-12] + k_inv[i-12] * p[i+12] - 2 * k_inv[i-24] * p[i] - k[i-12] * p[12] * p[i]
    for i in range(29, 41):
        dpdt[i] = 2 * k[i-24] * p[12] * p[i-12] - 2 * k_inv[i-24] * p[i]
    return dpdt

# 定义P13参与后续反应微分方程
def equations_process13(p, t, k, k_inv):
    k = k[337:352]
    k_inv = k_inv[336:351]
    dpdt = [0] * 41
    dpdt[13] = -k[0] * p[13]**2 + k_inv[0] * p[26]
    for i in range(1, 15):
        dpdt[13] += k_inv[i] * p[i+26] - k[i] * p[13] * p[i+13]
    for i in range(14, 26):
        dpdt[i] = k_inv[i-13] * p[i+13] - k[i-13] * p[13] * p[i]
    dpdt[26] = k[0] * p[13]**2 + k_inv[13] * p[39] - k_inv[0] * p[26] - k[13] * p[13] * p[26]
    dpdt[27] = 2 * k[1] * p[13] * p[14] + k_inv[14] * p[40] - 2 * k_inv[1] * p[27] - k[14] * p[13] * p[27]
    for i in range(28, 41):
        dpdt[i] = 2 * k[i-26] * p[13] * p[i-13] - 2 * k_inv[i-26] * p[i]
    return dpdt

# 定义P14参与后续反应微分方程
def equations_process14(p, t, k, k_inv):
    k = k[352:365]
    k_inv = k_inv[351:364]
    dpdt = [0] * 41
    dpdt[14] = -k[0] * p[14]**2 + k_inv[0] * p[28]
    for i in range(1, 13):
        dpdt[14] += k_inv[i] * p[i+28] - k[i] * p[14] * p[i+14]
    for i in range(15, 27):
        dpdt[i] = k_inv[i-14] * p[i+14] - k[i-14] * p[14] * p[i]
    dpdt[28] = k[0] * p[14]**2 - k_inv[0] * p[28]
    for i in range(29, 41):
        dpdt[i] = 2 * k[i-28] * p[14] * p[i-14] - 2 * k_inv[i-28] * p[i]
    return dpdt

# 定义P15参与后续反应微分方程
def equations_process15(p, t, k, k_inv):
    k = k[365:376]
    k_inv = k_inv[364:375]
    dpdt = [0] * 41
    dpdt[15] = -k[0] * p[15]**2 + k_inv[0] * p[30]
    for i in range(1, 11):
        dpdt[15] += k_inv[i] * p[i+30] - k[i] * p[15] * p[i+15]
    for i in range(16, 26):
        dpdt[i] = k_inv[i-15] * p[i+15] - k[i-15] * p[15] * p[i]
    dpdt[30] = k[0] * p[15]**2 - k_inv[0] * p[30]
    for i in range(31, 41):
        dpdt[i] = 2 * k[i-30] * p[15] * p[i-15] - 2 * k_inv[i-30] * p[i]
    return dpdt

# 定义P16参与后续反应微分方程
def equations_process16(p, t, k, k_inv):
    k = k[376:385]
    k_inv = k_inv[375:384]
    dpdt = [0] * 41
    dpdt[16] = -k[0] * p[16]**2 + k_inv[0] * p[32]
    for i in range(1, 9):
        dpdt[16] += k_inv[i] * p[i+32] - k[i] * p[16] * p[i+16]
    for i in range(17, 25):
        dpdt[i] = k_inv[i-16] * p[i+16] - k[i-16] * p[16] * p[i]
    dpdt[32] = k[0] * p[16]**2 - k_inv[0] * p[32]
    for i in range(33, 41):
        dpdt[i] = 2 * k[i-32] * p[16] * p[i-16] - 2 * k_inv[i-32] * p[i]
    return dpdt

# 定义P17参与后续反应微分方程
def equations_process17(p, t, k, k_inv):
    k = k[385:392]
    k_inv = k_inv[384:391]
    dpdt = [0] * 41
    dpdt[17] = -k[0] * p[17]**2 + k_inv[0] * p[34]
    for i in range(1, 7):
        dpdt[17] += k_inv[i] * p[i+34] - k[i] * p[17] * p[i+17]
    for i in range(18, 24):
        dpdt[i] = k_inv[i-17] * p[i+17] - k[i-17] * p[17] * p[i]
    dpdt[34] = k[0] * p[17]**2 - k_inv[0] * p[34]
    for i in range(35, 41):
        dpdt[i] = 2 * k[i-34] * p[17] * p[i-17] - 2 * k_inv[i-34] * p[i]
    return dpdt

# 定义P18参与后续反应微分方程
def equations_process18(p, t, k, k_inv):
    k = k[392:397]
    k_inv = k_inv[391:396]
    dpdt = [0] * 41
    dpdt[18] = -k[0] * p[18]**2 + k_inv[0] * p[36]
    for i in range(1, 5):
        dpdt[18] += k_inv[i] * p[i+36] - k[i] * p[18] * p[i+18]
    for i in range(19, 23):
        dpdt[i] = k_inv[i-18] * p[i+18] - k[i-18] * p[18] * p[i]
    dpdt[36] = k[0] * p[18]**2 - k_inv[0] * p[36]
    for i in range(37, 41):
        dpdt[i] = 2 * k[i-36] * p[18] * p[i-18] - 2 * k_inv[i-36] * p[i]
    return dpdt

# 定义P19参与后续反应微分方程
def equations_process19(p, t, k, k_inv):
    k = k[397:400]
    k_inv = k_inv[396:399]
    dpdt = [0] * 41
    dpdt[19] = -k[0] * p[19]**2 + k_inv[0] * p[38]
    for i in range(1, 3):
        dpdt[18] += k_inv[i] * p[i+38] - k[i] * p[19] * p[i+19]
    dpdt[20] = k_inv[1] * p[39] - k[1] * p[19] * p[20]
    dpdt[21] = k_inv[2] * p[40] - k[2] * p[19] * p[21]
    dpdt[38] = k[0] * p[19]**2 - k_inv[0] * p[38]
    dpdt[39] = 2 * k[1] * p[19] * p[20] - k_inv[1] * p[39]
    dpdt[40] = 2 * k[2] * p[19] * p[21] - k_inv[2] * p[40]
    return dpdt

# 定义P20参与后续反应微分方程
def equations_process20(p, t, k, k_inv):
    k = k[400:401]
    k_inv = k_inv[399:400]
    dpdt = [0] * 41
    dpdt[20] = -k[0] * p[20]**2 + k_inv[0] * p[40]
    dpdt[40] = k[0] * p[20]**2 - k_inv[0] * p[40]
    return dpdt