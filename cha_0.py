from itertools import groupby
import cv2

def getHProjection(image):  # 水平投影
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    a = []
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
        if h_[y] > 10:
            a.append(y)

    fun = lambda x: x[1] - x[0]
    z = []
    for k, g in groupby(enumerate(a), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:

            num = max(l1) - min(l1)
        else:

            num = l1[0] - l1[0]
        if num < 10:
            continue
        else:
            tu = (min(l1), max(l1))
            z.append(tu)
    image = image[z[0][0]:z[-1][1]]

    h_ = h_[z[0][0]:(z[-1][1] + 1)]

    return h_, image, z


def getVProjection(image,thre,i):  # 垂直投影

    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    a = []
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
        if w_[x] > thre[i][0]:
            a.append(x)
    fun = lambda x: x[1] - x[0]
    z = []
    for k, g in groupby(enumerate(a), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:
            num = max(l1) - min(l1)

        else:
            num = l1[0] - l1[0]
        if num < thre[i][1]:
            continue
        else:
            tu = (min(l1), max(l1))
            z.append(tu)
    w_ = w_[z[0][0]:(z[-1][1] + 1)]
    return w_, z

def getVpixel(image):  # 垂直投影

    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    a = []
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # print(w_)
    if len(w_) > 25:
        w_new = w_[3:(len(w_)-3)]
        # print("w_", w_)
        # print("w_new", w_new)
        max_value = max(w_new)
        min_value = min(w_new)
        min_idx = w_new.index(min_value)
 
        if min_value*3 <= max_value and abs(min_idx-len(w_new)/2) < 3 and min_value < 8:
            a.append(min_value)
    else:
        a = []
    return w_, a



def detetct_cha(image):
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    b = []
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
        if h_[y] > 1:
            b.append(y)
    line_num = len(b)
    return line_num

def dilate_erode(binary):
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    closing = cv2.dilate(binary, ker, iterations=1)  # 进行腐蚀操作
    return closing

def dilate_erode1(binary):
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    closing = cv2.dilate(binary, ker, iterations=1)  # 进行腐蚀操作
    return closing



# 改为直接传递二值化后的图像，且进行开闭运算后
def cut_0(img):

    pos_dict = {}
    H, ima, z = getHProjection(img)  # 水平投影
    (h, w) = ima.shape
    Position = []
    H_Start = []
    H_End = []
    Height = []
    p = []
    for i in range(len(z)):
        H_Start.append(z[i][0])
        H_End.append(z[i][1])
        Height.append((z[i][1] - z[i][0]))

    thre = [(3,7),(2,6),(3,6)]     # [(3,7),(4,7),(6,6)]
    # for i in range(len(H_Start)):
    for i in range(3):
        posi = []

        cropImg = img[H_Start[i]:H_End[i], 0:w]
        if i == 0 and i == 1:
            cropImg = dilate_erode(cropImg)
        W, z1 = getVProjection(cropImg,thre,i)
        # Wend = 0
        W_Start = 0
        W_End = 0
        st = []
        en = []
        po = []
        width = []
        # a = 0
        for j in range(len(z1)):
            st.append(z1[j][0])
            en.append(z1[j][1])
            width.append((z1[j][1] - z1[j][0]))

        for j in range(len(st)):
            if i == 0:
                sub_img = cropImg[0:cropImg.shape[0], st[j]:en[j]]
                line_num = detetct_cha(sub_img)
                if line_num > 20:#20
                    if Height[i] / width[j] <= 1.3:
                        Position.append([st[j], H_Start[i], int(en[j]-width[j]/2), H_End[i]])
                        Position.append([int(en[j]-width[j]/2+1), H_Start[i], en[j], H_End[i]])
                        posi.append([st[j], H_Start[i], int(en[j]-width[j]/2), H_End[i]])
                        posi.append([int(en[j]-width[j]/2+1), H_Start[i], en[j], H_End[i]])
                    elif Height[i] / width[j] < 4.2:
                        po.append([W_Start, W_End])
                        Position.append([st[j], H_Start[i], en[j], H_End[i]])
                        posi.append([st[j], H_Start[i], en[j], H_End[i]])

            elif i == 1:
                sub_img = cropImg[0:cropImg.shape[0], st[j]:en[j]]
                line_num = detetct_cha(sub_img)
                if line_num >= 0.5*cropImg.shape[0]:
                    if Height[i] / width[j] < 0.9 and j < 7:
                        Position.append([st[j], H_Start[i], int(en[j] - width[j] / 2), H_End[i]])
                        Position.append([int(en[j] - width[j] / 2 + 1), H_Start[i], en[j], H_End[i]])

                        posi.append([st[j], H_Start[i], int(en[j]-width[j]/2), H_End[i]])
                        posi.append([int(en[j]-width[j]/2+1), H_Start[i], en[j], H_End[i]])
                    else:
                        po.append([W_Start, W_End])
                        Position.append([st[j], H_Start[i], en[j], H_End[i]])
                        posi.append([st[j], H_Start[i], en[j], H_End[i]])

            elif i == 2:
                sub_img = cropImg[0:cropImg.shape[0], st[j]:en[j]]

                line_num = detetct_cha(sub_img)
                Vpixel, sign = getVpixel(sub_img)

                if line_num > 0.4*cropImg.shape[0]:
                    if len(sign) != 0:
                        Position.append([st[j], H_Start[i], int(en[j] - width[j] / 2), H_End[i]])
                        Position.append([int(en[j] - width[j] / 2 + 1), H_Start[i], en[j], H_End[i]])
                        posi.append([st[j], H_Start[i], int(en[j] - width[j] / 2), H_End[i]])
                        posi.append([int(en[j] - width[j] / 2 + 1), H_Start[i], en[j], H_End[i]])

                    else:
                        po.append([W_Start, W_End])
                        Position.append([st[j], H_Start[i], en[j], H_End[i]])
                        posi.append([st[j], H_Start[i], en[j], H_End[i]])


        pos_dict[i] = posi
  
    return Position, pos_dict


