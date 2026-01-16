import cv2
import numpy as np
import time

time_start = time.time()

img_name = "5.jpg"
img_path = "photos/" + img_name

# img_name = "restored_4.jpg"
# img_path = "restored_images/" + img_name

img = cv2.imread(img_path)
img_original = img.copy()

# 调整整体亮度到目标均值

# mu_target = 128.0
# med = np.median(img)
# alpha = mu_target / med
# img = np.clip(img * alpha, 0, 255).astype(np.uint8)

# img_dn = cv2.fastNlMeansDenoisingColored(average_light, None,
#                                         h=3, hColor=3,
#                                         templateWindowSize=5,
#                                         searchWindowSize=21)


# 按亮度分布自适应重映射
# CLAHE 局部自适应均衡

# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# l, a, b = cv2.split(lab)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
# l2 = clahe.apply(l)

# lab2 = cv2.merge([l2,a,b])
# img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

time_end = time.time()
print("Clahe time: {:.3f} s".format(time_end - time_start))

# 增强饱和度

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5, dtype=cv2.CV_8U)



#hsv2 = cv2.merge([h,s,v])
#img = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

time_end = time.time()
print("time: {:.3f} s".format(time_end - time_start))

# 再一次调整整体亮度到目标均值

mu_target = 180.0
#med = np.median(img)
med = cv2.mean(hsv[:,:,2])[0]
alpha = mu_target / med
img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)


time_end = time.time()
print("Initial light preproc time: {:.3f} s".format(time_end - time_start))

def gray_world_wb(img_bgr): # 灰度世界白平衡
    img = img_bgr.astype(np.float32)
    b,g,r = cv2.split(img)
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    m = (mb + mg + mr) / 3.0
    b *= (m / (mb + 1e-6))
    g *= (m / (mg + 1e-6))
    r *= (m / (mr + 1e-6))
    out = cv2.merge([b,g,r])
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_on_v(img_hsv, clip=2.0, grid=(8,8)): # 对 V 通道做 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    img_hsv[:,:,2] = clahe.apply(img_hsv[:,:,2])
    return img_hsv

def get_red_blue_mask(img_hsv): # 分割红色和蓝色区域

    # 自适应 V 下限：取 V 的某个分位数作为基础，再给个下限保护
    #v_p20 = np.percentile(img_hsv[:,:,2], 20)      # 你也可以试 10 或 30
    hist = cv2.calcHist([img_hsv[:,:,2]], [0], None, [256], [0,256])
    cdf = hist.cumsum()
    v_p20 = np.searchsorted(cdf, 0.2 * cdf[-1])


    v_min = int(max(20, v_p20 * 0.6)) # 暗光时别卡太高，亮时自动提高一点

    s_min = 80  # 纯色物块建议 70~120 之间调；越高越抗误检但可能漏暗处边缘

    # 蓝色 H 范围（OpenCV H: 0~179）
    lower_blue = np.array([95,  s_min, v_min])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # 红色两段
    lower_red1 = np.array([0,   s_min, v_min])
    upper_red1 = np.array([10,  255,   255])
    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([179, 255,   255])
    mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) | cv2.inRange(img_hsv, lower_red2, upper_red2)

    mask = mask_blue | mask_red

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def pop_color(img_hsv, mask, s_gain=1.35, v_gain=1.05): # 提升mask区域的饱和度和亮度
    m = (mask > 0)

    img_hsv[:,:,1][m] = np.clip(img_hsv[:,:,1][m] * s_gain, 0, 255)
    img_hsv[:,:,2][m] = np.clip(img_hsv[:,:,2][m] * v_gain, 0, 255)

    return img_hsv

# img = gray_world_wb(img)                 # 稳白平衡

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv2 = clahe_on_v(hsv, clip=2.0)         # 稳暗光

time_end = time.time()
print("time3: {:.3f} s".format(time_end - time_start))

mask = get_red_blue_mask(hsv2)            # 分割红/蓝

time_end = time.time()
print("time4: {:.3f} s".format(time_end - time_start))

#canny = cv2.Canny(mask, 100, 200)
#canny_dilated = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=5) # 膨胀边缘，覆盖更宽区域
hsv  = pop_color(hsv, mask, 1.4, 1.0) 
img  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#img += cv2.cvtColor(canny_dilated, cv2.COLOR_GRAY2BGR) * np.array([255,0,255], dtype=np.uint8)

time_end = time.time()
print("Light preproc time: {:.3f} s".format(time_end - time_start))

cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
cv2.imshow("Canny", mask)
cv2.moveWindow("Canny", 20, 600)
cv2.namedWindow("Enhanced", cv2.WINDOW_NORMAL)
cv2.imshow("Enhanced", img)
cv2.moveWindow("Enhanced", 600, 20)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()