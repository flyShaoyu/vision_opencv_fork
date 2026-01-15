import cv2
import numpy as np

img_name = "5.jpg"
img_path = "photos/" + img_name

# img_name = "restored_4.jpg"
# img_path = "restored_images/" + img_name

img = cv2.imread(img_path)
img_original = img.copy()

# 调整整体亮度到目标均值

mu_target = 128.0
med = np.median(img)
alpha = mu_target / med
img = np.clip(img * alpha, 0, 255).astype(np.uint8)

# img_dn = cv2.fastNlMeansDenoisingColored(average_light, None,
#                                         h=3, hColor=3,
#                                         templateWindowSize=5,
#                                         searchWindowSize=21)


# 按亮度分布自适应重映射
# CLAHE 局部自适应均衡

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
l2 = clahe.apply(l)

lab2 = cv2.merge([l2,a,b])
img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# 增强饱和度

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

s = np.clip(s * 1.5, 0, 255).astype(np.uint8)

hsv2 = cv2.merge([h,s,v])
img = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

# 再一次调整整体亮度到目标均值

mu_target = 180.0
med = np.median(img)
alpha = mu_target / med
img = np.clip(img * alpha, 0, 255).astype(np.uint8)


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

def clahe_on_v(img_bgr, clip=2.0, grid=(8,8)): # 对 V 通道做 CLAHE
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h,s,v2])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def get_red_blue_mask(img_bgr): # 分割红色和蓝色区域
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    # 自适应 V 下限：取 V 的某个分位数作为基础，再给个下限保护
    v_p20 = np.percentile(V, 20)      # 你也可以试 10 或 30
    v_min = int(max(20, v_p20 * 0.6)) # 暗光时别卡太高，亮时自动提高一点

    s_min = 80  # 纯色物块建议 70~120 之间调；越高越抗误检但可能漏暗处边缘

    # 蓝色 H 范围（OpenCV H: 0~179）
    lower_blue = np.array([95,  s_min, v_min])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 红色两段
    lower_red1 = np.array([0,   s_min, v_min])
    upper_red1 = np.array([10,  255,   255])
    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([179, 255,   255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask_blue | mask_red

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def pop_color(img_bgr, mask, s_gain=1.35, v_gain=1.05): # 提升mask区域的饱和度和亮度
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h,s,v = cv2.split(hsv)
    m = (mask > 0)

    s[m] = np.clip(s[m] * s_gain, 0, 255)
    v[m] = np.clip(v[m] * v_gain, 0, 255)

    hsv2 = cv2.merge([h,s,v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

# img = gray_world_wb(img)                 # 稳白平衡
img2 = clahe_on_v(img, clip=2.0)         # 稳暗光
mask = get_red_blue_mask(img2)            # 分割红/蓝
img  = pop_color(img, mask, 1.4, 1.0) 

cv2.namedWindow("Enhanced", cv2.WINDOW_NORMAL)
cv2.imshow("Enhanced", img)
cv2.moveWindow("Enhanced", 600, 20)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()