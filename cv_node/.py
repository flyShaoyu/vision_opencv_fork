import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import time
import cv2
import numpy as np
from cv_lib.ros.cv_bridge import ImagePublish_t, ImageReceive_t
from YOLOv11.yolo_lib import MyYOLO

def main():
    # 创建处理管道
    pipe = []
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(MyYOLO("/home/Elaina/yolo/best.pt", show=True))
    pipe.append(ImagePublish_t("yolo"))
    
    content = {}
    print_time = True
    
    try:
        while True:
            # 初始化图像（实际应用中应从相机获取）
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 处理管道
            for p in pipe:
                try:
                    if print_time:
                        start_time = time.time()
                    
                    p.update(image, content)
                    
                    if print_time:
                        end_time = time.time()
                        print(f"{type(p).__name__}: {(end_time - start_time)*1000:.2f} ms")
                
                except Exception as e:
                    print(f"处理模块 {type(p).__name__} 出错: {str(e)}")
                    continue
            
            # 显示结果
            cv2.imshow("Detection Result", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main()
