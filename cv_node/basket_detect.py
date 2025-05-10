# 导入所需的库
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import time
import cv2
import numpy as np
from cv_lib.ros.cv_bridge import ImagePublish_t, ImageReceive_t
from PoseSolver.Aruco import Aruco
from PoseSolver.PoseSolver import PoseSolver
from YOLOv11.yolo_lib import MyYOLO

def main():
    # 相机内参矩阵
    camera_matrix = np.array([
        [606.634521484375, 0, 433.2264404296875],
        [0, 606.5910034179688, 247.10369873046875],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32)
    
    # 畸变系数
    dist_coeffs = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
    
    # 创建处理管道
    pipe = []
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(MyYOLO("/home/Elaina/yolo/best.pt", show=True))
    pipe.append(ImagePublish_t("yolo"))
    
    # 添加位姿解算器
    pose_solver = PoseSolver(camera_matrix, dist_coeffs, marker_length=0.1)  # 假设标记长度为0.1米
    
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
            
            # 如果有检测到目标，进行位姿解算
            if "corners" in content and len(content["corners"]) > 0:
                print("\n" + "="*50)
                print(f"检测到 {len(content['corners'])} 个目标:")
                
                for i, corner_data in enumerate(content["corners"]):
                    corners = corner_data["corners"]
                    conf = corner_data["confidence"]
                    
                    try:
                        # 位姿解算
                        rvec, tvec = pose_solver.solve_pose(corners)
                        pose_solver.draw_axis(image, rvec, tvec)
                        
                        # 打印结果
                        print(f"\n目标 {i+1} (置信度: {conf:.2f}):")
                        print(f"角点坐标:")
                        for j, pt in enumerate(corners):
                            print(f"  点{j+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
                        print(f"位置向量: [{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]")
                    
                    except Exception as e:
                        print(f"目标 {i+1} 位姿解算失败: {str(e)}")
                        continue
                
                print("="*50 + "\n")
                
                # 清空当前帧数据
                content["corners"] = []
            
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