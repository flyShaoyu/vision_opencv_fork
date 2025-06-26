import sys
import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_lib.ros.basket_ros import ImagePublish_t
from PoseSolver.PoseSolver import PoseSolver
from YOLOv11.yolo_lib import MyYOLO
from sensor_msgs.msg import CompressedImage,Image
from cv_bridge import CvBridge

# 引入新的节点 ImageProcessingNode
class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')

        # 相机内参矩阵
        self.camera_matrix = np.array([
            [606.634521484375, 0, 433.2264404296875],
            [0, 606.5910034179688, 247.10369873046875],
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32)

        # 畸变系数
        self.dist_coeffs = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

        # 初始化各组件
        self.yolo_detector = MyYOLO("/home/Elaina/yolo/best.pt", show=True)
        self.image_publisher = ImagePublish_t("yolo/image")
        self.pose_solver = PoseSolver(
            self.camera_matrix,
            self.dist_coeffs,
            marker_length=0.1,
            print_result=True
        )

        # 图像缓冲区
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bridge = CvBridge()

        self.sub_com = self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.compressed_image_callback, 1)
        self.sub_raw = self.create_subscription(Image, "/camera/color/image_raw", self.regular_image_callback, 1)

        self.get_logger().info("ImageProcessingNode 初始化完成")

    def compressed_image_callback(self, msg: CompressedImage):
        """处理压缩图像消息的回调函数"""
        self.get_logger().info("接收到压缩图像消息")
        try:
            # 解压缩图像
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"处理压缩图像时出错: {str(e)}")
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def regular_image_callback(self, msg: Image):
        """处理非压缩图像消息的回调函数"""
        self.get_logger().info("接收到非压缩图像消息")
        try:
            # 转换非压缩图像
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"处理非压缩图像时出错: {str(e)}")
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def process_image(self):
        
        # YOLO检测
        corners = self.yolo_detector.update(self.image)

        # 发布处理后的图像
        self.image_publisher.update(self.image)

        # 位姿解算 (如果有检测结果)
        if corners is not None and len(corners) > 0:
            self.pose_solver.update(self.image, corners)

            if hasattr(self.pose_solver, 'pnp_result'):
                pose_info = self.pose_solver.pnp_result
                self.get_logger().info(f"\n目标 1:")
                self.get_logger().info(f"  Yaw: {pose_info['yaw']:.1f}°, Pitch: {pose_info['pitch']:.1f}°, Roll: {pose_info['roll']:.1f}°")
                self.get_logger().info(f"  位置: [{pose_info['tvec'][0][0]:.3f}, {pose_info['tvec'][1][0]:.3f}, {pose_info['tvec'][2][0]:.3f}]m")
                self.get_logger().info(f"  距离: {pose_info['distance']:.3f}m")

        # 显示结果
        cv2.imshow("Detection Result", self.image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    try:
        node = ImageProcessingNode()
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        node.destroy_node()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
