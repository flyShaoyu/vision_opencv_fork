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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_system_default

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

        self.sub = self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.image_callback, 1)

        self.get_logger().info("ImageProcessingNode 初始化完成")

    def image_callback(self, msg:CompressedImage):
        
        # 打印接收到的消息基本信息
        print("\n=== 接收到新的图像消息 ===")
        print(f"消息时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        print(f"消息格式: {msg.format}")
        print(f"数据长度: {len(msg.data)} 字节")
        print(f"QoS配置: {self.sub.qos_profile}")

        self.get_logger().info("图像回调函数被调用")

        try:
            # 1. 接收图像
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("成功更新图像")

            content = {}
            # 2. YOLO检测
            self.get_logger().info("开始YOLO检测")
            self.yolo_detector.update(self.image, content)
            self.get_logger().info(f"YOLO检测完成,检测到 {len(content.get('corners', []))} 个目标")

            # 3. 发布处理后的图像
            self.get_logger().info("准备发布处理后的图像")
            self.image_publisher.update(self.image)
            self.get_logger().info("图像发布完成")

            # 4. 位姿解算 (如果有检测结果)
            if 'corners' in content and len(content['corners']) > 0:
                self.get_logger().info(f"检测到 {len(content['corners'])} 个目标，开始位姿解算...")

                # 进行位姿解算
                self.pose_solver.update(self.image, content)

                if hasattr(self.pose_solver, 'pnp_result'):
                    pose_info = self.pose_solver.pnp_result
                    self.get_logger().info(f"\n目标 1:")
                    self.get_logger().info(f"  Yaw: {pose_info['yaw']:.1f}°, Pitch: {pose_info['pitch']:.1f}°, Roll: {pose_info['roll']:.1f}°")
                    self.get_logger().info(f"  位置: [{pose_info['tvec'][0][0]:.3f}, {pose_info['tvec'][1][0]:.3f}, {pose_info['tvec'][2][0]:.3f}]m")
                    self.get_logger().info(f"  距离: {pose_info['distance']:.3f}m")

                self.get_logger().info("位姿解算完成")

            # 显示结果
            cv2.imshow("Detection Result", self.image)
            cv2.waitKey(1)
            self.get_logger().info("图像显示更新")

        except Exception as e:
            self.get_logger().error(f"处理图像时出错: {str(e)}")
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")


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
        print("程序结束")


if __name__ == "__main__":
    main()
