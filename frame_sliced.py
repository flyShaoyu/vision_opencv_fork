#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
from datetime import datetime

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        # 创建保存图像的文件夹
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0
        self.save_interval = 30  # 每5帧保存一次（可根据需要调整）
        
        self.get_logger().info("图像订阅节点已启动，准备接收图像...")

    def image_callback(self, msg):
        try:
            # 转换压缩图像为OpenCV格式
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("图像解码失败！")
                return
            
            # 显示图像
            cv2.imshow("Camera Feed", cv_image)
            key = cv2.waitKey(1)
            
            # 保存图像（按间隔或按's'键手动保存）
            if self.frame_count % self.save_interval == 0 or key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.save_dir}/frame_{timestamp}_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f"图像已保存: {filename}")
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"处理图像时出错: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()