import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from DepthCamera import check_spearhead, rot_x, rot_y, rot_z, DepthCamNode, img_preprocess, get_yolo_result, tfmsg_to_Rt
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import json
import yaml

class ImageNode(DepthCamNode):
    def __init__(self):
        super().__init__('ros_image_node')

        self.grp = ReentrantCallbackGroup() # 组内可并发
        self.color_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.depcam_color_callback,
            qos_profile_sensor_data,
            callback_group=self.grp)
        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depcam_depth_callback,
            qos_profile_sensor_data,
            callback_group=self.grp)
        self.fuction_subscriber = self.create_subscription(
            String,
            '/update_exec_req',
            self.fuction_check,
            10,
            callback_group=self.grp)
        self.data_publisher = self.create_publisher(
            String,
            '/exec_result',
            10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.depcam_color_image = None
        self.depcam_depth_image = None
        self.pc_need = 0

        self.spearhead_need = threading.Event()
        self.YOLO_need = threading.Event()

        threading.Thread(target=self.spearhead_check_thread, daemon=True).start()
        threading.Thread(target=self.YOLO_detection_thread, daemon=True).start()

    def depcam_color_callback(self, msg):
        self.depcam_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depcam_depth_callback(self, msg):
        self.depcam_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32) / 1000.0
        if self.pc_need:
            self.pc = self.depth_camera.depth2points(self.depcam_depth_image)

    def fuction_check(self, msg):
        if msg.data == 'spearhead':
            self.spearhead_need.set()
            self.get_logger().info('Spearhead check requested.')
        elif msg.data == 'spearhead_stop':
            self.spearhead_need.clear()
            self.get_logger().info('Spearhead check stopped.')
        elif msg.data == 'YOLO':
            self.YOLO_need.set()
            self.get_logger().info('YOLO detection requested.')
        elif msg.data == 'YOLO_stop':
            self.YOLO_need.clear()
            self.get_logger().info('YOLO detection stopped.')
        # else:
        #     self.get_logger().warn(f'Unknown function request: {msg.data}')

    def spearhead_check_thread(self):
        while True:
            self.spearhead_need.wait()
            pc = self.pc
            T_box_cam_map = np.array([0, 0.33, 1.0]) # 填充 目标为右，下，前
            R_box_cam_map = rot_z(00) @ rot_y(00) @ rot_x(-0) # 填充 摇摆角@俯角@右转角，即为欧拉角
            source = "map"
            target = "camera_color_optical_frame"
            try:
                tf_cam_map = self.tf_buffer.lookup_transform(
                    target, source, rclpy.time.Time()
                )
                R_cam_map, t_cam_map = tfmsg_to_Rt(tf_cam_map)

                T_box_cam = T_box_cam_map @ R_cam_map.T + t_cam_map      # box 在 cam 下的平移
                R_box_cam = R_box_cam_map @ R_cam_map.T 
            except TransformException as ex:
                self.get_logger().warn(f"TF not ready: {ex}")

            box_check_t = check_spearhead(pc, self.depth_camera, T_box_cam, R_box_cam)
            self.get_logger().info(f"Spearhead check result: {box_check_t}")
            result_msg = String()
            result_dic = {"topic_name": "spearhead_check", "data": box_check_t}
            result_msg.data = json.dumps(result_dic)
            self.data_publisher.publish(result_msg)

    def YOLO_detection_thread(self):
        while True:
            self.YOLO_need.wait()
            color_img = self.depcam_color_image
            depression_angle = self.depth_camera.depression_angle
            target_loc = (0, 0, 1.0)  # 填充 目标位置为右，下，前
            target_direct = 0  # 填充 目标朝向为正前方
            target_size = (512, 512)  # 填充 目标尺寸为宽，高

            roi_img, roi_2d = img_preprocess(color_img, depression_angle, target_loc=target_loc, target_direct=target_direct, target_size=target_size)
            result = get_yolo_result(self.yolo_model, roi_img)
            self.get_logger().info(f"YOLO detection completed with {len(result)} results.")
            result_msg = String()
            result_dic = {"topic_name": "YOLO_detection", "data": [res.boxes.xyxy.tolist() for res in result]}
            result_msg.data = json.dumps(result_dic)
            self.data_publisher.publish(result_msg)

def main():
    rclpy.init()
    node = ImageNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()