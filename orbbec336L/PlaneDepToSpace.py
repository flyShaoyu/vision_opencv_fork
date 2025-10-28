import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
#import time
import yaml

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

def pixel_range_to_space(range, img, info = None):
    bridge = CvBridge()
    model = PinholeCameraModel()
    if info is None:
        model.fromCameraInfo(load_camera_info('orbbec336L/depth_camera_info.yaml'))
    else:
        model.fromCameraInfo(info)
    depth_img = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough').astype(np.float32) / 1000.0  # Convert mm to meters
    depth_data = np.zeros((range[2]-range[0], range[3]-range[1], 3), dtype=np.float32)
    for v in range(range[0], range[2]):
        for u in range(range[1], range[3]):
            depth = depth_img[v, u]
            X, Y, Z = pix_to_cam(u, v, depth, model)
            depth_data[v - range[0], u - range[1], 0] = X
            depth_data[v - range[0], u - range[1], 1] = Y
            depth_data[v - range[0], u - range[1], 2] = Z
    return depth_data

def load_camera_info(yaml_path: str) -> CameraInfo:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    msg = CameraInfo()
    msg.width  = data['image_width']
    msg.height = data['image_height']
    msg.distortion_model = data['distortion_model']
    msg.header.frame_id = data.get('header', {}).get('frame_id', '')
    def to_float_list(x):
        if isinstance(x, str):
            x = x.strip('[]').replace(',', ' ').split()
        return [float(i) for i in x]
    msg.d = to_float_list(data['distortion_coefficients']['data'])
    msg.k = to_float_list(data['camera_matrix']['data'])
    msg.r = to_float_list(data['rectification_matrix']['data'])
    msg.p = to_float_list(data['projection_matrix']['data'])
    msg.binning_x = data.get('binning_x', 0)
    msg.binning_y = data.get('binning_y', 0)
    roi = data.get('roi', {})
    msg.roi.x_offset = roi.get('x_offset', 0)
    msg.roi.y_offset = roi.get('y_offset', 0)
    msg.roi.height = roi.get('height', 0)
    msg.roi.width = roi.get('width', 0)
    msg.roi.do_rectify = roi.get('do_rectify', False)
    return msg

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.info_msg = None
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info()
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.img = None
    '''
    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model.fromCameraInfo(msg)
        self.cameraInfoInit = True
    '''
    def wait_for_camera_info(self, timeout_sec=1.0):
        #阻塞等待一次 /camera/depth/camera_info 消息
        future = rclpy.task.Future()
        def callback(msg):
            if not future.done():
                future.set_result(msg)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError("CameraInfo timeout")
        return future.result()

    def depth_callback(self, msg):
        self.img = msg


def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
