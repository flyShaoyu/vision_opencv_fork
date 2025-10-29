import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
#import time
import yaml
from scipy.signal import find_peaks

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

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

class DepthCamera:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.bin_width = 0.005  # Histogram bin width in meters

    def loadCameraInfo(self, info = None):
        if info is None:
            self.model.fromCameraInfo(load_camera_info('orbbec336L/depth_camera_info.yaml'))
        else:
            self.model.fromCameraInfo(info)

    def pixelRangeToSpace(self, range, img):
        depth_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough').astype(np.float32) / 1000.0  # Convert mm to meters
        depth_data = np.zeros((range[2]-range[0], range[3]-range[1], 3), dtype=np.float32)
        for v in range(range[0], range[2]):
            for u in range(range[1], range[3]):
                depth = depth_img[v, u]
                X, Y, Z = pix_to_cam(u, v, depth, self.model)
                depth_data[v - range[0], u - range[1], 0] = X
                depth_data[v - range[0], u - range[1], 1] = Y
                depth_data[v - range[0], u - range[1], 2] = Z
        return depth_data
    
    def depthImageFindCenter(self, range, img):
        depth_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough').astype(np.float32) / 1000.0  # Convert mm to meters
        depth_need = depth_img[range[0]:range[2], range[1]:range[3]]
        depth_valid = depth_need[~np.isnan(depth_need) & (depth_need != 0)]
        if depth_valid.size == 0:
            print("No valid depth data in the specified range.")
            return None
        # 建立深度直方图
        depth_min, depth_max = np.min(depth_valid), np.max(depth_valid)
        bins = int((depth_max - depth_min) / self.bin_width)
        hist, bin_edges = np.histogram(depth_valid, bins=bins, range=(depth_min, depth_max))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 每个bin中心
        # 寻找直方图中的峰值
        peaks, props = find_peaks(hist, height=0.1*depth_valid.size) # 最小高度为总点数的10%
        peak_positions = centers[peaks]
        peak_heights = props["peak_heights"]
        if len(peaks) == 0:
            print("No significant peaks found in depth histogram.")
            return []
        # 合并接近的峰值
        merge_thresh = 3 * self.bin_width # 合并阈值设为小于3个bin宽度（相邻或隔一个）
        merged = []
        cur_group = [0]
        for i in range(1, len(peak_positions)):
            if abs(peak_positions[i] - peak_positions[cur_group[-1]]) < merge_thresh:
                cur_group.append(i)
            else:
                merged.append(cur_group)
                cur_group = [i]
        merged.append(cur_group)
        # 计算每个大峰的中心和范围
        peaks_merged = []
        major_peak = None
        for group in merged:
            idx = peaks[group]
            total_count = np.sum(hist[idx])
            depth_range = (peak_positions[group[0]], peak_positions[group[-1]])
            center_depth = np.mean(peak_positions[group])
            peaks_merged.append({
                'center': center_depth,
                'count': int(total_count),
                'range': depth_range
            })
            if total_count > 0.5*depth_valid.size: # 如果某个峰值占比超过50%，则认为是主要峰值
                major_peak = peaks_merged[-1]
                break
            if total_count > 0.3*depth_valid.size and major_peak is None: # 如果某个峰值占比超过30%且没有超过50%的峰，则取深度最小的峰
                major_peak = peaks_merged[-1]
        print(f"Major peak found at depth {major_peak['center']:.3f} m , range [{major_peak['range'][0]:.3f}, {major_peak['range'][1]:.3f}] m, with count {major_peak['count']}, {major_peak['count']/depth_valid.size*100:.1f}% of valid points.") 
        valid_pixels = self.findValidPix(major_peak['range'][0], major_peak['range'][1], depth_need)
        center_u, center_v = self.findMassCenter(valid_pixels)
        return center_u + range[1], center_v + range[0], major_peak['center']

    def findValidPix(self, depth_floor, depth_ceiling, img):
        depth_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough').astype(np.float32) / 1000.0  # Convert mm to meters
        #valid_pixels = [[0 for i in range(depth_img.shape[0])] for j in range(depth_img.shape[1])]
        valid_pixels = np.zeros_like(depth_img, dtype=bool)
        for v in range(depth_img.shape[0]):
            for u in range(depth_img.shape[1]):
                depth = depth_img[v, u]
                if not np.isnan(depth) and depth >= depth_floor and depth <= depth_ceiling:
                    valid_pixels[v, u] = 1
        return valid_pixels
    
    def findMassCenter(self, valid_pixels):
        sum_u = 0
        sum_v = 0
        count = 0
        for v in range(valid_pixels.shape[0]):
            for u in range(valid_pixels.shape[1]):
                if valid_pixels[v, u] == 1:
                    sum_u += u
                    sum_v += v
                    count += 1
        if count == 0:
            return None
        center_u = sum_u / count
        center_v = sum_v / count
        return center_u, center_v

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
