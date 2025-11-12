import numpy as np
import open3d as o3d
from image_geometry import PinholeCameraModel
import sensor_msgs_py.point_cloud2 as pc2

def plan_PC_fit(depth_img, point_uv, point_cloud, camera_model, neighbor_radius=0.05, distance_thresh=0.01):
    '''
    根据深度图和点云拟合平面并返回
    depth_img: 深度图像
    point_uv: 属于平面的某点像素坐标(u, v)
    point_cloud: 点云数据 Nx3
    camera_model: 相机模型
    return: 整个平面的四个2d和3d角点(u, v), (x, y, z)
    '''
    # 1️⃣ 根据相机模型，从深度图反投影得到该像素的3D点
    u, v = point_uv
    depth = depth_img[int(v), int(u)]
    if depth <= 0:
        raise ValueError("该像素没有有效深度值。")

    ray = camera_model.projectPixelTo3dRay((u, v))
    P3D = np.array(ray) * depth / ray[2]  # 转为相机坐标系下的3D点

    # 2️⃣ 将该点转换到点云坐标系（如果两者有外参变换）
    # 同一坐标系
    P3D_world = P3D

    # 3️⃣ 构建 KDTree，找到与该点空间距离最近的一批点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    _, idxs, _ = kdtree.search_radius_vector_3d(P3D_world, neighbor_radius)
    local_pts = np.asarray(point_cloud)[idxs]

    if len(local_pts) < 10:
        raise RuntimeError("邻域点太少，可能搜索半径过小或点云太稀疏。")

    # 4️⃣ 拟合平面（RANSAC）
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(local_pts)
    plane_model, inliers = local_pcd.segment_plane(distance_threshold=distance_thresh,
                                                  ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    n = np.array([a, b, c])

    # 5️⃣ 在全局点云中提取平面内点
    dist = np.abs(point_cloud @ n + d) / np.linalg.norm(n)
    mask = dist < distance_thresh
    plane_pts = point_cloud[mask]

    # 6️⃣ 通过 PCA 求平面边界角点
    center = plane_pts.mean(axis=0)
    cov = np.cov((plane_pts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, np.argsort(-eigvals)[:2]]
    proj = (plane_pts - center) @ axes
    min_xy, max_xy = proj.min(0), proj.max(0)
    corners_local = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]]
    ])
    corners_3d = center + corners_local @ axes.T

    # 7️⃣ 投影回图像像素
    corners_2d = []
    for p in corners_3d:
        uv = camera_model.project3dToPixel(p)
        corners_2d.append(uv)
    corners_2d = np.array(corners_2d)

    return corners_2d, corners_3d, plane_model

if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    import time
    import cv2
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from get_point_loc import DepthCamera, pix_to_cam

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.info_msg = None
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info()
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo(info_d=self.info_msg)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.depth_img = None
        self.color_img = None
        self.point = (480,848)

        cv2.namedWindow("Color Image")
        cv2.setMouseCallback("Color Image", self.mouse_callback)

    '''
    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model_d.fromCameraInfo(msg)
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

    def color_callback(self, msg):
        self.color_img = msg
        #self.timetest()

    def pc_callback(self, msg):
        self.point_cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

    def depth_callback(self, msg):
        if self.color_img is None:
            return
        #self.timetest()
        self.depth_img = msg
        cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
        cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
        color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # 深度图叠加到彩色图，颜色表示深度
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
        color_resized = overlay
        u, v = self.point
        depth = cv2_depth_img[int(v), int(u)] / 1000.0  # 转换为米
        #print(cv2_color_img.shape, color_resized.shape)
        cv2.circle(color_resized, (int(u), int(v)), 5, (65535,65535,0), -1) # 黄色圆点
        # 显示圆点坐标及深度
        x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
        if z != 0:
            cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
        else:
            cv2.putText(color_resized, f"(None)", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
        cv2.putText(color_resized, f"({u},{v})", (int(u)+10, int(v)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
        points_2d, points_3d, plane_model = plan_PC_fit(
            cv2_depth_img / 1000.0,
            self.point,
            self.point_cloud,
            self.depth_camera.model_d
        )
        for i in range(4):
            px, py = points_2d[i]
            x, y, z = points_3d[i]
            cv2.circle(color_resized, (int(px), int(py)), 5, (0,65535,0), -1) # 绿色圆点
            cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(px)+10, int(py)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,65535,0), 1)
            cv2.putText(color_resized, f"({int(px)},{int(py)})", (int(px)+10, int(py)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,65535,0), 1)
        cv2.imshow("Color Image", color_resized)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
            depth = depth[y, x] / 1000.0  # 转换为米
            self.points[self.click_point] = (x, y)
            self.get_logger().info(f"point chosen at ({x}, {y})")

def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
