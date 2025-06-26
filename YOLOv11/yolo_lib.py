from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
from cv_mask.cv_mask import MaskProcessor

class MyYOLO:
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        self.mask_processor = MaskProcessor()  # 实例化掩膜处理器
        
        if use_intel:
            # OpenVINO 加速部分（若无需加速可忽略）
            import openvino.runtime as ov
            from openvino.runtime import Core
            import openvino.properties.hint as hints
            self.model = YOLO(Path(model_path).parent)
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
            core = Core()
            model = core.read_model(model_path)
            quantized_seg_compiled_model = core.compile_model(model, config=config)
            if self.model.predictor is None:
                custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
                args = {**self.model.overrides, **custom}
                self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model)
            self.model.predictor.model.ov_compiled_model = quantized_seg_compiled_model

    def update(self, image: np.ndarray, content: dict, confidence_threshold=0.8, epsilon=0.05):
        """
        处理图像并更新 content 字典，输出兼容 PoseSolver 的角点格式
        :param image: 输入图像(BGR格式)
        :param content: 存储结果的字典，包含 "corners" 字段
        :param confidence_threshold: 置信度阈值
        :param epsilon: 多边形拟合参数
        """
        results = self.model(image)
        content.clear()  # 清空旧数据
        content["contours"] = []
        content["approx_polygons"] = []
        content["corner_points"] = []  # 原始角点（调试用）
        content["corners"] = []         # 兼容 PoseSolver 的角点列表

        # 筛选有效掩膜
        valid_masks = self._filter_valid_masks(results, confidence_threshold)

        # 处理每个有效掩膜
        for mask_info in valid_masks:
            # 使用独立的掩膜处理器进行后处理
            processed_points = self.mask_processor.postprocess_mask(mask_info["mask"], image.shape[:2])
            contour = processed_points.astype(np.int32).reshape(-1, 1, 2)
            approx_polygon = self.mask_processor.fit_polygon(contour, epsilon)
            
            # 保存中间结果（调试用）
            self._save_intermediate_results(content, mask_info, contour, approx_polygon)
            
            # 提取并筛选角点
            corner_points = self.mask_processor.extract_corner_points(approx_polygon, mask_info["confidence"])
            if corner_points is None:
                continue  # 跳过顶点数不足的情况
            
            # 保存兼容 PoseSolver 的角点数据
            self._save_pose_solver_corners(content, corner_points, mask_info["confidence"])

        # 可视化结果
        if self.show and content["corners"]:
            self._visualize_results(content["approx_polygons"], image, confidence_threshold)

    def _filter_valid_masks(self, results, confidence_threshold):
        """筛选出符合置信度阈值的掩膜"""
        valid_masks = []
        for result in results:
            if result.masks is None:
                continue
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else np.array([0.0])
            for i in range(min(len(result.masks.xy), len(confidences))):
                try:
                    mask = result.masks.xy[i]
                    conf = float(confidences[i])
                    if conf < confidence_threshold:
                        continue
                    mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    if len(mask_points) < 4:
                        continue
                    valid_masks.append({"mask": mask_points, "confidence": conf})
                except Exception as e:
                    print(f"掩膜筛选错误: {str(e)}")
        return valid_masks

    def _save_intermediate_results(self, content, mask_info, contour, approx_polygon):
        """保存中间结果（用于调试和可视化）"""
        content["contours"].append({
            "contour": contour,
            "confidence": mask_info["confidence"]
        })
        content["approx_polygons"].append({
            "polygon": approx_polygon.reshape(-1, 1, 2),  # 保留原始格式用于可视化
            "confidence": mask_info["confidence"],
            "vertex_count": len(approx_polygon)
        })

    def _save_pose_solver_corners(self, content, corners_pts, confidence):
        """保存兼容 PoseSolver 的角点格式"""
        content["corners"].append({
            "corners": corners_pts,  # 形状 (4,2) 的 numpy 数组
            "confidence": confidence,
            "class_id": 0,  # 类别ID（可根据模型输出调整）
            "label": "object"  # 目标标签（可根据模型输出调整）
        })

    def _visualize_results(self, polygons_data, image, confidence_threshold):
        """可视化多边形拟合结果"""
        image_result = np.zeros_like(image)
        for poly_info in polygons_data:
            if poly_info["confidence"] < confidence_threshold:
                continue
            polygon = poly_info["polygon"]
            cv2.polylines(image_result, [polygon], True, (0, 0, 255), 2)
            for point in polygon:
                cv2.circle(image_result, tuple(point[0]), 3, (255, 0, 0), -1)
            cv2.putText(image_result, f"Conf: {poly_info['confidence']:.2f}, Vertices: {poly_info['vertex_count']}",
                       (polygon[0][0][0], polygon[0][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        image[:] = cv2.addWeighted(image, 1, image_result, 0.5, 0)