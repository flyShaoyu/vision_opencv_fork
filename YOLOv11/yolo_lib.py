from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path


class MyYOLO():
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        if use_intel:
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
        results = self.model(image)
        content["contours"] = []
        content["approx_polygons"] = []  # 存储多边形拟合结果

        # 存储所有符合阈值条件的候选掩膜
        valid_masks = []

        for result in results:
            if result.masks is None:
                continue

            # 获取当前结果中的所有检测框置信度
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else np.array([0.0])

            # 处理每个掩膜
            for i in range(min(len(result.masks.xy), len(confidences))):
                try:
                    mask = result.masks.xy[i]
                    conf = float(confidences[i])
                    
                    # 跳过置信度低于阈值的掩膜
                    if conf < confidence_threshold:
                        continue
                    
                    mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    if len(mask_points) < 4:
                        continue

                    valid_masks.append({
                        "mask": mask_points,
                        "confidence": conf,
                    })
                except Exception as e:
                    print(f"掩膜筛选出错: {str(e)}")
                    continue

        # 处理每个有效掩膜
        for mask_info in valid_masks:
            processed_points = self._postprocess_mask(mask_info["mask"], image.shape[:2])
            contour = processed_points.astype(np.int32).reshape(-1, 1, 2)
            
            # 多边形拟合
            perimeter = cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            
            content["contours"].append({
                "contour": contour,
                "confidence": mask_info["confidence"],
            })
            
            content["approx_polygons"].append({
                "polygon": approx_polygon,
                "confidence": mask_info["confidence"],
                "vertex_count": len(approx_polygon),  # 多边形顶点数
            })
            
            print(f"检测到掩膜 - 置信度: {mask_info['confidence']:.4f}, 拟合多边形顶点数: {len(approx_polygon)}")

        # 可视化结果
        if self.show and len(content["approx_polygons"]) > 0:
            self._visualize_results(content["approx_polygons"], image, confidence_threshold)

    def _postprocess_mask(self, mask_points, image_shape):
        """对预测掩膜进行后处理，提升质量"""
        # 创建掩膜
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        
        # 平滑处理
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 查找轮廓并返回
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) if contours else mask_points

    def _visualize_results(self, polygons_data, image, confidence_threshold=0.8):
        """可视化多边形拟合结果"""
        try:
            image_result = np.zeros_like(image)
            
            for polygon_info in polygons_data:
                confidence = polygon_info["confidence"]
                if confidence >= confidence_threshold:
                    polygon = polygon_info["polygon"]
                    vertex_count = polygon_info["vertex_count"]
                    
                    # 绘制填充区域
                    cv2.fillPoly(image_result, [polygon], (0, 255, 0, 50))
                    
                    # 绘制多边形轮廓（蓝色）
                    cv2.polylines(image_result, [polygon], True, (0, 0, 255), 2)
                    
                    # 标记多边形顶点（红色圆点）
                    for point in polygon:
                        cv2.circle(image_result, tuple(point[0]), 5, (255, 0, 0), -1)
                    
                    # 添加置信度和顶点数文本
                    cv2.putText(image_result, f"Conf: {confidence:.2f}, Vertices: {vertex_count}",
                               (polygon[0][0][0], polygon[0][0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 合并到原图
            image[:] = cv2.addWeighted(image, 1, image_result, 0.7, 0)
        
        except Exception as e:
            print(f"可视化出错: {str(e)}")
