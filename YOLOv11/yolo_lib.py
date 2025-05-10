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
                custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
                args = {**self.model.overrides, **custom}
                self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model)
            self.model.predictor.model.ov_compiled_model = quantized_seg_compiled_model

    def update(self, image: np.ndarray, content: dict):
        results = self.model(image)
        segmentation_mask = np.zeros_like(image, dtype=np.uint8)
        content["corners"] = []  # 初始化角点存储列表

        for result in results:
            if result.masks is None:
                continue
            
            # 获取当前目标的置信度（取第一个检测框的置信度）
            conf = result.boxes.conf[0].item() if len(result.boxes.conf) > 0 else 0.0
            
            # 只处理置信度>=0.9的目标
            if conf >= 0.9:
                for mask in result.masks.xy:
                    mask_points = np.array(mask, dtype=np.int32)
                    
                    # 计算掩膜的凸包来获取四个角点
                    hull = cv2.convexHull(mask_points)
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 对四个点进行排序：左上、右上、右下、左下
                    box = sorted(box, key=lambda x: x[0])
                    left = sorted(box[:2], key=lambda x: x[1])
                    right = sorted(box[2:], key=lambda x: x[1], reverse=True)
                    sorted_corners = np.array([left[0], right[0], right[1], left[1]])
                    
                    # 存储角点和置信度
                    content["corners"].append({
                        "corners": sorted_corners,
                        "confidence": conf
                    })
                    
                    # 绘制分割掩码
                    segmentation_mask = cv2.fillPoly(segmentation_mask, [mask_points], (0, 255, 0))
        
        # 可视化结果
        if self.show:
            result = results[0]
            image_result = result.plot()
            
            # 在可视化图像上标记掩膜的角点
            for corner_data in content["corners"]:
                for pt in corner_data["corners"]:
                    cv2.circle(image_result, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
                # 绘制连接线
                pts = corner_data["corners"].reshape((-1,1,2))
                cv2.polylines(image_result, [pts], True, (255,0,0), 2)
            
            image[:] = image_result