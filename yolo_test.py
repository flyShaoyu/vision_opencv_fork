from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyYOLO():
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        self.use_intel = use_intel
        self.model_path = model_path
        
        # Intel OpenVINO 优化（可选）
        if self.use_intel:
            try:
                import openvino.runtime as ov
                from openvino.runtime import Core
                import openvino.properties.hint as hints
                
                self.model = YOLO(Path(model_path).parent)
                config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
                core = Core()
                model = core.read_model(model_path)
                self.quantized_seg_compiled_model = core.compile_model(model, config=config)
                
                # 初始化 predictor（根据 Ultralytics 接口调整）
                if self.model.predictor is None:
                    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
                    args = {**self.model.overrides, **custom}
                    self.model.predictor = self.model._smart_load("predictor")(
                        overrides=args, 
                        _callbacks=self.model.callbacks
                    )
                    self.model.predictor.setup_model(model=self.model.model)
                self.model.predictor.model.ov_compiled_model = self.quantized_seg_compiled_model
                logging.info("Intel OpenVINO 优化已启用")
            except Exception as e:
                logging.warning(f"OpenVINO 初始化失败: {str(e)}，将使用默认 CPU 推理")
                self.use_intel = False

    def update(self, image: np.ndarray, content: dict, confidence_threshold=0.5):
        """主更新函数，处理目标检测和角点提取"""
        content.clear()
        content["detected"] = False
        content["corners"] = None
        content["confidence"] = 0.0
        content["mask"] = None

        try:
            # 执行目标检测
            results = self.model(image, conf=confidence_threshold)
            if not results or len(results) == 0:
                logging.info("未检测到任何结果")
                return
            
            # 获取第一个检测结果
            result = results[0]
            if not hasattr(result, 'boxes') or not hasattr(result, 'masks'):
                logging.info("检测结果格式不完整")
                return
            
            # 检查是否有检测到的目标
            if len(result.boxes) == 0 or result.masks is None:
                logging.info("未检测到带掩码的目标")
                return
            
            # 获取所有检测框的置信度
            confidences = result.boxes.conf.cpu().numpy()
            
            # 获取掩码列表
            if hasattr(result.masks, 'xy'):
                masks_xy = result.masks.xy  # 每个掩码的轮廓点
            else:
                logging.info("Masks对象不包含xy属性")
                return
            
            # 确保掩码数量和置信度数量匹配
            if len(masks_xy) != len(confidences):
                valid_range = min(len(masks_xy), len(confidences))
                logging.warning(f"掩码数量({len(masks_xy)})与置信度数量({len(confidences)})不匹配，使用有效范围: {valid_range}")
                masks_xy = masks_xy[:valid_range]
                confidences = confidences[:valid_range]
            
            # 如果有多个检测结果，选择置信度最高的
            if len(confidences) > 0:
                best_idx = np.argmax(confidences)
                confidence = float(confidences[best_idx])
                mask_points = np.array(masks_xy[best_idx], dtype=np.float32).reshape(-1, 2)
                
                if len(mask_points) < 4:
                    logging.warning(f"掩膜点数不足: {len(mask_points)}，需要至少4个点")
                    return
                
                # 后处理掩膜
                processed_mask = self._postprocess_mask(mask_points, image.shape[:2])
                binary_mask = self._generate_binary_mask(processed_mask, image.shape[:2])
                final_corners = self._optimized_corner_detection(binary_mask, image.shape[:2])
                
                # 验证角点有效性
                if self._validate_corners(final_corners):
                    ordered_corners = self._order_corners(final_corners)
                    content.update({
                        "detected": True,
                        "corners": ordered_corners,
                        "confidence": confidence,
                        "mask": binary_mask,
                        "raw_mask_points": mask_points,
                        "processed_mask_points": processed_mask
                    })
                    logging.info(f"成功检测到角点，置信度: {confidence:.4f}")
                else:
                    logging.warning("角点检测失败，结果无效")
            else:
                logging.info("未检测到有效目标")
                
        except Exception as e:
            logging.error(f"检测流程出错: {str(e)}")
            content["detected"] = False

        # 可视化结果（可选）
        if self.show and content["detected"]:
            self._visualize_results(image, content)

    def _postprocess_mask(self, mask_points, image_shape):
        """对预测掩膜进行平滑和形态学处理"""
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        
        # 高斯平滑
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 提取最大轮廓
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_points
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 多边形逼近
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        return cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2).astype(np.float32)

    def _generate_binary_mask(self, points, image_shape):
        """生成严格二值化掩膜"""
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [points.astype(np.int32)], 255)
        _, binary_mask = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 增强边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _optimized_corner_detection(self, binary_mask, image_shape):
        """基于二值化图像的多策略角点检测"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.warning("未找到轮廓")
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # 面积过滤
        if contour_area < (image_shape[0] * image_shape[1]) * 0.01:
            logging.warning(f"轮廓面积过小: {contour_area}")
            return None

        # 多边形逼近优先
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)

        # Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(
            binary_mask, 
            maxCorners=4, 
            qualityLevel=0.01, 
            minDistance=max(image_shape)//20
        )
        if corners is not None and len(corners) == 4:
            return corners.reshape(-1, 2)

        # 方向投影法后备
        return self._directional_projection(largest_contour, image_shape)

    def _directional_projection(self, contour, image_shape):
        """基于四个方向的极值点检测"""
        contour_points = contour.reshape(-1, 2)
        center = np.mean(contour_points, axis=0)
        directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        extreme_points = []
        
        for dir in directions:
            projections = np.dot(contour_points - center, dir)
            idx = np.argmax(projections)
            extreme_points.append(contour_points[idx])
        
        return np.array(extreme_points, dtype=np.float32)

    def _validate_corners(self, corners):
        """验证角点是否为有效四边形"""
        if corners is None or len(corners) != 4:
            return False
        try:
            return np.all(np.isfinite(corners)) and corners.shape == (4, 2)
        except:
            return False

    def _order_corners(self, corners):
        """将角点排序为 左上-右上-右下-左下"""
        if len(corners) != 4:
            return corners
        
        # 基于质心排序
        center = np.mean(corners, axis=0)
        sums = corners.sum(axis=1)
        diffs = corners[:, 0] - corners[:, 1]
        
        top_left = corners[np.argmin(sums)]
        bottom_right = corners[np.argmax(sums)]
        top_right = corners[np.argmax(diffs)]
        bottom_left = corners[np.argmin(diffs)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _visualize_results(self, image, content):
        """可视化检测结果"""
        try:
            mask = content["mask"]
            corners = content["corners"].astype(np.int32)
            
            # 绘制掩膜
            overlay = image.copy()
            cv2.fillPoly(overlay, [content["processed_mask_points"].astype(np.int32)], (0, 255, 0, 50))
            
            # 绘制角点和轮廓
            cv2.polylines(overlay, [corners.reshape(-1, 1, 2)], True, (255, 0, 255), 2)
            for i, (x, y) in enumerate(corners):
                cv2.circle(overlay, (x, y), 8, [ (0,0,255), (0,255,0), (255,0,0), (255,255,0) ][i], -1)
                cv2.putText(overlay, f"P{i+1}", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # 融合结果
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            cv2.imshow("YOLO Detection", image)
            cv2.waitKey(1)
        except Exception as e:
            logging.error(f"可视化出错: {str(e)}")