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

    def update(self, image: np.ndarray, content: dict, confidence_threshold=0.5):
        results = self.model(image)
        content["corners"] = []
    
        # 存储所有符合阈值条件的候选掩膜
        valid_masks = []

        for result in results:
            if result.masks is None:
                continue

            # 获取当前结果中的所有检测框置信度
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'conf'):
                confidences = result.boxes.conf.cpu().numpy()
            else:
                confidences = np.array([0.0])  # 默认值

            # 确保掩膜和置信度数量匹配
            num_masks = len(result.masks.xy)
            num_confs = len(confidences)
            valid_range = min(num_masks, num_confs)

            for i in range(valid_range):
                try:
                    mask = result.masks.xy[i]
                    conf = float(confidences[i])
                
                    # 跳过置信度低于阈值的掩膜
                    if conf < confidence_threshold:
                        continue
                    
                    mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    if len(mask_points) < 4:
                        continue

                    # 只存储基本信息，延迟处理
                    valid_masks.append({
                        "mask": mask_points,
                        "confidence": conf,
                        "raw_result": result,  # 保存原始结果用于后续处理
                        "mask_index": i        # 保存掩膜索引
                    })
                except Exception as e:
                    print(f"掩膜筛选出错: {str(e)}")
                    continue

        # 选择置信度最高的掩膜进行处理
        if valid_masks:
            best_mask_info = max(valid_masks, key=lambda x: x["confidence"])
        
            # 对最高置信度掩膜进行后处理
            processed_points = self._postprocess_mask(best_mask_info["mask"], image.shape[:2])
            
            # 生成二值化图像用于角点检测
            binary_mask = self._generate_binary_mask(processed_points, image.shape[:2])
            
            # 在二值化图像上检测角点
            final_corners = self._optimized_corner_detection(binary_mask, image.shape[:2])
            ordered_corners = self._order_corners(final_corners)
        
            # 保存处理结果
            content["corners"].append({
                "corners": ordered_corners,
                "confidence": best_mask_info["confidence"],
                "raw_points": processed_points
            })
        
            # 调试信息
            print(f"最高置信度: {best_mask_info['confidence']:.4f}")

        # 可视化结果
        if self.show and len(results) > 0:
            self._visualize_results(results[0], image, content, confidence_threshold)

    def _generate_binary_mask(self, points, image_shape):
        """生成严格二值化的掩膜图像(0/255)"""
        # 创建初始掩膜
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [points.astype(np.int32)], 255)
        
        # 二值化处理 - 使用Otsu自适应阈值
        _, binary_mask = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作增强边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary_mask

    def _optimized_corner_detection(self, binary_mask, image_shape) -> np.ndarray:
        """基于二值化图像的角点检测"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((4, 2), dtype=np.float32)
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 多边形逼近
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)
        
        # 使用Shi-Tomasi角点检测
        corners = cv2.goodFeaturesToTrack(binary_mask, 4, 0.01, 10)
        if corners is not None and len(corners) == 4:
            return corners.reshape(-1, 2)
        
        # 方向投影法作为后备
        contour_points = largest_contour.reshape(-1, 2)
        center = np.mean(contour_points, axis=0)
        directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        extreme_points = []
        for dir in directions:
            projections = np.dot(contour_points - center, dir)
            idx = np.argmax(projections)
            extreme_points.append(contour_points[idx])
        
        return np.array(extreme_points, dtype=np.float32)

    def _postprocess_mask(self, mask_points, image_shape):
        """对预测掩膜进行后处理，提升质量"""
        # 创建掩膜
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        
        # 平滑处理 - 可调整高斯核大小
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        # 形态学操作 - 可调整核大小和迭代次数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 查找轮廓并优化
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_points  # 返回原始点作为后备
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 多边形逼近 - 可调整epsilon值
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 返回优化后的轮廓点
        return approx.reshape(-1, 2).astype(np.float32)

    def _is_valid_parallelogram(self, corners, image_shape):
        """判断四边形是否满足平行四边形几何条件"""
        if len(corners) != 4:
            return False
        
        # 计算对边向量差（绝对值）
        vec_top = corners[1] - corners[0]
        vec_bottom = corners[3] - corners[2]
        vec_left = corners[0] - corners[3]
        vec_right = corners[2] - corners[1]
        
        edge_diff = (np.linalg.norm(vec_top + vec_bottom) + 
                    np.linalg.norm(vec_left + vec_right)) / 2
        
        # 计算对角线中点距离
        diag_mid_diff = np.linalg.norm(
            (corners[0] + corners[2])/2 - (corners[1] + corners[3])/2
        )
        
        # 阈值设置（基于图像尺寸的相对误差）
        max_dim = max(image_shape)
        edge_threshold = 0.03 * max_dim  # 对边误差阈值（3%图像边长）
        mid_threshold = 0.02 * max_dim   # 中点误差阈值（2%图像边长）
        
        return edge_diff < edge_threshold and diag_mid_diff < mid_threshold

    def _correct_parallelogram(self, corners, image_shape):
        """仅修正单个异常角点的平行四边形拟合"""
        best_corners = corners.copy()
        min_error = self._calculate_parallelogram_error(corners, image_shape)
        
        # 尝试修正每个角点
        for i in range(4):
            corrected = self._generate_corrected_corners(corners, i)
            current_error = self._calculate_parallelogram_error(corrected, image_shape)
            
            if current_error < min_error:
                min_error = current_error
                best_corners = corrected
        
        return best_corners

    def _generate_corrected_corners(self, corners, bad_corner_idx):
        """根据平行四边形性质生成修正后的角点"""
        others = np.delete(corners, bad_corner_idx, axis=0)
        if bad_corner_idx == 0:
            # 左上 = 左下 + (右上 - 右下)
            return np.array([others[2] + (others[0] - others[1]), others[0], others[1], others[2]])
        elif bad_corner_idx == 1:
            # 右上 = 左上 + (右下 - 左下)
            return np.array([others[0], others[0] + (others[2] - others[1]), others[2], others[1]])
        elif bad_corner_idx == 2:
            # 右下 = 右上 + (左下 - 左上)
            return np.array([others[0], others[1], others[1] + (others[2] - others[0]), others[2]])
        else:  # bad_corner_idx == 3
            # 左下 = 左上 + (右下 - 右上)
            return np.array([others[0], others[1], others[2], others[0] + (others[2] - others[1])])

    def _calculate_parallelogram_error(self, corners, image_shape):
        """计算四边形与平行四边形的几何误差（仅用于比较）"""
        if len(corners) != 4:
            return float('inf')
        
        vec_top = corners[1] - corners[0]
        vec_bottom = corners[3] - corners[2]
        vec_left = corners[0] - corners[3]
        vec_right = corners[2] - corners[1]
        
        # 对边向量和的范数（理想值为0）
        edge_error = np.linalg.norm(vec_top + vec_bottom) + np.linalg.norm(vec_left + vec_right)
        # 对角线中点距离（理想值为0）
        mid_error = np.linalg.norm((corners[0]+corners[2])/2 - (corners[1]+corners[3])/2)
        
        return edge_error + mid_error  # 未归一化误差，仅用于相对比较

    def _order_corners(self, corners):
        """
        确保角点顺序为左上、右上、右下、左下
        """
        if len(corners) != 4:
            return corners
            
        # 计算质心
        center = np.mean(corners, axis=0)
        
        # 计算每个点的x+y和x-y值
        sums = [pt[0] + pt[1] for pt in corners]
        diffs = [pt[0] - pt[1] for pt in corners]
        
        # 左上角: x+y最小
        top_left_idx = np.argmin(sums)
        top_left = corners[top_left_idx]
        
        # 右下角: x+y最大
        bottom_right_idx = np.argmax(sums)
        bottom_right = corners[bottom_right_idx]
        
        # 右上角: x-y最大（在剩余点中）
        remaining = [i for i in range(4) if i not in (top_left_idx, bottom_right_idx)]
        top_right_idx = remaining[np.argmax([corners[i][0]-corners[i][1] for i in remaining])]
        top_right = corners[top_right_idx]
        
        # 左下角: 剩余点
        bottom_left = corners[[i for i in range(4) if i not in (top_left_idx, bottom_right_idx, top_right_idx)][0]]
        
        # 返回有序角点
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _visualize_results(self, result, image, content, confidence_threshold=0.5):
        """可视化检测结果"""
        try:
            # 创建空白图像用于绘制
            image_result = np.zeros_like(image)
        
            # 绘制筛选后的掩膜
            for corner_data in content["corners"]:
                confidence = corner_data["confidence"]
                if confidence >= confidence_threshold:
                    corners = corner_data["corners"].astype(np.int32)
                    raw_points = corner_data["raw_points"].astype(np.int32)
                
                    # 绘制掩膜区域
                    cv2.fillPoly(image_result, [raw_points], (0, 255, 0, 50))  # 半透明绿色
                
                    # 绘制连接线
                    cv2.polylines(image_result, [corners.reshape((-1, 1, 2))],
                              True, (255, 0, 255), 2)

                    # 标记四个极值点（不同颜色）
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
                    labels = ["TL", "TR", "BR", "BL"]  # 左上、右上、右下、左下
                    for i, pt in enumerate(corners):
                        cv2.circle(image_result, tuple(pt), 8, colors[i], -1)
                        cv2.putText(image_result, labels[i],
                                (pt[0] + 10, pt[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                
                    # 添加置信度文本
                    cv2.putText(image_result, f"Conf: {confidence:.2f}",
                                (corners[0][0], corners[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
            # 将绘制结果合并到原图上
            if len(content["corners"]) > 0:
                image[:] = cv2.addWeighted(image, 1, image_result, 0.7, 0)
            else:
                image[:] = image_result  # 如果没有筛选出掩膜，显示原图
            
        except Exception as e:
            print(f"可视化出错: {str(e)}")
