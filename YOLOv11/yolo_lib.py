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
            final_corners = self._optimized_corner_detection(processed_points, image.shape[:2])
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

    def _optimized_corner_detection(self, points: np.ndarray, image_shape) -> np.ndarray:
        """优化的角点检测方法，结合轮廓分析和透视变换原理"""
        # 创建掩膜并查找轮廓
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [points.astype(np.int32)], 255)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((4, 2), dtype=np.float32)
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Douglas-Peucker算法进行多边形逼近
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果逼近结果刚好是四边形，直接使用
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)
        
        # 计算凸包
        hull = cv2.convexHull(largest_contour)
        
        # 角点检测方法
        if len(hull) >= 4:
            # Shi-Tomasi角点检测
            corners = cv2.goodFeaturesToTrack(mask_img, 4, 0.01, 10)
            if corners is not None:
                corners = corners.reshape(-1, 2)
                
                # 确保有4个角点
                if len(corners) == 4:
                    return corners
        
        # 作为后备，使用方向投影法
        contour_points = largest_contour.reshape(-1, 2)
        center = np.mean(contour_points, axis=0)
        
        # 定义四个方向：上、下、左、右
        directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        
        extreme_points = []
        for dir in directions:
            projections = np.dot(contour_points - center, dir)
            idx = np.argmax(projections)
            extreme_points.append(contour_points[idx])
        
        return np.array(extreme_points, dtype=np.float32)

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
        
        # 右上角: x-y最大
        top_right_idx = np.argmax(diffs)
        top_right = corners[top_right_idx]
        
        # 左下角: x-y最小
        bottom_left_idx = np.argmin(diffs)
        bottom_left = corners[bottom_left_idx]
        
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




# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = MyYOLO("yolov8n-seg.pt", show=True)
    
    # 读取测试图像
    image = cv2.imread("test.jpg")
    if image is None:
        print("无法读取测试图像，请确保图像路径正确")
    else:
        # 创建内容字典
        content = {"corners": []}
        
        # 处理图像
        model.update(image, content)
        
        # 显示结果
        cv2.imshow("YOLO Segmentation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    