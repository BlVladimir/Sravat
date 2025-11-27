import cv2
import numpy as np

class ArUcoPlaneDetector:
    def __init__(self, dictionary_type=cv2.aruco.DICT_6X6_250):
        """Инициализация детектора ArUco маркеров"""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Цвета для визуализации
        self.colors = {
            'contour': (255, 0, 0),      # Синий контур
            'fill': (0, 255, 255),       # Желтая заливка
            'center': (0, 255, 0),       # Зеленый центр
            'corners': [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Угловые точки
        }
    
    def detect_markers(self, frame):
        """Детектирует ArUco маркеры и возвращает их центры и углы"""
        corners, ids, rejected = self.detector.detectMarkers(frame)
        
        if ids is None:
            return frame, [], []
        
        # Рисуем обнаруженные маркеры
        output_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)
        
        # Собираем центры маркеров
        centers = []
        for corner in corners:
            center = np.mean(corner[0], axis=0)
            centers.append(center)
            # Рисуем центр маркера
            cv2.circle(output_frame, tuple(center.astype(int)), 3, self.colors['center'], -1)
        
        return output_frame, corners, centers
    
    def sort_points(self, points):
        """Сортирует точки в порядке: top-left, top-right, bottom-right, bottom-left"""
        points = np.array(points)
        
        if len(points) != 4:
            return None
            
        # Сортируем по y-координате
        y_sorted = points[np.argsort(points[:, 1])]
        
        # Верхние две точки
        top_points = y_sorted[:2]
        # Нижние две точки  
        bottom_points = y_sorted[2:]
        
        # Сортируем верхние точки по x
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        tl, tr = top_sorted[0], top_sorted[1]
        
        # Сортируем нижние точки по x
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bl, br = bottom_sorted[0], bottom_sorted[1]
        
        return [tl, tr, br, bl]
    
    def calculate_plane_dimensions(self, src_points):
        """Вычисляет ширину и высоту плоскости"""
        width = max(
            np.linalg.norm(src_points[0] - src_points[1]),  # Верхняя сторона
            np.linalg.norm(src_points[2] - src_points[3])   # Нижняя сторона
        )
        
        height = max(
            np.linalg.norm(src_points[0] - src_points[3]),  # Левая сторона  
            np.linalg.norm(src_points[1] - src_points[2])   # Правая сторона
        )
        
        return width, height
    
    def create_homography_transform(self, frame, centers):
        """Создает гомографию для преобразования плоскости"""
        if len(centers) != 4:
            return None, None, None
        
        # Сортируем точки
        src_points = np.float32(self.sort_points(centers))
        if src_points is None:
            return None, None, None
        
        # Вычисляем ширину и высоту
        width, height = self.calculate_plane_dimensions(src_points)
        
        # Целевые точки для прямоугольника
        dst_points = np.float32([
            [0, 0],
            [width, 0], 
            [width, height],
            [0, height]
        ])
        
        # Вычисляем матрицу гомографии
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Применяем перспективное преобразование
        warped = cv2.warpPerspective(frame, H, (int(width), int(height)))
        
        return warped, H, src_points
    
    def draw_plane(self, frame, centers, src_points):
        """Рисует плоскость и маркеры на кадре"""
        output_frame = frame.copy()
        
        if len(centers) == 4 and src_points is not None:
            # Рисуем контур плоскости
            pts = src_points.astype(int)
            cv2.polylines(output_frame, [pts], True, self.colors['contour'], 3)
            
            # Рисуем заливку плоскости с прозрачностью
            overlay = output_frame.copy()
            cv2.fillPoly(overlay, [pts], self.colors['fill'])
            cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0, output_frame)
            
            # Рисуем угловые точки
            for i, (point, color) in enumerate(zip(pts, self.colors['corners'])):
                cv2.circle(output_frame, tuple(point), 3, color, -1)
        
        return output_frame
    
    def process_frame(self, frame):
        """Обрабатывает один кадр: детектирует маркеры и строит плоскость"""
        # Детекция маркеров
        output_frame, corners, centers = self.detect_markers(frame)
        
        warped_image = None
        homography_matrix = None
        src_points = None
        
        # Если найдено 4 маркера
        if len(centers) == 4:
            # Создаем гомографию
            warped_image, homography_matrix, src_points = self.create_homography_transform(frame, centers)
            
            # Рисуем плоскость
            output_frame = self.draw_plane(output_frame, centers, src_points)
        
        return output_frame, warped_image, homography_matrix, centers, src_points


class ContourDetector:
    def __init__(self):
        self.area_threshold = 500
    
    def is_contour_inside_plane(self, contour, plane_points):
        """Проверяет, полностью ли контур находится внутри плоскости"""
        if plane_points is None or len(plane_points) != 4:
            return False
        
        # Преобразуем plane_points в правильный формат для OpenCV
        plane_contour = np.array(plane_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Преобразуем контур в массив точек
        contour_points = contour.reshape(-1, 2)
        
        # Проверяем каждую точку контура
        for point in contour_points:
            # Проверяем, находится ли точка внутри четырехугольника
            result = cv2.pointPolygonTest(plane_contour, tuple(point.astype(float)), False)
            if result < 0:
                return False
        return True
    
    def process_frame(self, frame, plane_points=None):
        """Обрабатывает кадр и рисует контуры только внутри плоскости"""
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(img_blur, 155, 200, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = frame.copy()
        
        # Фильтруем контуры по площади и расположению внутри плоскости
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                # Если задана плоскость, проверяем находится ли контур внутри
                if plane_points is not None:
                    if self.is_contour_inside_plane(contour, plane_points):
                        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Если плоскость не задана, рисуем все контуры
                    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return result


def main():
    """Основная функция для демонстрации работы классов"""
    detector = ArUcoPlaneDetector()
    contour_detector = ContourDetector()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обрабатываем кадр для детекции маркеров и плоскости
        output_frame, warped_image, homography, centers, plane_points = detector.process_frame(frame)
        
        # Обрабатываем кадр для детекции контуров (передаем точки плоскости)
        result = contour_detector.process_frame(output_frame, plane_points)
        
        cv2.imshow('Contours in Plane', result)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()