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
        
        # Если найдено 4 маркера
        if len(centers) == 4:
            # Создаем гомографию
            warped_image, homography_matrix, src_points = self.create_homography_transform(frame, centers)
            
            # Рисуем плоскость
            output_frame = self.draw_plane(output_frame, centers, src_points)
        
        return output_frame, warped_image, homography_matrix, centers

class ObjDetection:
    def __init__(self):
        pass

    def detect_objects(self, current_frame, background_frame, min_area=1000):
        # Размытие для уменьшения шума
        self.current_blur = cv2.GaussianBlur(current_frame, (5,5), 0)
        self.background_blur = cv2.GaussianBlur(background_frame, (5,5), 0)
        
        # Разность кадров
        self.diff = cv2.absdiff(self.background_blur, self.current_blur)
        
        # Бинаризация
        self._, self.thresh = cv2.threshold(self.diff, 25, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции
        self.kernel = np.ones((5,5), np.uint8)
        self.thresh = cv2.dilate(self.thresh, self.kernel, iterations=2)
    
        return self.thresh

def main():
    """Основная функция для демонстрации работы класса"""

    cap = cv2.VideoCapture(0)

    ret, background = cap.read()

    # Создаем детектор
    detector = ArUcoPlaneDetector()
    obj=ObjDetection()
    # Инициализируем видеопоток
    

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        # Обрабатываем кадр
        output_frame, warped_image, homography, centers = detector.process_frame(frame)
        

        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        mask = obj.detect_objects(gray, cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
            
                cv2.drawContours(output_frame, [contour], -1, (0,255,0), 2)
    
        cv2.imshow('ArUco Markers Detection', output_frame)        

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()