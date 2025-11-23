class Config:
    colors = {
        'contour': (255, 0, 0),  # Синий контур
        'fill': (0, 255, 255),  # Желтая заливка
        'center': (0, 255, 0),  # Зеленый центр
        'corners': [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Угловые точки
    }

    min_area = 100
    approximation_epsilon = 0.008