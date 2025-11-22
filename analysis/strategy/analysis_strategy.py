from typing import Protocol


class AnalysisStrategyInterface(Protocol):
    """Интерфейс стратегии обработки картинки. Использовать в проверках на тип"""
    def __call__(self, base64_input:str)->str:
        """
        Использует объект как функцию для обработки.
        base64_input - строковое представление Base64.
        Возвращает строковое (временно) представление Base64.
        """
        pass

class EmptyAnalysisStrategy:
    """Никак не обрабатывает картинку. Для теста браузера"""
    def __call__(self, base64_input:str)->str:
        return base64_input