from typing import Protocol

class AnalysisStrategyInterface(Protocol):
    """Интерфейс стратегии обработки картинки. Использовать в проверках на тип"""
    def __call__(self, data:str)->str:
        """
            Использует объект как функцию для обработки.
            data - строковое (временно) представление Base64.
            Возвращает строковое (временно) представление Base64.
        """
        pass

class EmptyAnalysisStrategy(AnalysisStrategyInterface):
    """Никак не обрабатывает картинку"""
    def __call__(self, data:str)->str:
        return data