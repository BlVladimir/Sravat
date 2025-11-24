# Sravat

## Как использовать requirements.txt:
1. pip install -r requirements.txt - загружает все модули из файла
2. pip freeze > requirements.txt - дозапись еще не записанных модулей

**После любого загруженного через pip модуля ЗАПИСЫВАЙТЕ ЕГО В ФАЙЛ!!!**

## Как создать venv:
1. python -m venv venv - создание виртуального окружения
2. venv\Scripts\activate - активация (Windows)

```mermaid
classDiagram
    class State {
    <<dataclass>>
    +method: Method
    +centers: list
    +src_points: list
    +current_frame: np.ndarray
    }
    
    class Function {
        <<abstract>>
        #logger
        #state: State
        +__call__()*
    }
    
    
    class MainAnalysisStrategy {
        -logger
        -state: State
        -_transition: dict
        -to_cv2(base64_string) np.ndarray
        -to_base64(image) str
    }
    
    class FacadeAnalysis {
        -strategy: AnalysisStrategyInterface
        +analyze_frame(base64_input) str
    }
    
    class AnalysisStrategyInterface {
        <<interface>>
        +__call__(base64_input) str*
    }
    
    AnalysisStrategyInterface <|.. MainAnalysisStrategy
    
    %% Композиция / Агрегация
    MainAnalysisStrategy *-- State
    
    FacadeAnalysis o-- AnalysisStrategyInterface
