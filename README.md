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
    direction LR

    class Environment {
        <<dataclass>>
        +detector
        +aruco_dict
        +aruco_params
        +centers: list
        +src_points: list
        +current_frame: np.ndarray?
    }

    class Function {
        <<abstract>>
        +logger
        +env: Environment
    }

    class DetectMarkers {
        +env: Environment
    }

    class CreateHomographyTransform {
        +env: Environment
    }

    class DrawPlane {
        +env: Environment
    }

    class MainAnalysisStrategy {
        +env: Environment
        +detect_markers: DetectMarkers
        +create_homography_transform: CreateHomographyTransform
        +draw_plane: DrawPlane
    }

    class RunTime {
        +logger
        +analise_strategy: MainAnalysisStrategy
        +cap
    }

    class ANSIColorFormatter {
        +format(record)
    }

    %% Inheritance (project classes only)
    Function <|-- DetectMarkers
    Function <|-- CreateHomographyTransform
    Function <|-- DrawPlane

    %% Composition / Aggregation
    MainAnalysisStrategy *-- Environment
    MainAnalysisStrategy o-- DetectMarkers
    MainAnalysisStrategy o-- CreateHomographyTransform
    MainAnalysisStrategy o-- DrawPlane

    RunTime o-- MainAnalysisStrategy
