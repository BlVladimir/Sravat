from analysis.analysis_strategy import MainAnalysisStrategy
from logger_config import setup_logging


class RunTime:
    """Замена сайта в окне"""
    obj:'RunTime' = None

    def __init__(self):
        self.obj = self
        setup_logging()
        self.analise_strategy = MainAnalysisStrategy()


    def __call__(self):
        while True:
            if not self.analise_strategy('kk'):
                break



if __name__ == '__main__':
    runtime = RunTime()
    runtime()