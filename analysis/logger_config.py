import logging


class ANSIColorFormatter(logging.Formatter):
    """Форматтер с ANSI-цветами для терминала"""

    WHITE = '\033[97m'
    GREY = '\033[37m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def format(self, record):

        color = self.WHITE
        if record.levelno >= logging.CRITICAL:
            color = self.RED
        elif record.levelno >= logging.ERROR:
            color = self.RED
        elif record.levelno >= logging.WARNING:
            color = self.YELLOW
        elif record.levelno <= logging.DEBUG:
            color = self.GREY

        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def setup_logging():
    """Настройка логирования. Необходим вызов функции в начале выполнения программы"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setFormatter(ANSIColorFormatter(logging.BASIC_FORMAT))

    logger.addHandler(console)