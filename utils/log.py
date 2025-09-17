import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from colorlog import ColoredFormatter
from datetime import datetime


# 控制台日志处理器
def create_log_console_handler():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # 设置日志输出格式和颜色
    log_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    color_format = ColoredFormatter(
        log_format,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )
    console_handler.setFormatter(color_format)
    return console_handler


def create_file_handler(logfile_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 示例输出: 20231015_143045
    # log_filename = f"log/{timestamp}_{logfile_name}.log"
    # 创建一个handler，每个文件最大3MB
    file_handler = RotatingFileHandler(filename=logfile_path,
                                       backupCount=5,
                                       maxBytes=3 * 1024 * 1024,
                                       encoding='utf-8')
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    return file_handler



solar_logger = logging.getLogger('solar')
solar_logger.level = logging.DEBUG

# 测试日志
solar_logger.info("程序启动")
