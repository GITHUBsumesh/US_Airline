import sys
from ml.src.utils.path_config import ML_ROOT
from pathlib import Path
from src.logging.logger import logging

class AirLineException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message

        _, _, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = str(Path(exc_tb.tb_frame.f_code.co_filename).resolve())
        else:
            self.lineno = None
            self.file_name = "Unknown File"

    def __str__(self):
        s=f"Error occured in python script name [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}] "
        logging.error(s)
        return (
            f"Error occured in python script name [{self.file_name}] "
            f"line number [{self.lineno}] error message [{self.error_message}]"
        )