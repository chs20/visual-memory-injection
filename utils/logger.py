"""Logger utility for capturing console output to log files."""

import sys


class Logger:
    """Class to capture all print outputs and write to both console and log file."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message, to_file=True, to_console=True):
        message = str(message)
        if to_console:
            self.terminal.write(message + "\n")
        if to_file:
            self.log_file.write(message + "\n")
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

