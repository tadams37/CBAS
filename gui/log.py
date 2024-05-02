# log.py - Logging
import logging


class AppendFileLineToLog(logging.Filter):
    """Custom logging format"""
    def filter(_, record):
        record.filename_lineno = "%s:%d" % (record.filename, record.lineno)
        return True


class NotebookLoggingHandler(logging.Handler):
    """Format log entries and make them appear in Jupyter Lab's log output"""

    def __init__(self, log_level):
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter('%(message)s (%(filename_lineno)s)'))
        self.setLevel(log_level)
        self.log_output_widget = None

    def start_log_output(self, widget):
        self.log_output_widget = widget

    def stop_log_output(self):
        self.log_output_widget = None

    def emit(self, message):
        """Write message to log"""
        if not self.log_output_widget is None:

            with self.log_output_widget:
                print(self.format(message))

log = logging.getLogger("")  # Get handle on root logger
log_handler = NotebookLoggingHandler(logging.INFO)
log.addHandler(log_handler)
log.addFilter(AppendFileLineToLog())
log.setLevel(logging.INFO)
