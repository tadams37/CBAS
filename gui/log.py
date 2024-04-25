# log.py - Logging
import logging
import ipywidgets as widgets


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
        self.log_output_widget = widgets.Output()

    def emit(self, message):
        """Write message to log"""
        with self.log_output_widget:
            print(self.format(message))

log = logging.getLogger(__name__)
log_handler = NotebookLoggingHandler(logging.INFO)
log.addHandler(log_handler)
log.addFilter(AppendFileLineToLog())
log.setLevel(logging.INFO)
