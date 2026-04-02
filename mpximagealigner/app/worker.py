import sys
import io
import traceback

from PyQt6.QtCore import QThread, pyqtSignal


class _Redirector(io.TextIOBase):
    """Forwards all write() calls to a Qt signal so output appears in the GUI log."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, text):
        if text:
            self._signal.emit(str(text))
        return len(text)

    def flush(self):
        pass


class AlignmentWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        redirector = _Redirector(self.log)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector
        try:
            from mpximagealigner import alignment
            alignment.run_alignment(
                **self.params,
                cancelled=lambda: self._cancel_requested,
            )
            self.finished.emit(True, "Alignment completed successfully.")
        except Exception:
            self.finished.emit(False, traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
