# safe_io.py
"""Defensive I/O stream protection.
Prevents OSError: [Errno 5] Input/output error when standard I/O streams point to closed/detached terminals.
"""

import sys

class SafeStream:
    """Wraps a text stream to gracefully suppress EIO (Errno 5) and BrokenPipeError."""
    def __init__(self, target):
        self._target = target

    def write(self, s):
        try:
            if self._target:
                return self._target.write(s)
        except (OSError, BrokenPipeError):
            pass

    def flush(self):
        try:
            if self._target:
                return self._target.flush()
        except (OSError, BrokenPipeError):
            pass

    def isatty(self):
        try:
            return self._target.isatty() if self._target else False
        except Exception:
            return False

    def fileno(self):
        try:
            return self._target.fileno()
        except Exception:
            raise OSError(5, "Input/output error suppressed")

    def __getattr__(self, attr):
        return getattr(self._target, attr)

if sys.stdout and not isinstance(sys.stdout, SafeStream):
    sys.stdout = SafeStream(sys.stdout)
if sys.stderr and not isinstance(sys.stderr, SafeStream):
    sys.stderr = SafeStream(sys.stderr)
