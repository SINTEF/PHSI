"""
Errors carrying partial results and details of what failed.
"""


class SimulationError(RuntimeError):
    def __init__(self, message, *, t, x, diagnostics=None):
        super().__init__(message)
        self.t = t
        self.x = x
        self.diagnostics = diagnostics or {}


class FitError(RuntimeError):
    def __init__(self, message, *, fit_info, partial_model=None):
        super().__init__(message)
        self.fit_info = fit_info
        self.partial_model = partial_model


class SupportNonconvergence(RuntimeError):
    pass
