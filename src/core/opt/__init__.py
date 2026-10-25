"""The optimizer: analyses and passes over our IR, driven by a PassManager.

Importing this package registers every pass in passes/ and every analysis in analysis/:
adding one is dropping a file there with a @register_pass / @register_analysis class.
"""

import importlib
import pkgutil

from . import analysis, passes
from .pass_manager import (
    ANALYSES, CFG_ANALYSES, PASSES, PRESERVE_ALL, PRESERVE_NONE, Analysis, AnalysisManager,
    FunctionPass, PassManager, Repeat, register_analysis, register_pass,
)
from .pipeline import PIPELINES, build_pipeline
from .verifier import VerifyError, verify

for _pkg in (analysis, passes):
    for _mod in pkgutil.iter_modules(_pkg.__path__):
        importlib.import_module(f"{_pkg.__name__}.{_mod.name}")

__all__ = [
    "ANALYSES", "Analysis", "AnalysisManager", "CFG_ANALYSES", "FunctionPass", "PASSES",
    "PIPELINES", "PRESERVE_ALL", "PRESERVE_NONE", "PassManager", "Repeat", "VerifyError",
    "build_pipeline", "register_analysis", "register_pass", "verify",
]
