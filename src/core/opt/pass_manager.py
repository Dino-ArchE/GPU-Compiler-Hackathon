"""Pass infrastructure: registries, the analysis cache, and the PassManager.

Adding a pass = one file in passes/ with one class:

    @register_pass
    class MyPass(FunctionPass):
        name = "my-pass"
        preserves = CFG_ANALYSES          # what stays valid when this pass changes something

        def run(self, function, am):      # am: AnalysisManager, am.get("defuse") etc.
            ...
            return changed                # True if the function was modified

Analyses are the same shape (`Analysis`, `@register_analysis`), but `run` returns a result
object that the AnalysisManager caches until a pass that does not preserve it changes the code.
"""

import sys
import time

PASSES = {}      # name -> FunctionPass subclass
ANALYSES = {}    # name -> Analysis subclass

# analyses that depend only on the block graph, not on the instructions inside blocks
CFG_ANALYSES = frozenset({"cfg", "dominators", "loops"})
PRESERVE_ALL = frozenset({"*"})
PRESERVE_NONE = frozenset()


def register_pass(cls):
    if not cls.name:
        raise TypeError(f"{cls.__name__} needs a `name`")
    if cls.name in PASSES:
        raise ValueError(f"pass {cls.name!r} registered twice ({PASSES[cls.name].__name__}, {cls.__name__})")
    PASSES[cls.name] = cls
    return cls


def register_analysis(cls):
    if not cls.name:
        raise TypeError(f"{cls.__name__} needs a `name`")
    if cls.name in ANALYSES:
        raise ValueError(f"analysis {cls.name!r} registered twice")
    ANALYSES[cls.name] = cls
    return cls


class Analysis:
    """Computes facts about a Function. Must not modify it."""

    name = None

    def run(self, function, am):
        raise NotImplementedError


class FunctionPass:
    """Transforms one Function in place and reports whether it changed anything."""

    name = None
    preserves = PRESERVE_NONE    # analysis names still valid after a change ("*" = all)

    def run(self, function, am):
        raise NotImplementedError


class AnalysisManager:
    """Per-function cache of analysis results. `module` gives passes the module-level state
    (shared array declarations); it may be None for a function analysed on its own."""

    def __init__(self, function, module=None):
        self.function = function
        self.module = module
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            if name not in ANALYSES:
                raise KeyError(f"unknown analysis {name!r}; known: {sorted(ANALYSES)}")
            self.cache[name] = ANALYSES[name]().run(self.function, self)
        return self.cache[name]

    def invalidate(self, preserved=PRESERVE_NONE):
        if "*" in preserved:
            return
        self.cache = {k: v for k, v in self.cache.items() if k in preserved}


class Repeat:
    """A group of passes run again and again until a whole round changes nothing."""

    def __init__(self, passes, max_iterations=10):
        self.passes = list(passes)
        self.max_iterations = max_iterations

    def __repr__(self):
        return f"Repeat({self.passes})"


class PassStats:
    def __init__(self):
        self.runs = 0
        self.changes = 0
        self.removed = 0      # net instructions removed (negative = added)
        self.seconds = 0.0


class PassManager:
    """Runs a pipeline (pass names and Repeat groups) over every function of a module.

    verify:      run the IR verifier after each pass that changed something (catches pass bugs
                 at the pass that caused them instead of as wrong PTX)
    print_after: pass names (or "all") whose result is dumped to stderr, for diffing
    """

    def __init__(self, pipeline, verify=True, print_after=(), stats=False):
        self.pipeline = list(pipeline)
        self.verify = verify
        self.print_after = set(print_after)
        self.stats = {} if stats else None
        self._check(self.pipeline)

    def _check(self, pipeline):
        for item in pipeline:
            if isinstance(item, Repeat):
                self._check(item.passes)
            elif item not in PASSES:
                raise KeyError(f"unknown pass {item!r}; known: {sorted(PASSES)}")

    def run(self, module):
        changed = False
        for function in module.functions:
            am = AnalysisManager(function, module)
            changed |= self._run_items(self.pipeline, function, am)
        return changed

    def _run_items(self, items, function, am):
        changed = False
        for item in items:
            if isinstance(item, Repeat):
                for _ in range(item.max_iterations):
                    if not self._run_items(item.passes, function, am):
                        break
                    changed = True
            else:
                changed |= self._run_pass(item, function, am)
        return changed

    def _run_pass(self, name, function, am):
        p = PASSES[name]()
        before = function.num_instructions()
        start = time.perf_counter()
        changed = bool(p.run(function, am))
        elapsed = time.perf_counter() - start
        if changed:
            am.invalidate(p.preserves)
            if self.verify:
                from .verifier import verify
                verify(function, after=name)
        if self.stats is not None:
            s = self.stats.setdefault(name, PassStats())
            s.runs += 1
            s.changes += changed
            s.removed += before - function.num_instructions()
            s.seconds += elapsed
        if changed and (name in self.print_after or "all" in self.print_after):
            from core.ptx.printer import print_function
            print(f"// ----- after {name} -----\n{print_function(function)}", file=sys.stderr)
        return changed

    def report(self, file=sys.stderr):
        if not self.stats:
            return
        print(f"// {'pass':<22}{'runs':>5}{'changed':>9}{'removed':>9}{'ms':>8}", file=file)
        for name, s in self.stats.items():
            print(f"// {name:<22}{s.runs:>5}{s.changes:>9}{s.removed:>9}{s.seconds * 1e3:>8.2f}", file=file)
