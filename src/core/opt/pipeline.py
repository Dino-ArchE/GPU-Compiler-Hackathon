"""Named pipelines. -O0 is codegen output untouched; each level adds passes on top.

Every pipeline starts with split-blocks: the analyses need a bra/ret to end its block.
"""

from .pass_manager import Repeat

PIPELINES = {
    0: [],
    1: ["split-blocks"],
    2: ["split-blocks"],
}


def build_pipeline(level=None, passes=None):
    """A pipeline from an -O level, or from an explicit list of pass names (always canonicalized)."""
    if passes is not None:
        names = [p for p in passes if p]
        return names if not names or names[0] == "split-blocks" else ["split-blocks", *names]
    return list(PIPELINES[level])


__all__ = ["PIPELINES", "Repeat", "build_pipeline"]
