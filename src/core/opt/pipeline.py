"""Named pipelines. -O0 is codegen output untouched; each level adds passes on top.

Every pipeline starts with split-blocks: the analyses need a bra/ret to end its block.
"""

from .pass_manager import Repeat

# the scalar clean-up loop: each pass exposes work for the others, so run them to a fixed point
SCALAR = Repeat(["const-fold", "instcombine", "copy-prop", "cse", "dce"])

PIPELINES = {
    0: [],
    # -O1: the fundamental clean-ups
    1: ["split-blocks", SCALAR],
    # -O2: + passes using facts ptxas cannot see (launch geometry, read-only buffers, alignment)
    2: ["split-blocks", SCALAR,
        "addr-fold", SCALAR,
        "ld-nc"],
}


def build_pipeline(level=None, passes=None):
    """A pipeline from an -O level, or from an explicit list of pass names (always canonicalized)."""
    if passes is not None:
        names = [p for p in passes if p]
        return names if not names or names[0] == "split-blocks" else ["split-blocks", *names]
    return list(PIPELINES[level])


__all__ = ["PIPELINES", "Repeat", "build_pipeline"]
