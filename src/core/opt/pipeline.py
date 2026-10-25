"""Named pipelines. -O0 is codegen output untouched; each level adds passes on top.

Every pipeline starts with split-blocks: the analyses need a bra/ret to end its block.
"""

from .pass_manager import Repeat

# the scalar clean-up loop: each pass exposes work for the others, so run them to a fixed point
SCALAR = ["const-fold", "instcombine", "copy-prop", "cse", "dce", "simplify-cfg"]
# with range facts (launch bounds, alignment) deciding compares and div/rem
SCALAR_RANGE = ["range-fold", *SCALAR]

PIPELINES = {
    0: [],
    # -O1: the fundamental clean-ups
    1: ["split-blocks", Repeat(SCALAR), "licm", Repeat(SCALAR)],
    # -O2: + passes using facts ptxas cannot see (launch geometry, read-only buffers, alignment,
    # tolerance-bounded approximate math)
    2: ["split-blocks", "launch-specialize", Repeat(SCALAR_RANGE),
        "unroll", Repeat(SCALAR_RANGE), "addr-fold", Repeat(SCALAR_RANGE),
        "licm", "strength-reduce", Repeat(SCALAR), "fast-math", "copy-prop",
        "vectorize-mem", "fuse-mad", "if-convert", "simplify-cfg", "dce", "ld-nc"],
}


def build_pipeline(level=None, passes=None, disable=()):
    """A pipeline from an -O level, or from an explicit list of pass names (always canonicalized),
    without the passes named in `disable` (also inside Repeat groups; split-blocks always stays)."""
    if passes is not None:
        names = [p for p in passes if p]
        pipeline = names if not names or names[0] == "split-blocks" else ["split-blocks", *names]
    else:
        pipeline = list(PIPELINES[level])
    return _without(pipeline, set(disable) - {"split-blocks"})


def _without(pipeline, disable):
    out = []
    for item in pipeline:
        if isinstance(item, Repeat):
            out.append(Repeat(_without(item.passes, disable), item.max_iterations))
        elif item not in disable:
            out.append(item)
    return out


__all__ = ["PIPELINES", "Repeat", "build_pipeline"]
