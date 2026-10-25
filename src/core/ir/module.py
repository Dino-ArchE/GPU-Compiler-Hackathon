from .value import Symbol


class Module:
    """A PTX file: the header fields and its functions (always one kernel in this project)."""

    def __init__(self, version="8.0", target="sm_89", address_size=64):
        self.version = version
        self.target = target
        self.address_size = address_size
        self.shared = []  # (Symbol, align, nbytes); nbytes None = .extern (sized at launch)
        self.functions = []

    def add_function(self, function):
        self.functions.append(function)
        return function

    def add_shared(self, name, align, nbytes=None):
        sym = Symbol(name)
        self.shared.append((sym, align, nbytes))
        return sym

    def __str__(self):
        """The whole PTX file. Shared arrays are module-scope globals, declared ahead of the
        entries that use them."""
        out = [
            f".version {self.version}",
            f".target {self.target}",
            f".address_size {self.address_size}",
        ]
        if self.shared:
            out.append("")
            for sym, align, nbytes in self.shared:
                if nbytes is None:   # dynamic shared memory: size comes from the launch
                    out.append(f".extern .shared .align {align} .b8 {sym}[];")
                else:
                    out.append(f".shared .align {align} .b8 {sym}[{nbytes}];")
        for function in self.functions:
            out += ["", str(function)]
        text = "\n".join(out) + "\n"
        if not text.isascii():
            bad = sorted({c for c in text if not c.isascii()})
            raise ValueError(f"PTX must be ASCII; found {bad}")
        return text