from .value import Symbol


class Module:
    """A PTX file: the header fields and its functions (always one kernel in this project)."""

    def __init__(self, version="8.0", target="sm_89", address_size=64):
        self.version = version
        self.target = target
        self.address_size = address_size
        self.shared = []  # (Symbol, align, nbytes)
        self.functions = []

    def add_function(self, function):
        self.functions.append(function)
        return function

    def add_shared(self, name, align, nbytes):
        sym = Symbol(name)
        self.shared.append((sym, align, nbytes))
        return sym

    def __str__(self):
        return f".version {self.version}\n.target {self.target}\n.address_size {self.address_size}"