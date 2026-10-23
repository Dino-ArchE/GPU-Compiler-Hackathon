class Module:
    """A PTX file: the header fields and its functions (always one kernel in this project)."""

    def __init__(self, version="8.0", target="sm_89", address_size=64):
        self.version = version
        self.target = target
        self.address_size = address_size
        self.functions = []

    def add_function(self, function):
        self.functions.append(function)
        return function
