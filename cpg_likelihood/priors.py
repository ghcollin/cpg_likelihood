import numpy as np


class Prior(object):
    def __init__(self, name, display):
        self.name, self._display = name, display
    
    def xform(self, theta):
        raise NotImplementedError()

    def id(self):
        return self.name

    def display(self):
        return self._display

    def display_value(self, theta):
        return self.xform(theta)


class Log10UniformPrior(Prior):
    def __init__(self, start, stop, name, display, display_log=False):
        super().__init__(name, display)
        self.start, self.delta = start, stop - start
        self.display_log = display_log

    def display_value(self, theta):
        return (self.start + theta*self.delta) if self.display_log else self.xform(theta)

    def xform(self, theta):
        return 10**(self.start + theta*self.delta)


class LinearUniformPrior(Prior):
    def __init__(self, start, stop, name, display):
        super().__init__(name, display)
        self.start, self.delta = start, stop - start

    def xform(self, theta):
        return self.start + theta*self.delta


class UnitLinearUniformPrior(Prior):
    def __init__(self, name, display):
        super().__init__(name, display)

    def xform(self, theta):
        return theta


class TanUniformPrior(Prior):
    def __init__(self, start_slope, stop_slope, name, display, display_angle=False):
        super().__init__(name, display)
        self.start, self.delta = np.arctan(start_slope), np.arctan(stop_slope) - np.arctan(start_slope)
        self.display_angle = display_angle

    def display_value(self, theta):
        return (self.start + theta*self.delta) if self.display_angle else self.xform(theta)

    def xform(self, theta):
        return np.tan(self.start + theta*self.delta)