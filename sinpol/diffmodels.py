"""Module to calculate different diffraction intensity."""
import numpy as np
from numba import float64
from numba.experimental import jitclass
#pylint: disable=R0913
def coth(x):
    """Calculates hyperbolic cotangent."""
    return 1. / np.tanh(x)
scdata=[('r',float64[:]),('q',float64[:]),('p',float64[:]),('s',float64[:]),('b',float64[:])]
@jitclass(scdata)
class DiffModels:
    """Class different diffraction models"""
    def __init__(self,r,q,p,s,b):
        """r,q,p,s,b are auxialiary parameters use to calculate the intensity.
        """
        self.r=r
        self.q=q
        self.p=p
        self.s=s
        self.b=b
    def sears_lr(self):
        """Calculates the reflected intensity for the Laue case using the Sears models, 
        see reference V. Sears, Acta Crystallogr., Sect. A: Found. Crystallogr. 53, 35 (1997).
         Return: 1d array
        """
        t=self.r*np.exp(-self.q)/(self.r*np.cosh(self.r)+self.p*np.sinh(self.r))
        return t
    def sears_lt(self):
        """Calculates the transmitted intensity for the Laue case using the Sears models, 
        see reference V. Sears, Acta Crystallogr., Sect. A: Found. Crystallogr. 53, 35 (1997).
        Return: 1d array
        """
        t=np.exp(-self.p)*(np.cosh(self.s)-(self.q/self.s)*np.sinh(self.s))
        return t
    def sears_br(self):
        """Calculates the reflection intensity for the Bragg case using the Sears models, 
        see reference V. Sears, Acta Crystallogr., Sect. A: Found. Crystallogr. 53, 35 (1997).
         Return: 1d array
        """
        rr=self.b/(self.r*coth(self.r)+self.p)
        return rr
    def sears_bt(self):
        """Calculates the transmitted intensity for the Bragg case using the Sears models, 
         see reference V. Sears, Acta Crystallogr., Sect. A: Found. Crystallogr. 53, 35 (1997).
         Return: 1d array
        """
        rr=np.exp(-self.p)*self.b*np.sinh(self.s)/self.s
        return rr
    