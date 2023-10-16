"""Module to calculate crystal data."""
import sys
from fractions import Fraction as Fr
import numpy as np
from cctbx  import sgtbx
import periodictable as ptbl
from scipy import integrate
import scipy.misc
from elementy import PeriodicTable
# pylint: disable=line-too-long
#pylint: disable=C0103
#pylint: disable=C3001
#pylint: disable=C0415
#pylint: disable=R0913
#pylint: disable=R0902
#pylint: disable=R0903
#pylint: disable=C0103
#pylint: disable=C0200
#pylint: disable=R0914
def interlog(xtemp):
    """
    return: float
    """
    xcal = lambda x: x / (np.exp(-x) - 1)
    xcal2 = integrate.quad(xcal, xtemp, 0)
    return xcal2[0] / xtemp
def bernoulli(n):
    """ Calculates Bernoulli number.
        n: interger
        return: Fraction
    """
    A = [0] * (n + 1)
    for m in range(n + 1):
        A[m] = Fr(1, m + 1)
        for j in range(m, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
    return A[0]
def rfunc(x):
    """ return : float """
    term = 21
    rarr = np.ndarray(shape=(term, 1), dtype=float)
    for n in range(term):
        rarr[n] = (bernoulli(n) * np.power(x, (n - 1)) / (scipy.special.factorial(n) * (n + 5 / 2))) + bernoulli(
            22) * np.power(x, 21) / (scipy.special.factorial(22) * (22 + 5 / 2))
    rval = np.ndarray.sum(rarr)
    return rval


def cal_energy(wvls):
    """ Convert wsvelenghts into energy 
    wvls: 1d array  of wavelengths
    return: 1d array of enegy in mev """
    energ = .081820 / (wvls* wvls)
    return energ
class CrystalData:
    """Class calcualtes crystal data parameters"""
    def __init__(self, elm):
        """
        elm : path to cif file 
        """
        if sys.version_info[0] < 3:
            from diffpy.Structure.Parsers import getParser
        else:
            from diffpy.structure.parsers import getParser
        self.p = getParser('cif')
        self.pf = self.p.parseFile(elm)
        self.neutronics = self.Neutronics(self.pf.element[0])
        self.nxs = self.NxsCalculate(self.neutronics)
    def latticeparm(self):
        """ Return a 6 columns array of lattice values """
        return np.array([[self.pf.lattice.a,self.pf.lattice.b,self.pf.lattice.c,self.pf.lattice.alpha,self.pf.lattice.beta,self.pf.lattice.gamma]])
    def atmposition(self):
        """ Return a 3D array of Atomic position x,y,z """
        return np.array([self.pf.x,self.pf.y,self.pf.z]).T
    def hklmaker(self,amax,cutval):
        """ return a 3d hkl array
        amax : maximum hkl value to consider in array 
        cutval: implement selection value for hkl array base  onH*H + K*K +L*L 
        """
        max_hkl=amax
        min_hkl=-1*max_hkl
        sgi =sgtbx.space_group_info(self.p.spacegroup.number)
        sg=sgi.group()
        harr=[]
        karr=[]
        larr=[]
        for h in range(min_hkl,(max_hkl+1)):
            for k in range(min_hkl,(max_hkl+1)):
                for l in range(min_hkl,(max_hkl+1)):
                    if sg.is_sys_absent((h,k,l)) is not True:
                        if  h==0 and k==0 and l==0:
                            continue
                        harr.append(h)
                        karr.append(k)
                        larr.append(l)
        hkln=np.ndarray(shape=(len(harr),3), dtype=int)
        for j in range (len(harr)):
            hkln[j][0]=harr[j]
            hkln[j][1]=karr[j]
            hkln[j][2]=larr[j]
        hklnb=self.hklnbuilder(hkln,cutval)
        return hklnb
    def hklnbuilder(self,hkln,cutval):
        """ return a 3d hkl array
        max : maximum hkl value to consider in array 
        cutval: implement selection value for hkl array base  onH*H + K*K +L*L 
        """
        hklnsum=np.zeros([len(hkln)])
        for j in range (len(hkln)):
            hklnsum[j]=(np.power(hkln[j,0],2)+np.power(hkln[j,1],2)+np.power(hkln[j,2],2))
        hklna= np.where(hklnsum<=cutval)
        return hkln[hklna[0]]
    def cstructure(self,amax,cutval):
        """ Create crystal structure """
        return np.array([self.latticeparm(),self.atmposition(),self.hklmaker(amax,cutval)],dtype=object)
    class Neutronics:
        "Class of neutronics parameters"
        def __init__(self, elem):
            """
            Returns the crystal neutronics information
            elm : path to cif file 
            """
            self.pty= PeriodicTable()
            self.ns = getattr(ptbl, elem).neutron # neutron scattering lengths, cross sections etc
            self.abs=self.ns.absorption*1e-8
            self.C1=self.abs*np.sqrt(.0253)
            self.inc=self.ns.incoherent*1e-8
            self.coh=self.ns.coherent*1e-8
            self.xsbound=self.inc+self.coh
            self.b=self.ns.b_c*1e-5
            self.debye=self.pty.elements[elem]['debye_temperature']
            self.xtemp=self.debye/293.15
            self.atnum=self.pty.elements[elem]['atomic_number']
            self.C2=np.round(4.27*np.exp(self.atnum/61))
            self.Bzero=float(2873.0/(self.atnum *self.debye))
            self.Btemp=4.0*self.Bzero*float(interlog(self.xtemp))/self.xtemp
            self.xsfree=np.power(self.atnum/(self.atnum +1),2)*self.xsbound
            self.mass=self.pty.elements[elem]['mass']
            self.density=self.pty.elements[elem]['density']
            self.N_density=1e-24*float(self.density) * 6.02E23 / self.mass
            self.dat=np.array([self.b,self.abs,self.coh,self.inc,self.xsbound,self.N_density,self.mass,self.debye,self.xtemp])
    class NxsCalculate:
        """This class gather methods related to calculation of neutron scattering cross sections. 
        """

        def __init__(self, ntni):
            """
            ntni : crystal neutronic object 
            """
            self.crysdat=ntni
        def nxs_tdssp(self,wvls):
            """ Calculation of  single phonon thermal diffuse cross section, see A. Freund, Nucl. Instrum. Methods Phys. Res. 213, 495 (1983). 
                wvls: 1d array  of wavelengths
            """
            if self.crysdat.xtemp <= 6:
                rre = rfunc(self.crysdat.xtemp)
            else:
                rre = 3.3 * np.power(self.crysdat.xtemp, -3.5)
            tdssph1 = (self.crysdat.abs / (36 * self.crysdat.atnum) * np.sqrt(self.crysdat.debye) / np.sqrt((cal_energy(wvls)))) * rre
            return tdssph1
        def nxs_tdsmp(self,wvls):
            """ Calculation of  multiple phonons thermal diffuse cross section, see A. Freund, Nucl. Instrum. Methods Phys. Res. 213, 495 (1983). 
                wvls: 1d array  of wavelengths """
            tdsm = self.crysdat.xsfree* (1 - np.exp(-1 * (self.crysdat.Bzero + self.crysdat.Btemp) * self.crysdat.C2 * cal_energy(wvls)))
            return tdsm
        def nxs_absorption(self,wvls):
            """ Calculation of  absorption cross section, see A. Freund, Nucl. Instrum. Methods Phys. Res. 213, 495 (1983). 
                wvls: 1d array  of wavelengths """
            val = self.crysdat.C1 / (np.sqrt(cal_energy(wvls)))
            return val
