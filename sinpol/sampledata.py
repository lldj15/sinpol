""" Module to calculate sample distirbution """
from math import log, floor, ceil, fmod
import random
from scipy.integrate import quad
from mpmath import erfinv
import numpy as np
import matplotlib.pyplot as plt
#pylint: disable=C0103
#pylint: disable=C0200
#pylint: disable=W0612
#pylint: disable=W0613
#pylint: disable=R0913
#pylint: disable=R0902
#pylint: disable=R0914
#pylint: disable=R0915
# pylint: disable=line-too-long
def bunge(odf,tau):
    """
        odf : 3d-matrix of orientation distribution angles 
        tau: azimuthal angle
    """
    rho=odf[:,2]
    psi=odf[:,1]
    phi=odf[:,0]
    zerom=np.zeros((len(odf)))
    onem=np.ones((len(odf)))
    tau=np.radians(tau)
    rot_zpa=np.zeros([len(odf),3,3])
    rot_x=np.zeros([len(odf),3,3])
    rot_z=np.zeros([len(odf),3,3])
    bun=np.zeros([len(odf),3,3])
    rot_zpa[:,0,0]=np.cos(phi)
    rot_zpa[:,0,1]=np.sin(phi)
    rot_zpa[:,0,2]=zerom
    rot_zpa[:,1,0]=-np.sin(phi)
    rot_zpa[:,1,1]=np.cos(phi)
    rot_zpa[:,1,2]=zerom
    rot_zpa[:,2,0]=zerom
    rot_zpa[:,2,1]=zerom
    rot_zpa[:,2,2]=onem
    rot_z[:,0,0]=np.cos(rho)
    rot_z[:,0,1]=np.sin(rho)
    rot_z[:,0,2]=zerom
    rot_z[:,1,0]=-np.sin(rho)
    rot_z[:,1,1]=np.cos(rho)
    rot_z[:,1,2]=zerom
    rot_z[:,2,0]=zerom
    rot_z[:,2,1]=zerom
    rot_z[:,2,2]=onem
    rot_x[:,0,0]=onem
    rot_x[:,0,1]=zerom
    rot_x[:,0,2]=zerom
    rot_x[:,1,0]=zerom
    rot_x[:,1,1]=np.cos(psi)
    rot_x[:,1,2]=np.sin(psi)
    rot_x[:,2,0]=zerom
    rot_x[:,2,1]=-np.sin(psi)
    rot_x[:,2,2]=np.cos(psi)
    rot_s=np.array([[np.round(np.cos(tau),15),np.round(np.sin(tau),15),0],[-np.round(np.sin(tau),15),np.round(np.cos(tau),15),0],[0,0,1]])
    for i in range(len(odf)):
        bun[i]=np.dot(np.dot(np.dot(rot_z[i],rot_x[i]),rot_zpa[i]),rot_s)
    return bun
def indexomatirx(tr,tau):
    """
        tr : matrix of  crystal index direction
       
    """
    tau=np.radians(tau)
    MM=(np.power(tr[0],2)+np.power(tr[1],2)+np.power(tr[2],2))
    M=np.sqrt(MM)
    NN=(np.power(tr[3],2)+np.power(tr[4],2)+np.power(tr[5],2))
    N=np.sqrt(NN)
    t11=tr[3]/N
    t12=(tr[1]*tr[5]-tr[2]*tr[4])/(M*N)
    t13=tr[0]/M
    t21=tr[4]/N
    t22=(tr[2]*tr[3]-tr[0]*tr[5])/(M*N)
    t23=tr[1]/M
    t31=tr[5]/N
    t32=(tr[0]*tr[4]-tr[1]*tr[3])/(M*N)
    t33=tr[2]/M
    tex=np.array([[t11,t12,t13],[t21,t22,t23],[t31,t32,t33]])
    rot_s=np.array([[np.round(np.cos(tau),15),np.round(np.sin(tau),15),0],[-np.round(np.sin(tau),15),np.round(np.cos(tau),15),0],[0,0,1]])
    orn=np.dot(tex,rot_s)
    orn=orn[np.newaxis,:,:]
    return orn
class SampleData:
    """Class calculates sample data parameters"""
    def __init__(self,omg,tau, ptcm,grainsize,numg,modev):
        """constructor
        Parameters
        ----------
        tau :  azimuthal angle in laboratory frame
        omg :  vertical angle  in laboratory frame
        ptcm  : plate thickness in  cm 
        grainm  : grain size in microns
        ncolumn : number of discretizes columns in the sample """
        self.plate=ptcm*1e8   # convert plate from centimeters  to Angstroms
        self.grainsize=grainsize*1e4 # convert grain size from microns to angstroms
        self.Orn=int(numg) # total number of orientation
        self.cols=int(np.ceil(self.Orn*self.grainsize/self.plate))#number of column to intergrate over
        self.tau=tau
        self.omg=omg
        self.gd=self.GrainSizeDistribution(self.plate,self.grainsize,self.Orn,self.cols)
        self.mos=self.MosaicDistribution(self.Orn,modev)
        self.odf=self.CreateDistribution(self.Orn)
    def singlecrystaldeg(self,phi1,Phi,phi2,name:str,name2:str,dev):
        " Calculates orientation matrix from inputed Euler angles"
        odf=np.array([[np.radians(phi1),np.radians(Phi),np.radians(phi2)]])
        omb=bunge(odf,self.tau)
        do=f"{name}"
        mo=f"{name2}"
        gs=getattr(self.gd, do)(0)
        nu=np.radians(getattr(self.mos, mo)(dev))
        robjt=np.array([omb,np.radians(self.omg),gs,nu,self.cols],dtype=object)
        return robjt,omb
    def singlecrystalhkl(self,tr,name:str,name2:str,dev):
        """Calculates orientation matrix from inputed crystal plane and crystal direction"""
        odf=indexomatirx(tr,self.tau)
        omb=indexomatirx(tr,self.tau)#bunge(odf,self.tau)
        do=f"{name}"
        mo=f"{name2}"
        gs=getattr(self.gd, do)(0)
        nu=np.radians(getattr(self.mos, mo)(dev))
        robjt=np.array([omb,np.radians(self.omg),gs,nu,self.cols],dtype=object)
        return robjt,odf
    def polycrystalrand(self,name:str,name2:str,gdev,mdev,seed):
        "Calculate orientation matrix for a random texture polycrystal"
        omb=bunge(self.odf.random(seed),self.tau)
        do=f"{name}"
        mo=f"{name2}"
        gs=getattr(self.gd, do)(gdev)
        nu=np.radians(getattr(self.mos, mo)(mdev))
        robjt=np.array([omb,np.radians(self.omg),gs,nu,self.cols],dtype=object)
        return robjt,self.odf.random(seed)
    def polycrystalGauss(self,bv,tr,name:str,name2:str,gdev,mdev):
        "Calculate orientation matrix for a textured polycrystal"
        omb=bunge(self.odf.texturegaussian(bv,tr),self.tau)
        do=f"{name}"
        mo=f"{name2}"
        gs=getattr(self.gd, do)(gdev)
        nu=np.radians(getattr(self.mos, mo)(mdev))
        robjt=np.array([omb,np.radians(self.omg),gs,nu,self.cols],dtype=object)
        return robjt,self.odf.texturegaussian(bv,tr)
    def polycrystalMatlab(self):
        "to be done"
    class CreateDistribution:
        """ Class to create Orientation Distribution """
        def __init__(self, onumb):
            """constructor
            Parameters
            ----------
            onumb : number  of  crystal orientation"""
            self.onumb=onumb
        def halton(self,  dim ,loc):
            """  Calculates Halton distirbution  base on seed"""
            nbpts=int(self.onumb)
            h = np.empty(nbpts * dim)
            h.fill(np.nan)
            p = np.empty(nbpts)
            p.fill(np.nan)
            P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            lognbpts = log(nbpts + 1)
            for i in range(dim):
                b = P[i]
                n = int(ceil(lognbpts / log(b)))
                for t in range(n):
                    p[t] = pow(b, -(t + loc))
                for j in range(nbpts):
                    d = j + 1
                    sum_ = fmod(d, b) * p[0]
                    for t in range(1, n):
                        d = floor(d / b)
                        sum_ += fmod(d, b) * p[t]
                    h[j * dim + i] = sum_
            return h.reshape(nbpts, dim)
        def random(self,x):
            """ Create random distribution"""
            Onum=int(self.onumb)
            ys = np.array(self.halton(3, x))
            xx = (ys[:, 0] * 2 * np.pi)
            xx = np.reshape(xx, (Onum, 1))
            yy = np.arccos(2*ys[:, 1]-1)
            yy = np.reshape(yy, (Onum, 1))
            zz = (ys[:, 2] * 2 * np.pi)
            zz = np.reshape(zz, (Onum, 1))
            yss=np.hstack((xx,yy,zz))
            return yss

        def AsympBessel(self,alp, s):
            """ Asymp Bessel  """
            value= (np.exp(s)/np.power(2*np.pi*s,.5))*(1-((4*np.power(alp,2)-1)/(8*s))+((4*np.power(alp,2)-1)*(4*np.power(alp,2)-9))/(2*np.power(8*s,2))-((4*np.power(alp,2)-1)*(4*np.power(alp,2)-9)*(4*np.power(alp,2)-25))/(6*np.power(8*s,3)))
            return value
        def Matthies(self,z,bb):
            """  Texture Calculation """
            b=bb
            val=np.exp(-np.log(2)*np.power(z,2))*np.power(z,2)/(np.power(1-np.power(np.sin(.25*b)*z,2),.5))
            return val
        def Indexfinder(self,a,value):
            """ Finder """
            b=a[a>value].min()
            c=a[a<value].max()
            minidx=np.where(a==c)
            maxidx=np.where(a==b)
            return minidx[0][0],maxidx[0][0]
        def texturegaussian(self,bv,tr):
            """Create orientation distribution function of base on a gaussian distribution
              :param bv: std deviation of.
              :type bv: :class:`float`
              :param tr: 1*6 array represent hkl plane and uvw direction
              :returns: :3d array -- orientation distribution"""
            orn=self.onumb
            b = np.radians(bv)
            S = np.log(2.0) / (2.0 * np.power(np.sin(b / 4.0), 2))
            wa=np.arange(0,181,.01)
            wr=wa/57.3
            wr=wr.reshape(len(wa),1)
            z=np.zeros([len(wr),1])
            z=np.sin(.5*wr)/np.sin(.25*b)
            warr=np.zeros([len(z),1])
            nms=1/(self.AsympBessel(0, S)-self.AsympBessel(1, S))
            for i in range(len(z)):
                intg,errint=quad(self.Matthies,0,z[i],b)
                warr[i]=nms*np.exp(S)*np.power(2*np.sin(.25*b),3)*intg/(2*np.pi)
            MM=(np.power(tr[0],2)+np.power(tr[1],2)+np.power(tr[2],2))
            M=np.sqrt(MM)
            NN=(np.power(tr[3],2)+np.power(tr[4],2)+np.power(tr[5],2))
            N=np.sqrt(NN)
            t11=tr[3]/N
            t12=(tr[0]*tr[5]-tr[2]*tr[4])/(M*N)
            t13=tr[0]/M
            t21=tr[4]/N
            t22=(tr[2]*tr[3]-tr[0]*tr[5])/(M*N)
            t23=tr[1]/M
            t31=tr[5]/N
            t32=(tr[0]*tr[4]-tr[1]*tr[3])/(M*N)
            t33=tr[2]/M
            tex=np.array([[t11,t12,t13],[t21,t22,t23],[t31,t32,t33]])
            texrot=np.zeros([orn,3,3])
            for j in range (orn):
                xs=np.random.rand()
                minv,maxv=self.Indexfinder(warr,xs)
                zx=z[minv]
                w=2*np.arcsin(zx*np.sin(b/4))
                qh=np.random.rand()
                qhe=np.random.rand()
                chi =np.arccos(2*qh-1)
                eta = qhe* 2.0 * np.pi
                a11=(1-np.cos(w))*np.power(np.sin(chi),2)*np.power(np.cos(eta),2)+np.cos(w)
                a12=(1-np.cos(w))*np.power(np.sin(chi),2)*np.cos(eta)*np.sin(eta)+np.sin(w)*np.cos(chi)
                a13=(1-np.cos(w))*np.sin(chi)*np.cos(chi)*np.cos(eta)-np.sin(w)*np.sin(chi)*np.sin(eta)
                a21=(1-np.cos(w))*np.power(np.sin(chi),2)*np.cos(eta)*np.sin(eta)-np.sin(w)*np.cos(chi)
                a22=(1-np.cos(w))*np.power(np.sin(chi),2)*np.power(np.sin(eta),2)+np.cos(w)
                a23=(1-np.cos(w))*np.sin(chi)*np.cos(chi)*np.sin(eta)+np.sin(w)*np.sin(chi)*np.cos(eta)
                a31=(1-np.cos(w))*np.sin(chi)*np.cos(chi)*np.cos(eta)+np.sin(w)*np.sin(chi)*np.sin(eta)
                a32=(1-np.cos(w))*np.sin(chi)*np.cos(chi)*np.sin(eta)-np.sin(w)*np.sin(chi)*np.cos(eta)
                a33=(1-np.cos(w))*np.power(np.cos(chi),2)+np.cos(w)
                rotma=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
                bmex=np.dot(tex,rotma[:,:,0])
                texrot[j,0,0]=bmex[0,0]
                texrot[j,0,1]=bmex[0,1]
                texrot[j,0,2]=bmex[0,2]
                texrot[j,1,0]=bmex[1,0]
                texrot[j,1,1]=bmex[1,1]
                texrot[j,1,2]=bmex[1,2]
                texrot[j,2,0]=bmex[2,0]
                texrot[j,2,1]=bmex[2,1]
                texrot[j,2,2]=bmex[2,2]
            phi1=np.arctan(-texrot[:,2,0]/texrot[:,2,1])
            phi2=np.arctan(texrot[:,0,2]/texrot[:,1,2])
            PHI=np.arctan(texrot[:,2,0]/(texrot[:,2,2]*np.sin(phi1)))
            PHI= np.reshape(PHI, (orn, 1))
            phi1= np.reshape(phi1, (orn, 1))
            phi2= np.reshape(phi2, (orn, 1))
            yss = np.zeros([orn, 3])
            yss=np.hstack((phi1,PHI,phi2))
            return yss
    class GrainSizeDistribution:
        """  Class to calculate grain size distributuion"""
        def __init__(self,ptcm,grainm,orn,cols):
            """
                ptcm  : plate thickness in  cm 
                grainm  : grain size in microns
                ncolumn : number of discretizes columns in the sample """
            self.plate=ptcm   # convert plate from centimeters  to Angstroms
            self.grainsize=grainm# convert grain size from microns to angstroms
            self.cols=cols
            self.Orn=orn # total number of orientation
        def single(self,b):
            " return  single crystal thickness"
            return np.array([self.plate])
        def uniform(self,b):
            " Create grain distribution of the same size"
            return np.ones([self.Orn])*self.grainsize
        def lognormal(self,b):
            """Create grain size distribution of base on a lognormal distribution
              :param b: std deviation 
              :type b: :class:`float`
              :returns: :1d array --  grain distribution"""
            orn=self.Orn
            pt=self.plate
            tv=self.grainsize
            y=np.random.uniform(0,1,orn)
            cv=b
            ys=2*y-1
            gg=np.zeros([orn])
            for f in range(orn):
                yss=ys[f]
                gg[f]= erfinv(yss)
            d=tv*np.power((1+cv),-1/6)*np.exp((np.sqrt(2*np.log(1+cv))/3.0)*gg)
            npt=np.sum(d)
            ptr=(pt/npt)*self.cols
            return d*ptr
    class MosaicDistribution:
        """ Class to calculate mosaic distribution """
        def __init__(self,orn,mu):
            """Constructor  mosaic distribution 
               orn: number of orientation.
              :type orn : :class:`int`
              :param mu:fwhm  of mosaic
              :type mu : :class:`float`"""
            self.orn=orn
            self.mu=mu
            self.mul=(self.mu)-.5*(self.mu)
            self.muh=(self.mu)+.5*(self.mu)
        def uniform(self,dev):
            """Create mosaic distribution of base on a uniform distribution
              :param dev: std deviation of.
              :type dev: :class:`float`
              :returns: :1d array --  mosaic distribution"""
            return np.ones([self.orn])*self.mu
        def random(self,dev):
            """Create mosaic distribution of base on a random distribution
              :param dev: std deviation of.
              :type dev: :class:`float`
              :returns: :1d array --  mosaic distribution"""
            return np.random.uniform(self.mul,self.muh,self.orn)
        def gaussian(self,dev):
            """Create mosaic distribution of base on a gausssian distribution
              :param dev: std deviation of.
              :type dev: :class:`float`
              :returns: :1d array --  mosaic distribution"""
            nums = []
            for i in range(self.orn):
                temp = random.gauss(self.mu, dev)
                nums.append(temp)
            return  np.array(nums)
        def weibull(self,dev):
            """Create mosaic distribution of base on a weibull distribution
              :param dev: std deviation of.
              :type dev: :class:`float`
              :returns: :1d array --  mosaic distribution"""
            nums = []
            for i in range(self.orn):
                temp = random.weibullvariate(self.mu, dev)
                nums.append(temp)
            return  np.array(nums)
        def lognormal(self,dev):
            """Create mosaic distribution of base on a lognormal distribution
              :param dev: std deviation of.
              :type dev: :class:`float`
              :returns: :1d array --  mosaic distribution """
            dist = np.random.lognormal(self.mu, dev, self.orn)
            count, bins, ignored = plt.hist(dist,density=True)
            bpts = np.linspace(min(bins),max(bins), self.orn)
            plt.show(block=False)
            return (np.exp(-(np.log(bpts) - self.mu)**2 / (2 * dev**2))/ (bpts * dev * np.sqrt(2 * np.pi)))
        