"""Module to calculate Neutran transmission intensity."""
import math as ma
import threading
from numba import numba as nb, jit,njit, float64,int64
from numba import cfunc
from numba.types import intc, CPointer
from NumbaQuadpack import quadpack_sig, dqags
from scipy import LowLevelCallable
import numpy as np
from sinpol.diffmodels import DiffModels as dm
# pylint: disable=line-too-long
#pylint: disable=R0913
#pylint: disable=R0902
#pylint: disable=R0914
#pylint: disable=C0103
#pylint: disable=C0200
#pylint: disable=W0612
#pylint: disable=W0613
#pylint: disable=E1133
@jit(nopython=True)
def volume(a, b, c, alp, bet, gam):
    """Calculates Volume."""
    thetsqrtv = ma.pow(ma.cos(ma.radians(alp)), 2) + ma.pow(ma.cos(ma.radians(bet)), 2) + ma.pow(
        ma.cos(ma.radians(gam)), 2) - 2 * ma.cos(ma.radians(alp)) * ma.cos(ma.radians(bet)) * ma.cos(ma.radians(gam))
    vol = a * b * c * ma.sqrt(1 - thetsqrtv)
    return vol
def jit_integrand_function(integrand_function):
    """ Calculate integrand"""
    jitted_function = nb.jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n,x):
        """ Calculate integrand"""
        return jitted_function(x[0],x[1])
    return LowLevelCallable(wrapped.ctypes)
@nb.cfunc(quadpack_sig)
def integrand(x,*arg):
    """ Calculate integrand"""
    return x / (np.exp(-x) - 1)
xc1= integrand.address
@njit
def interlog(xtemp):
    """ Calculate integrand"""
    xcal2=dqags(xc1,xtemp,0)
    return xcal2[0] / xtemp
@jit(nopython=True)
def dhkl(a, b, c, alp, bet, gam, h, k, l, vol):
    """Calculates lattice spacing."""
    hbc = np.power(h, 2) * np.power(b, 2) * np.power(c, 2) * np.power(np.sin(np.radians(alp)), 2)
    kac = np.power(k, 2) * np.power(a, 2) * np.power(c, 2) * np.power(np.sin(np.radians(bet)), 2)
    lab = np.power(l, 2) * np.power(a, 2) * np.power(b, 2) * np.power(np.sin(np.radians(gam)), 2)
    abg = 2.0 * h * k * a * b * np.power(c, 2) * (
    np.cos(np.radians(alp)) * np.cos(np.radians(bet)) - np.cos(np.radians(gam)))
    bga = 2.0 * k * l * np.power(a, 2) * b * c * (
    np.cos(np.radians(bet)) * np.cos(np.radians(gam)) - np.cos(np.radians(alp)))
    agb = 2.0 * h * l * a * np.power(b, 2) * c * (
    np.cos(np.radians(alp)) * np.cos(np.radians(gam)) - np.cos(np.radians(bet)))
    den = hbc + kac + lab + abg + bga + agb
    d =vol/np.power(den,.5)
    return d
def nb_multithread(func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_nb(*args):
        """
        Run the given function inside *numthreads* threads, splitting its
        arguments into equal-sized chunks.
        """
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_nb

@jit(nopython=True)
def gaussian(x, mu, s):
    """  Calculate gaussian distribution"""
    gs = 1 / np.power((2 * np.pi * np.power(s, 2)),.5) * np.exp(-np.power(x - mu, 2) / (2 * np.power(s, 2)))
#     print x
    return gs
@jit(nopython=True)
def Q_cal(lamb, fs, nc, gangle):
    """ Calculate Q value """
    den = np.sin(2 * gangle)
    aval=np.power(lamb, 3) * fs * np.power(nc, 2)
    val=aval/den
    return val
@jit(nopython=True)
def fsquare(h, k, l, atp, atpossize, w, b_coh):
    """ Calculate square of strucutre factor"""
    debw = np.exp(-2 * w)
    fsq=np.empty((len(l)))
    for i in nb.prange(len(l)):
        cos_exp = 0
        sin_exp = 0
        for atm in range(atpossize):
            exp = 2.0 * np.pi * (h[i]  * atp[atm][0] + k[i]  * atp[atm][1] + l[i] * atp[atm][2])
            cos_exp += np.cos(exp)
            sin_exp += np.sin(exp)
            fsq[i] = (np.power(cos_exp, 2) + np.power(sin_exp, 2)) * np.power(b_coh, 2)*debw[i]
    return fsq
@njit('(float64[:,:,:], float64[:,:])',parallel=True,fastmath=True)
def tintense(I,b):
    """ Calculate transmission intensity in area of low to no absorption"""
    r,c,t=I.shape
    l,n=b.shape
    rintense=np.empty((r,c))
    for i in nb.prange(l):
        for j in range(n):
            iA=I[i,j,:]/b[i,j]
            ib=1.0-iA
            isum=np.sum(ib)
            ic=isum+1.0
            ifinal=1.0/ic
            rintense[i,j]=ifinal
    return rintense
@njit('(float64[:], float64[:,:])', parallel=True,fastmath=True)
def calsin(w,l):
    """ Calculate theta Bragg"""
    o,hh=l.shape
    e=len(w)
    # print(w.shape,l.shape)
    data = np.empty((e, o, hh))
    for i in nb.prange(e):
        for j in range(o):
            for k  in range(hh):
                data[i,j,k]=ma.asin(w[i]/(2.0*l[j,k]))
                # print( data[i,j,k])
    data=np.swapaxes(data,0,1)
    return data
@njit('(float64[:,:], float64[:,:])',parallel=True,fastmath=True)
def sampcal(s,h):
    """ Calculate diffraction cosine angle"""
    su,hu=s.shape
    dat=np.empty((su,hu))
    for i in nb.prange(su):
        for j in range (hu):
            dat[i,j]=s[i,j]/h[i,j]
    return dat
@njit('(float64[:,:],int64[:,:])',fastmath=True,parallel=True)
def backgrd(bm,hk):
    print(bm.shape,hk.shape)
    """Distribute background over all grains   """
    bm=np.swapaxes(bm,0,1)
    print(bm.shape,hk.shape)
    t,l=bm.shape
    hm,klm=hk.shape
    br=np.empty((t,l,hm))
    for i in nb.prange(t):
        for j in range (l):
            for q in range(hm):
                br[i,j,q]=bm[i,j]
    br=np.swapaxes(br,0,1)
    return  br
@njit(fastmath=True,parallel=True)
def gamasin(gam):
    """ Calculate theta Bragg"""
    t,l=gam.shape
    br=np.empty((t,l))
    for j in nb.prange(t):
        for k in range(l):
            br[j,k]=np.arcsin(gam[j,k])
    return  br
@nb.njit('(float64[:,:,:],float64[:,:,:],float64[:,:,:])',fastmath=True,parallel=True)
def findidx(tmin,tcal,tmax):
    """ Find the indices necessary needed to carry  diffraction and transmission calculation"""
    qa=np.where(np.logical_and(tmin<=tcal,tcal<tmax))
    qb=np.empty((len(qa[0]),3),dtype=int64)
    for i in nb.prange (len(qa[0])):
        qb[i,0]=qa[0][i]
        qb[i,1]=qa[1][i]
        qb[i,2]=qa[2][i]
    return qb
@njit('(int64[:,:],float64[:,:,:])',fastmath=True)
def numba_dot(A, B):
    """ dot product  in numba """
    m, n = A.shape
    u,t,p = B.shape
    C = np.zeros((m,u,p))
    for i in range(m):
        for g in range(u):
            for j in range(n):
                for k in range(p):
                    C[i, k] += A[i, j] * B[j,g, k]
    return C
@njit(fastmath=True,parallel=True)
def numba_dot2(A, B):
    """ dot product  in numba"""
    m, n = A.shape
    u,t,p = B.shape
    C = np.zeros((u,t,p))
    for i in range(t):
        C[:,i,:]=np.dot(A,B[:,i,:])
    return C
@njit('(float64[:],int64)',fastmath=True,parallel=True)
def mrepeat(mos,hk):
    """Repeat mosaic distribitution across hkl matrix, for calculation"""
    br=np.empty((len(mos),hk))
    for i in nb.prange(len(mos)):
        for j in nb.prange (hk):
            br[i,j]=mos[i]
    return  br
@njit('(int64,float64[:])',fastmath=True,parallel=True)
def lrepeat(b,lat):
    """   Repeat lattice spacing across grain  distribution calculation"""
    br=np.empty((b,len(lat)))
    for i in nb.prange(b):
        for j in range (len(lat)):
            br[i,j]=lat[j]
    return  br
@nb.njit('(float64[:,:],int64)',fastmath=True,parallel=True)
def trepeat(theta,l):
    """ Rewrite as  3d array dependent on number of grains, length of wavelength bin,  and number of lattice planes  """
    a,b=theta.shape
    br=np.empty((a,l,b))
    for i in nb.prange(a):
        for j in nb.prange (l):
            for k in nb.prange (b):
                br[i,j,k]=theta[i,k]
    return  br
@nb.njit(fastmath=True,parallel=True)
def newlamb(lam,lmin):
    """  Set the minimum value for lambda to be used in transmission calculation"""
    a,b=lam.shape
    br=np.empty((a,b))
    for i in nb.prange(a):
        for j in nb.prange (b):
            if lam[i,j]>lmin:
                br[i,j]=lam[i,j]
            else:
                br[i,j]=lmin
    return  br
# @njit(('float64[:,:],float64,float64[:],float64[:],int64[:,:],float64[:],float64[:,:],float64[:],float64[:],float64[:]'))
def scatvar(bunge,omg,gs,mosaic,hkl,lattice,atp,xtal,strain, wvltg):
    """ Calculate all parameters needed to calculate transmission and diffraction intensities"""
    samplec=np.empty((len(hkl),bunge.shape[0],bunge.shape[1]))
    stbm=np.zeros((len(hkl),bunge.shape[0],bunge.shape[1]))
    mosaica=mrepeat(mosaic, len(hkl))
    beammatrix=np.empty((3,3))
    samplec=np.dot(hkl,bunge)
    lathkl=dhkl(lattice[0,0],lattice[0,1],lattice[0,2],lattice[0,3],lattice[0,4],lattice[0,5],hkl[:,0],hkl[:,1],hkl[:,2],volume(lattice[0,0],lattice[0,1],lattice[0,2],lattice[0,3],lattice[0,4],lattice[0,5]))
    lathkl=lrepeat(len(bunge),lathkl)
    hklmag=float(lattice[0,2])/lathkl
    hklmag=np.reshape(hklmag,(len(hkl),len(bunge)))
    beammatrix[0,0]=np.cos(omg)
    beammatrix[0,1]=0
    beammatrix[0,2]=-np.sin(omg)
    beammatrix[1,0]=0
    beammatrix[1,1]=1
    beammatrix[1,2]=0
    beammatrix[2,0]=np.sin(omg)
    beammatrix[2,1]=0
    beammatrix[2,2]=np.cos(omg)
    gammazero=np.cos(omg)
    stbm=np.dot(samplec,beammatrix)
    gamma=stbm[:,:,2].T*lathkl/lattice[0,2]
    lathkln=lathkl*strain+lathkl
    lmbda= 2.0*lathkln*gamma
    thetahkl=gamasin(gamma)
    thetamax=thetahkl+5*mosaica
    thetamin=thetahkl-5*mosaica
    lmbda=newlamb(lmbda,wvltg.min())
    thetai=calsin(wvltg,lathkln)
    sampz=sampcal(samplec[:,:,2],hklmag)
    sampz=np.swapaxes(sampz,0,1)
    qa=findidx(trepeat(thetamin,len(wvltg)), thetai, trepeat(thetamax,len(wvltg)))
    bankdec=np.sin (2.0*omg)*(np.power(samplec[:,:,0]/hklmag ,2)-np.power(samplec[:,:,2]/hklmag ,2))-2.0*np.cos(2.0*omg)*samplec[:,:,0]/hklmag *samplec[:,:,2]/hklmag
    detecV=np.abs((samplec[:,:,1].T/hklmag[:,0])*np.sin(thetahkl))  # Vertical length of detector
    detecH=np.cos(2.0*thetahkl)  # horizontal lenght of detector
    return mosaica,gammazero,lmbda,thetahkl,thetai,qa,sampz
# @njit(('float64[:,:,:],float64[:,:],float64[:,:],float64,int64[:,:],float64[:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:,:],float64[:]'),fastmath=True)
def _transmission(b,gs,mosaic,omg,hkl,lattice,atp,xtal,nxsa,nxsts,nxstm,strain,wvltg ):
    """ Calculate  neutron transmission intensity over a plate"""
    trans=np.ones((len(b),len(wvltg)))
    hplanck = 4.135E-15 # evslsvi
    kb = 8.617E-5
    for j in range(len(b)):
        mosaica,gammazero,lmbda,thetahkl,thetai,qa,sampz=scatvar(b[j],omg,gs[j],mosaic[j],hkl,lattice,atp,xtal,strain[j], wvltg)
        mu=xtal[5]*(nxsa+nxsts+nxstm)*float(len(b[j]))
        transtemp=np.ones((len(b[j]),len(wvltg),len(hkl)))
        backmu=np.exp(-mu*gs[j][:,np.newaxis])
        transtemp=backgrd(backmu,hkl)
        wval=9.0e36 *(6.0 * np.power(float(hplanck), 2) / ( float(xtal[7]) * float(kb) * float(xtal[6])* 939.565 * 1E6 )) * np.power(np.sin(thetahkl[qa[:,0],qa[:,2]]), 2)/(np.power(lmbda[qa[:,0],qa[:,2]], 2)) * (float(interlog(xtal[8])) + 0.25)
        fsq=fsquare(hkl[qa[:,2],0],hkl[qa[:,2],1],hkl[qa[:,2],2],atp,len(atp),wval,xtal[0])
        Q=Q_cal(lmbda[qa[:,0],qa[:,2]], fsq,1.0/volume(lattice[0,0],lattice[0,1],lattice[0,2],lattice[0,3],lattice[0,4],lattice[0,5]),thetahkl[qa[:,0],qa[:,2]])
        mos=gaussian(thetai[qa[:,0],qa[:,1],qa[:,2]],thetahkl[qa[:,0],qa[:,2]],mosaica[qa[:,0],qa[:,2]])
        gamhkl=np.cos(omg)-2.0*np.sin(thetahkl[qa[:,0],qa[:,2]])*sampz[qa[:,0],qa[:,2]]
        gamhkl[np.abs(gamhkl)<.1]=.1
        sigma=Q*mos
        bn1=sigma*gs[j][qa[:,0]]
        bnn1=bn1/gammazero
        cn1=np.array(np.abs(gammazero/gamhkl))
        an1=mu[qa[:,1]]*gs[j][qa[:,0]]
        pn1=(an1+bn1)*(1+cn1)*.5
        qn1=(an1+bn1)*(1-cn1)*.5
        rn1=np.sqrt(np.power(pn1,2)-cn1*np.power(bn1,2))
        sn1=np.sqrt(np.power(qn1,2)+cn1*np.power(bn1,2))
        ta=np.where(gamhkl<=0)
        tb=np.where(gamhkl>0)
        transtemp[qa[ta[0],0],qa[ta[0],1],qa[ta[0],2]]=dm(rn1[ta[0]],qn1[ta[0]], pn1[ta[0]],sn1[ta[0]],bnn1[ta[0]]).sears_lr()
        transtemp[qa[tb[0],0],qa[tb[0],1],qa[tb[0],2]]=dm(rn1[tb[0]],qn1[tb[0]], pn1[tb[0]],sn1[tb[0]],bnn1[tb[0]]).sears_lt()
        transrt=tintense(transtemp,backmu)
        transrt[np.isnan(transrt)]=1
        transrt[np.isinf(transrt)]=1
        transrt[transrt==0]=1
        trans[j]=np.prod(transrt,axis=0)*np.mean(backmu,axis=0)
    return np.mean(trans,axis=0)
class ScatteringCalc:
    """ Calculate scattering and transmission intensities """ 
    def __init__(self,samp,hkl,lattice,atp,xtal,nxsa,nxsts,nxstm,strain, wvltg):
        self.samp=samp
        self.hkl=hkl
        self.atp=atp
        self.lattice=lattice
        self.xtal=xtal
        self.nxsa=nxsa
        self.nxsts=nxsts
        self.nxstm=nxstm
        self.strain=strain
        self.wvltg=wvltg
    def transmission(self):
        """ Calculate  Transmission intensity"""
        bunge_b=np.array_split(self.samp[0],self.samp[4])
        gs_b=np.split(self.samp[2],self.samp[4])
        mosaic_b=np.split(self.samp[3],self.samp[4])
        strain_b=np.array_split(self.strain,self.samp[4])
        return _transmission(bunge_b,gs_b,mosaic_b,self.samp[1],self.hkl,self.lattice,self.atp,self.xtal,self.nxsa,self.nxsts,self.nxstm,strain_b,self.wvltg)
    def diffraction(self,detect,dsp):
        """ to be done """
        