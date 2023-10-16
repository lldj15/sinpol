"""Module to calculate different stress strain models."""
import math as ma
import numpy as np

#pylint: disable=R0913
#pylint: disable=R0902
#pylint: disable=C0103
#pylint: disable=C0200
#pylint: disable=R0914
# pylint: disable=line-too-long
def volume(a, b, c, alp, bet, gam):
    """Calculates Volume."""
    thetsqrtv = ma.pow(ma.cos(ma.radians(alp)),2) + ma.pow(ma.cos(ma.radians(bet)),2)+ma.pow(ma.cos(ma.radians(gam)),2)-2*ma.cos(ma.radians(alp)) * ma.cos(ma.radians(bet))*ma.cos(ma.radians(gam))
    vol = a * b * c * ma.sqrt(1 - thetsqrtv)
    return vol
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
def gammacal(hk):
    """Calculates gamma coefficient."""
    num=np.square(hk[:,0])*np.square(hk[:,1])+np.square(hk[:,1])*np.square(hk[:,2])+np.square(hk[:,0])*np.square(hk[:,2])
    den=np.square(np.square(hk[:,0])+np.square(hk[:,1])+np.square(hk[:,2]))
    return num/den
class StressStrainModels:
    """Class different stress strain models"""
    def __init__(self,bunge,hkl,lattice, stress,c11,c12,c44):
        """constructor
        Parameters
        ----------
        bunge : orientation matrices
        hkl :  miller index
        lattice : lattice parameters 
        stress : stress matrix
        c11,c12,c44: stiffness coefficient """
        self.bunge=bunge
        self.stress=stress
        self.hkl=hkl
        self.lattice=lattice
        self.samplec=np.dot(self.hkl,self.bunge)
        self.volume=volume(self.lattice[0,0],self.lattice[0,1],self.lattice[0,2],self.lattice[0,3],self.lattice[0,4],self.lattice[0,5])
        self.lathkl=dhkl(self.lattice[0,0],self.lattice[0,1],self.lattice[0,2],self.lattice[0,3],self.lattice[0,4],self.lattice[0,5],self.hkl[:,0],self.hkl[:,1],self.hkl[:,2],self.volume)
        self.hklmag=float(self.lattice[0,2])/self.lathkl
        self.hklmag=np.reshape(self.hklmag,(len(self.hkl),1))
        self.c11=c11
        self.c12=c12
        self.c44=c44
        self.c11p=(3*self.c11/5)+(2*c12/5)+(4*c44/5)
        self.c12p=(self.c11-2*self.c44)/5 +4*c12/5
        self.c44p=(self.c11p-self.c12p)/2
    def voigt(self):
        """Calculates Voigt strain from a stress matrix using the model by Popa ,
         see reference https://doi.org/10.1107/97809553602060000967
         Return: 2d array of number of grain orientation and number of lattice planes
        """
        s11v=(self.c11p+self.c12p)/((self.c11p-self.c12p)*(self.c11p+2*self.c12p))
        s12v=-self.c12p/((self.c11p-self.c12p)*(self.c11p+2*self.c12p))
        s44v=1/self.c44p/4
        smatrixv=1e-3*np.array([[s11v,s12v,s12v,0,0,0],[s12v,s11v,s12v,0,0,0],
                           [s12v,s12v,s11v,0,0,0],[0,0,0,s44v,0,0],[0,0,0,0,s44v,0],[0,0,0,0,0,s44v]])

        voightstrain=np.dot(smatrixv,self.stress)
        B1=self.samplec[:,:,0]/self.hklmag
        B2=self.samplec[:,:,1]/self.hklmag
        B3=self.samplec[:,:,2]/self.hklmag
        vstrain=np.power(B1,2)*voightstrain[0,0]+np.power(B2,2)*voightstrain[1,0]+np.power(B3,2)*voightstrain[2,0]+2*B2*B3*voightstrain[3,0]+2*B1*B3*voightstrain[4,0]+2*B1*B2*voightstrain[5,0]# Voight conribution of the strain
        return vstrain.T
    def reuss(self):
        """Calculates Reuss strain from a stress matrix using the model by Popa , 
        see reference https://doi.org/10.1107/97809553602060000967.
        ::Return: 2d array of number of grain orientation and number of lattice planes
        """
        s11=(self.c11+self.c12)/((self.c11-self.c12)*(self.c11+2*self.c12))#Reuss
        s12=-self.c12/((self.c11-self.c12)*(self.c11+2*self.c12))#Reuss
        s44=1/self.c44/4
        smatrix=1e-3*np.array([[s11,s12,s12,0,0,0],[s12,s11,s12,0,0,0],
                           [s12,s12,s11,0,0,0],[0,0,0,s44,0,0],[0,0,0,0,s44,0],[0,0,0,0,0,s44]])##  Reuss
        a11=self.bunge[:,0,0]
        a12=self.bunge[:,0,1]
        a13=self.bunge[:,0,2]
        a21=self.bunge[:,1,0]
        a22=self.bunge[:,1,1]
        a23=self.bunge[:,1,2]
        a31=self.bunge[:,2,0]
        a32=self.bunge[:,2,1]
        a33=self.bunge[:,2,2]
        Qmatrix=np.array([[a11*a11,a12*a12,a13*a13,2*a12*a13,2*a11*a13,2*a11*a12],[a21*a21,a22*a22,a23*a23,2*a22*a23,2*a21*a23,2*a21*a22],[a31*a31,a32*a32,a33*a33,2*a32*a33,2*a31*a33,2*a31*a32],
                           [a21*a31,a22*a32,a33*a23,a22*a33+a23*a32,a33*a21+a31*a23,a22*a31+a21*a32],[a11*a31,a12*a32,a13*a33,a33*a12+a32*a13,a11*a33+a13*a31,a11*a32+a12*a31],
                           [a11*a21,a12*a22,a13*a23,a22*a13+a23*a12,a11*a23+a13*a21,a11*a22+a12*a21]])
        rstress=np.array([np.dot(Qmatrix[:,:,i],self.stress)for  i in range (len(self.bunge))])
        rhomatrix=np.array([[1],[1],[1],[2],[2],[2]])
        rrstress=rhomatrix*rstress
        rstrain=np.dot(smatrix,rrstress)
        A1=self.hkl[:,0]/self.hklmag[:,0]
        A2=self.hkl[:,1]/self.hklmag[:,0]
        A3=self.hkl[:,2]/self.hklmag [:,0]
        Amatrix=np.array([[np.power(A1,2),np.power(A2,2),np.power(A3,2),2*A2*A3,2*A1*A3,2*A1*A2]])
        rstrainv=np.array([np.dot(Amatrix[0,:,h],rstrain[:,i,0]) for h in range(len(self.hkl)) for  i in range (len(self.bunge))])
        rstrainv=rstrainv.reshape(len(self.hkl),len(self.bunge))
        return rstrainv.T
    def hill (self):
        """Calculates Hill strain from a stress matrix using the model by Popa , 
        see reference https://doi.org/10.1107/97809553602060000967.
        Return: 2d array of number of grain orientation and number of lattice planes
        """
        return (self.reuss()+self.voigt())/2.0
    def kronerrandom(self):
        """Calculates Eshelby-Kroner strain from a stress matrix using the model by Popa , 
        see reference https://doi.org/10.1107/97809553602060000967.
         Return: 2d array of number of grain orientation and number of lattice planes
        """
        s11=(self.c11+self.c12)/((self.c11-self.c12)*(self.c11+2*self.c12))#Reuss
        s12=-self.c12/((self.c11-self.c12)*(self.c11+2*self.c12))#Reuss
        s44=1/self.c44
        kk=1/(3*(s11+2*s12))
        muk=1/s44
        nuk=1/(2*(s11-s12))
        mukr=muk/nuk
#         gamzero=1/(3+2/mukr)
        alpv=3/8*(3*kk+4*muk)-muk/5 *(3+2/mukr)
        aptg=-9/2 * muk*(1-1/mukr)
        betv=(3/4)*kk*muk*(1-.1*((6/mukr)+9+(20/mukr)*muk/kk))
        betgv=-9/4 *kk* muk*(1-1/mukr)
        gamv=(-3/4)*kk*nuk*muk
        gamval=gammacal(self.hkl)
        gvalu=gamval
        G=np.zeros([len(gvalu)])
        for j in range(len(gvalu)):
            gcoeff=np.array([1,alpv+aptg*gvalu[j],betv+betgv*gvalu[j],gamv])
            gsolve=np.roots(gcoeff)
            G[j]=gsolve.max()
        ES1=(1/(9*kk))-1/(6*G)
        ES2=1/(G)
        s12k=ES1
        s11k=ES1+ES2/2
        s44k=0
        B1=self.samplec[:,:,0]/self.hklmag
        B2=self.samplec[:,:,1]/self.hklmag
        B3=self.samplec[:,:,2]/self.hklmag
        kstrain=np.zeros([len(self.hkl),len(self.bunge)])
        for i in range(len(self.hkl)):
            smatrix=1e-3*np.array([[s11k[i],s12k[i],s12k[i],0,0,0],[s12k[i],s11k[i],s12k[i],0,0,0],[s12k[i],s12k[i],s11k[i],0,0,0],[0,0,0,s44k,0,0],[0,0,0,0,s44k,0],[0,0,0,0,0,s44k]])
            kroenerstrain=np.dot(smatrix,self.stress)
            kstrain[i,:]=np.power(B1[i,:],2)*kroenerstrain[0,0]+np.power(B2[i,:],2)*kroenerstrain[1,0]+np.power(B3[i,:],2)*kroenerstrain[2,0]+2*B2[i,:]*B3[i,:]*kroenerstrain[3,0]+2*B1[i,:]*B3[i,:]*kroenerstrain[4,0]+2*B1[i,:]*B2[i,:]*kroenerstrain[5,0]# kroener conribution of the strain
        return kstrain.T
