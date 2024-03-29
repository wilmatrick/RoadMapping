from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt

class SF_Cylinder(SelectionFunction):
    """Class that implements a cylindrical selection function around the sun"""
    def __init__(self,rsun,dsun,zmin,zmax,df=None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a cylindrical selection function
        INPUT:
            rsun: radius of cylinder around sun [_REFR0]
            dsun: distance of sun to galactic center [_REFR0]
            zmin, zmax: minimum and maximum height above the plane in units of [_REFR0]
        OUTPUT:
            cylindrical selection function object
        HISTORY:
        """
        SelectionFunction.__init__(self,df=df)

        if zmin > zmax:
            sys.exit("Error in SF_Cylinder.__init__(rsun,dsun,zmin,zmax):"+\
                     " zmin has to be smaller than zmax.")

        #Parameters of the cylindrical selection function:
        self._rsun = rsun
        self._dsun = dsun
        self._zmin = zmin
        self._zmax = zmax

        #Borders:
        self._Rmin = dsun - rsun
        self._Rmax = dsun + rsun
        self._pmax = math.degrees(math.asin(rsun / dsun))
        self._pmin = - self._pmax

        return None

    #-----------------------------------------------------------------------

    def _contains(self,R,z,phi=None):

        sys.exit("Error in SF_Cylinder._contains(): This method is not implemented yet. TO DO!!!!!!!!!!!!!!!!!!!!!")

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._contains(): "+
                     "Function not yet implemented to take care of imcompleteness.")

    #-----------------------------------------------------------------------

    def _phimax_rad(self,R):
        """largest possible phi at given radius"""

        if isinstance(R,numpy.ndarray):
            return numpy.array([self._phimax_rad(rr) for rr in R])

        if R < self._Rmin or R > self._Rmax:
            return 0.    

        cosphi = (R**2 - self._rsun**2 + self._dsun**2) / (2. * self._dsun * R)
        phimax = numpy.fabs(numpy.arccos(cosphi))  #rad
        return phimax

    #-----------------------------------------------------------------------

    def _densfunc(self,R,z,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False):

        if self._densInterp is None:
            sys.exit("Error in SF_Cylinder._densfunc(): "+\
                     "self._densInterp is None. Initialize density grid"+\
                     " first before calling this function.")

        #take care of array input:
        if isinstance(z,numpy.ndarray) or isinstance(R,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)

        #outside of obseved volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_Cylinder._densfunc(): "+\
               "Specify phi in case of the cylindrical selection function, "+\
               "when using throw_error_outside=True or set_outside_zero=True.")

            x = self._dsun - R * numpy.cos(numpy.radians(phi))
            y = R * numpy.sin(numpy.radians(phi))

            outside = ((x**2 + y**2) > self._rsun**2) + \
                      (z < self._zmin) + \
                      (z > self._zmax)

            if numpy.sum(outside) > 0:
                if throw_error_outside:
                    print "x^2+y^2 = ",x[outside]**2 + y[outside]**2
                    print "R_sun^2     = ",self._rsun**2
                    print "R: ",self._Rmin," <= ",R[outside]," <= ",self._Rmax,"?"
                    print "z: ",self._zmin," <= ",z[outside]," <= ",self._zmax,"?"
                    dphi = 0.5 * self._phimax_rad(R[outside])
                    print "phi: ",-dphi," <= ",phi[outside]," <= ",+dphi,"?"
                    sys.exit("if yes: something is wrong!")
                if set_outside_zero:
                    return 0.
            
        if not isinstance(R,numpy.ndarray) or len(R) == 1:
            temp = self._densInterp.ev(R,numpy.fabs(z))     #in different versions of scipy ev() returns a result of shape () or (1,)
            temp = numpy.reshape(numpy.array([temp]),(1,)) #this line tries to circumvent the problem
            d = numpy.exp(temp[0])    #the exp of the grid, as the grid is in log
        else:
            temp = self._densInterp.ev(R,numpy.fabs(z))
            d = numpy.exp(temp)    #the exp of the grid, as the grid is in log
            if not len(R) == len(d):
                sys.exit("Error in SF_Wedge._densfunc(): somethin with the array output is wrong.")
        return d

    #-------------------------------------------------------------------

    def _fastGLint_cylinder(self,func,xgl,wgl):

        """integrate the given function func(R,z,phi) over the cylindrical effective volume 
           by hand, analogous to Bovy, using Gauss Legendre quadrature."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._fastGLint_cylinder(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #R coordinates: R_i = 0.5 * (Rmax - Rmin) * (xgl_i + 1) + Rmin:
        Rgl_i = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin

        #z coordinates: z_j = 0.5 * (zmax - zmin) * (xgl_j + 1) + zmin:
        zgl_j = 0.5 * (self._zmax - self._zmin) * (xgl + 1.) + self._zmin

        #account for integration limits and Jacobian in weights w_i and w_j:
        w_i = (self._Rmax - self._Rmin) * wgl    #account for integration limits in R and phi
        w_i *= Rgl_i #account for cylindrical coordinates Jacobian
        w_j = 0.5 * (self._zmax - self._zmin) * wgl     #account for integration limits in z

        #mesh everything:
        ngl = len(xgl)
        Rglm_i  = numpy.zeros((ngl,ngl))
        zglm_j  = numpy.zeros((ngl,ngl))
        wm_i    = numpy.zeros((ngl,ngl))
        wm_j    = numpy.zeros((ngl,ngl))
        for ii in range(ngl):
            for jj in range(ngl):
                Rglm_i[ii,jj] = Rgl_i[ii]
                zglm_j[ii,jj] = zgl_j[jj]
                wm_i[ii,jj] = w_i[ii]
                wm_j[ii,jj] = w_j[jj]

        #flatten everything:
        wm_i = wm_i.flatten()
        wm_j = wm_j.flatten()
        Rglm_i = Rglm_i.flatten()
        zglm_j = zglm_j.flatten()

        #phi coordinates (dummy):
        phi_ij = numpy.zeros_like(Rglm_i)

        #evaluate function at each grid point:
        func_ij = func(Rglm_i,zglm_j,phi_ij)

        #angular extend at each grid point:
        phimaxm_i = self._phimax_rad(Rglm_i)

        #total:
        tot = numpy.sum(wm_i * wm_j * func_ij * phimaxm_i)
        return tot

    #-----------------------------------------------------------------

    def _Mtot_fastGL(self,xgl,wgl):

        """integrate total mass inside effective volume by hand, analogous to Bovy, using Gauss Legendre quadrature.
           The integration accounts for integration limits - we therefore do not have to set the density outside the cylinder to zero."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._Mtot_fastGL(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #define function func(R,z,phi) to integrate:
        func = lambda rr,zz,pp: self._densfunc(rr,zz,phi=pp,set_outside_zero=False,throw_error_outside=True,consider_incompleteness=False)
        
        #total mass in selection function:
        Mtot = self._fastGLint_cylinder(func,xgl,wgl)
        return Mtot

    #-----------------------------------------------------------------

    def _spatialSampleDF_complete(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,recalc_densgrid=True):

        #initialize interpolated density grid:
        if not quiet: print "Initialize interpolated density grid"
        if recalc_densgrid:
            self.densityGrid(
                nrs_nonfid=nrs,
                nzs_nonfid=nzs,
                ngl_vel_nonfid=ngl_vel, 
                n_sigma=n_sigma,
                vT_galpy_max=vT_galpy_max,
                test_sf=test_sf,
                _multi=_multi,
                recalculate=recalc_densgrid)
        # *Note:* The use of the flag "recalc_densgrid" is actually not 
        # needed here, because setting the analogous flag "recalculate" 
        # in densityGrid should have the same effect.

        #maximum of density:
        zprime = min(math.fabs(self._zmin),math.fabs(self._zmax))
        if self._zmin < 0. and self._zmax > 0.:
            zprime = 0.
        densmax = self._df.density(self._Rmin,zprime,ngl=ngl_vel,nsigma=n_sigma,vT_galpy_max=vT_galpy_max)
        #print densmax
        #print self._densfunc(self._Rmin,zprime,phi=0.)
        #sys.exit("test")
        
        #number of found mockdata:
        nfound = 0
        nreject = 0
        Rarr = []
        zarr = []
        phiarr = []

        if not quiet: print "Start sampling"

        while nfound < nmock:
            
            eta = numpy.random.random(4)

            #cylindrical selection function:
            z = (self._zmax - self._zmin) * eta[0] + self._zmin   #uniform distributed
            rc = math.sqrt(eta[1]) * self._rsun   #radius within cylinder, distributed according to P(<rc) ~ rc^2 --> rc ~ sqrt(y) (according to inverse function method)
            psi = 2. * math.pi * eta[2] #angle within cylinder, uniformly distributed

            #transformation to (R,phi):
            x = self._dsun - rc * math.cos(psi)
            y = rc * math.sin(psi)
            R = math.sqrt(x**2 + y**2)
            phi = math.atan2(y,x)   #rad
            phi = math.degrees(phi) #deg

            #density at this point:
            dens = self._densfunc(R,z,phi=phi,set_outside_zero=False,throw_error_outside=True,consider_incompleteness=False)

            #Rejection method:
            dtest = densmax * eta[3]
            if dtest < dens:
                Rarr.extend([R])
                zarr.extend([z])
                phiarr.extend([phi])
                nfound += 1
                if not quiet: print nfound," found"
            else:
                nreject += 1
                if not quiet: print nreject," rejected"

        return numpy.array(Rarr),numpy.array(zarr),numpy.array(phiarr)

    #-----------------------------------------------------------------------------

    def _dMdR(self,R,ngl=20):

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._dMdR(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        
        if isinstance(R,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdR(rr) for rr in R])

        if R > self._Rmax or R < self._Rmin:
            return 0.

        if self._zmin == -self._zmax:
            zlow = 0.
            f = 2.
        else:
            zlow = self._zmin
            f = 1.

        phimax = self._phimax_rad(R)    #radians

        dM = 2. * phimax * R * f * (scipy.integrate.fixed_quad(lambda zz: self._densfunc(R,zz,consider_incompleteness=False),
                                       zlow,self._zmax,n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------

    def _dMdRdz(self,R,*args):

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._dMdRdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        z = args[0]
        phimax = self._phimax_rad(R)    #radians
        return 2. * phimax * R * self._densfunc(R,z,consider_incompleteness=False)      


    def _dMdz(self,z,ngl=None):

        if ngl is None: ngl = 20
        
        if isinstance(z,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdz(zz) for zz in z])


        dM = (scipy.integrate.fixed_quad(self._dMdRdz,
                                         self._Rmin,self._Rmax,args=[z],n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------

    def _surfDens(self,R,phi=None,ngl=20):

        if self._with_incompleteness:
            sys.exit("Error in SF_Cylinder._surfDens(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        
        if phi is None: sys.exit("Error in SF_Cylinder._surfDens(): Specify phi.")
        
        if isinstance(R,numpy.ndarray):
            #Recursion if input is array:
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(R)
            return numpy.array([self._surfDens(rr,phi=pp,ngl=ngl) for rr,pp in zip(R,phi)])
        elif isinstance(phi,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(phi)
            return numpy.array([self._surfDens(rr,phi=pp,ngl=ngl) for rr,pp in zip(R,phi)])

        if self._zmin == -self._zmax:
            zlow = 0.
            f = 2.
        else:
            zlow = self._zmin
            f = 1.

        dM = f * (scipy.integrate.fixed_quad(lambda zz: self._densfunc(R,zz,phi=phi,set_outside_zero=True,throw_error_outside=False,consider_incompleteness=False),
                                       zlow,self._zmax,n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------  


    def _dMdphi(self,phi,ngl=None):

        if ngl is None: ngl = 20
        
        if isinstance(phi,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdphi(pp) for pp in phi])

        if numpy.fabs(phi) > self._pmax:
            return 0. 

        #the small and large radius at which a secant under the angle phi 
        #intersects the circle around the sun:
        d = self._dsun
        cp = math.cos(math.radians(phi))
        sp = math.sin(math.radians(phi))
        r = self._rsun
        Rlow = d * cp - math.sqrt(r**2 - d**2 * sp**2)
        Rup  = d * cp + math.sqrt(r**2 - d**2 * sp**2)

        dM = (scipy.integrate.fixed_quad(lambda rr: rr * self._surfDens(rr,phi=phi,ngl=ngl),
                                         Rlow,Rup,n=ngl))[0]
        return dM

    #-------------------------------------------------------------------------------------

    def _dMdv(self,v,return_vz=False,return_vR=False,return_vT=False,ngl=20,_multi=None,xgl=None,wgl=None):

        # sample points (x) and weights (w) for Gauss-Legendre (gl) quadrature
        if (xgl is None) or (wgl is None): 
            xgl,wgl = numpy.polynomial.legendre.leggauss(ngl) 

        if isinstance(v,numpy.ndarray) and len(v) > 1:
            if _multi is None or _multi == 1:
                            return numpy.array([self._dMdv(
                                    vv,
                                    return_vz=return_vz,
                                    return_vR=return_vR,
                                    return_vT=return_vT,
                                    xgl=xgl,
                                    wgl=wgl,
                                    _multi=None
                                    ) for vv in v])
            else:
                multOut = multi.parallel_map(
                        (lambda x: self._dMdv(
                                    v[x],
                                    return_vz=return_vz, 
                                    return_vR=return_vR, 
                                    return_vT=return_vT,
                                    xgl=xgl,
                                    wgl=wgl,
                                    _multi=None
                                    )),
                        range(len(v)),
                        numcores=numpy.amin([
                                len(v),
                                multiprocessing.cpu_count(),
                                _multi])
                        )           
                return numpy.array(multOut)

        #define the function which to marginalize over (R,z,phi):
        pvfunc = lambda rr,zz,pp: (self._dMdvdRdz(
                                    v,
                                    rr,
                                    zz,
                                    return_vz=return_vz,
                                    return_vR=return_vR,
                                    return_vT=return_vT,
                                    xgl=xgl,
                                    wgl=wgl
                                    ))
        #pp is here only a dummy, but is required as input format of func

        #integrate:
        dM = self._fastGLint_cylinder(pvfunc,xgl,wgl)
        return dM
