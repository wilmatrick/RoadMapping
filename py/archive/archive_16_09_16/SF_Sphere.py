from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt
from coord_trafo import galcencyl_to_radecDM, radecDM_to_galcencyl

class SF_Sphere(SelectionFunction):
    """Class that implements a spherical selection function around the sun"""
    def __init__(self,dmax,Rcen,zcen=0.,phicen_deg=0.,df=None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a spherical selection function
        INPUT:
            dmax: radius of sphere around center (@ sun) [_REFR0*ro]
            Rcen: distance of center (@ sun) to Galactic center [_REFR0*ro]
            zcen: height of center over Galactic plane [_REFR0*ro] --> Nonzero does not work yet. TO DO???????????????????????
            phicen_deg: azimuth of center [deg]
        OUTPUT:
            spherical selection function object
        HISTORY:
            2015-11-30 - Started SF_Sphere.py on the basis of BovyCode/py/SF_Sphere.py - Trick (MPIA)
                       - Renamed rsun (radius of sphere) --> dmax, dsun --> Rcen, _phimax_rad() --> _deltaphi_max_rad()
        """
        SelectionFunction.__init__(self,df=df)

        if zcen != 0.:
            sys.exit("Error in SF_Sphere: Nonzero zcen is not implemented yet.")

        #Parameters of the spherical selection function:
        self._dmax = dmax
        self._Rcen = Rcen
        self._zcen = zcen
        self._phicen_deg = phicen_deg

        #Borders:
        self._Rmin = Rcen - dmax
        self._Rmax = Rcen + dmax
        self._zmin = zcen - dmax
        self._zmax = zcen + dmax
        self._pmax = phicen_deg + math.degrees(math.asin(dmax / Rcen))
        self._pmin = phicen_deg - math.degrees(math.asin(dmax / Rcen))

        return None

    #-----------------------------------------------------------------------

    def _contains(self,R,z,phi=None):

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._contains(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        # Recursion if input is array:
        if phi is None: sys.exit("Error in SF_Sphere._contains(): Always specify phi in case of the spherical selection function.")
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(R)
            return numpy.array([self._contains(rr,zz,phi=pp) for rr,zz,pp in zip(R,z,phi)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(z)
            return numpy.array([self._contains(rr,zz,phi=pp) for rr,zz,pp in zip(R,z,phi)])
        elif isinstance(phi,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(phi)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(phi)
            return numpy.array([self._contains(rr,zz,phi=pp) for rr,zz,pp in zip(R,z,phi)])

        #rotate x axis to go through center of sphere:
        phip = phi - self._phicen_deg

        # transform from cylindrical galactocentric coordinates
        # to rectangular coordinates around center of sphere:
        x = self._Rcen - R * math.cos(math.radians(phip))
        y = R * math.sin(math.radians(phip))

        # Pythagorean theorem to test if this point 
        # is inside of observed volume:
        if (x**2 + y**2 + z**2) > self._dmax**2:
            return 0.   #outside
        else:
            return 1.   #inside

    #-----------------------------------------------------------------------

    def _deltaphi_max_rad(self,R,z):
        """largest possible phi at given radius and height"""

        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            return numpy.array([self._deltaphi_max_rad(rr,zz) for rr,zz in zip(R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            return numpy.array([self._deltaphi_max_rad(rr,zz) for rr,zz in zip(R,z)])

        if numpy.fabs(z) > self._dmax:
            return 0. 

        rc = math.sqrt(self._dmax**2 - z**2)  #radius of circle around sphere at height z
       
        rmax = self._Rcen + rc
        rmin = self._Rcen - rc
        if R < rmin or R > rmax:
            return 0. 

        cosphi = (R**2 - rc**2 + self._Rcen**2) / (2. * self._Rcen * R)
        phimax = numpy.fabs(numpy.arccos(cosphi))  #rad
        return phimax

    #-----------------------------------------------------------------------

    def _zmaxfunc(self,R,phi):
        """largest possible z at given radius and phi"""

        #rotate x axis to go through center of sphere:
        phip = phi - self._phicen_deg

        x = self._Rcen - R * numpy.cos(numpy.radians(phip))
        y = R * numpy.sin(numpy.radians(phip))
        if x**2 + y**2 > self._dmax**2:
            return 0.
        zmax = numpy.sqrt(self._dmax**2 - x**2 - y**2)
        return zmax


    #-----------------------------------------------------------------------

    def _densfunc(self,R,z,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False):

        if self._densInterp is None:
            sys.exit("Error in SF_Sphere._densfunc(): "+\
                     "self._densInterp is None. Initialize density grid"+\
                     " first before calling this function.")

        if consider_incompleteness and self._with_incompleteness:
            sys.exit("Error in SF_Sphere._densfunc(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #take care of array input:
        if isinstance(z,numpy.ndarray) or isinstance(R,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)

        #outside of obseved volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_Sphere._densfunc(): "+\
               "Specify phi in case of the spherical selection function, "+\
               "when using throw_error_outside=True or set_outside_zero=True.") 

            #rotate x axis to go through center of sphere:
            phip = phi - self._phicen_deg

            x = self._Rcen - R * numpy.cos(numpy.radians(phip))
            y = R * numpy.sin(numpy.radians(phip))

            outside = (x**2 + y**2 + z**2) > self._dmax**2

            if numpy.sum(outside) > 0:
                if throw_error_outside:
                    print "x^2+y^2+z^2 = ",(x[outside][0])**2 + (y[outside][0])**2 + (z[outside][0])**2
                    print "d_max^2     = ",self._dmax**2
                    print "R: ",self._Rmin," <= ",R[outside][0]," <= ",self._Rmax,"?"
                    print "z: ",self._zmin," <= ",z[outside][0]," <= ",self._zmax,"?"
                    dphi = numpy.degrees(0.5 * self._deltaphi_max_rad(R[outside][0],z[outside][0]))
                    print "phi: ",self._phicen_deg-dphi," <= ",phi[outside][0]," <= ",self._phicen_deg+dphi,"?"
                    print hahaha
                    sys.exit("Error in SF_Sphere._densfunc(). If yes, something is wrong. Testing of code is required.")
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
                sys.exit("Error in SF_Sphere._densfunc(): something with the array output is wrong.")
        return d

    #-----------------------------------------------------------------

    def _fastGLint_sphere(self,func,xgl,wgl):

        """integrate the given function func(R,z,phi) over the spherical effective volume 
           by hand, analogous to Bovy, using Gauss Legendre quadrature."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._fastGLint_sphere(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #R coordinates: R_j = 0.5 * (Rmax - Rmin) * (xgl_j + 1) + Rmin:
        Rgl_j = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin

        #account for integration limits and Jacobian in weights w_i and w_j:
        zmaxRgl_j = numpy.sqrt(self._dmax**2 - (self._Rcen - Rgl_j)**2)     #maximum height z_max of sphere at phi=0 at R=Rgl
        w_j = (self._Rmax - self._Rmin) * zmaxRgl_j * wgl    #account for integration limits
        w_j *= Rgl_j #account for cylindrical coordinates Jacobian
        w_i = wgl 

        #mesh everything:
        ngl = len(xgl)
        Rglm_j  = numpy.zeros((ngl,ngl))
        zglm_ij = numpy.zeros((ngl,ngl))
        wm_i    = numpy.zeros((ngl,ngl))
        wm_j    = numpy.zeros((ngl,ngl))
        for ii in range(ngl):
            for jj in range(ngl):
                Rglm_j[ii,jj] = Rgl_j[jj]
                zglm_ij[ii,jj] = 0.5 * zmaxRgl_j[jj] * (xgl[ii] + 1.)   #z coordinates: zgl_ij = 0.5 * zmax(Rgl_j) * (xgl_i + 1)
                wm_i[ii,jj] = w_i[ii]
                wm_j[ii,jj] = w_j[jj]

        #flatten everything:
        wm_i = wm_i.flatten()
        wm_j = wm_j.flatten()
        Rglm_j = Rglm_j.flatten()
        zglm_ij = zglm_ij.flatten()

        #phi coordinates (dummy):
        phi_ij = numpy.zeros_like(Rglm_j) + self._phicen_deg

        #evaluate function at each grid point:
        func_ij = func(Rglm_j,zglm_ij,phi_ij)

        #angular extend at each grid point:
        phimaxm_ij = self._deltaphi_max_rad(Rglm_j,zglm_ij)

        #total:
        tot = numpy.sum(wm_i * wm_j * func_ij * phimaxm_ij)
        return tot



    #-----------------------------------------------------------------

    def _Mtot_fastGL(self,xgl,wgl):

        """integrate total mass inside effective volume by hand, analogous to Bovy, using Gauss Legendre quadrature.
           The integration accounts for integration limits - we therefore do not have to set the density outside the sphere to zero."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._Mtot_fastGL(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #define function func(R,z,phi) to integrate:
        func = lambda rr,zz,pp: self._densfunc(rr,zz,phi=pp,set_outside_zero=False,throw_error_outside=True,consider_incompleteness=False)
        
        #total mass in selection function:
        Mtot = self._fastGLint_sphere(func,xgl,wgl)
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
                recalculate=recalc_densgrid
                )
        # *Note:* The use of the flag "recalc_densgrid" is actually not 
        # needed here, because setting the analogous flag "recalculate" 
        # in densityGrid should have the same effect.

        #maximum of density:
        zprime = min(math.fabs(self._zmin),math.fabs(self._zmax))
        if self._zmin < 0. and self._zmax > 0.:
            zprime = 0.
        densmax = self._df.density(self._Rmin,zprime,ngl=ngl_vel,nsigma=n_sigma,vTmax=vT_galpy_max)
        #print densmax
        #print self._densfunc(self._Rmin,zprime,phi=self._phicen_deg)
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

            #spherical selection function:
            psi = 2. * math.pi * eta[0] #azimuth angle within sphere, uniformly distributed [rad]
            rc = (eta[1])**(1./3.) * self._dmax #radius in spherical coordinates, distributed according to p(rc) ~ rc^2
            theta = math.asin(2. * eta[2] - 1.)#altitute angle, distributed according to p(theta) ~ cos(theta), [rad]

            #transformation to (R,phi,z):
            x = self._Rcen - rc * math.cos(psi) * math.cos(theta)
            y = rc * math.sin(psi) * math.cos(theta)
            z = rc * math.sin(theta)

            R = math.sqrt(x**2 + y**2)
            phi = math.atan2(y,x)   #rad
            phi = math.degrees(phi) #deg
            phi = phi + self._phicen_deg #rotate x axis to go through center of sphere

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
                if not quiet: 
                    print nreject," rejected"

        return numpy.array(Rarr),numpy.array(zarr),numpy.array(phiarr)

    #-----------------------------------------------------------------------------

    def _dMdzdR(self,z,*args):

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._dMdzdR(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        R = args[0]
        phimax = self._deltaphi_max_rad(R,z)    #radians
        return 2. * phimax * R * self._densfunc(R,z,phi=self._phicen_deg,set_outside_zero=True,consider_incompleteness=False)
       

    def _dMdR(self,R,ngl=20):
        
        if isinstance(R,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdR(rr,ngl=ngl) for rr in R])

        if R > self._Rmax or R < self._Rmin:
            return 0.

        zmax = self._zmaxfunc(R,0.)        
        #by integrating up to this zmax at every phi, these integration limits cover also ranges that are outside of the sphere,
        #especially at larger phi. The self._densfunc function returns 0 if outside of the observed volume, 
        #therefore this does not affect the result.

        dM = 2. * (scipy.integrate.fixed_quad(self._dMdzdR,
                                       0.,zmax,args=[R],n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------

    def _dMdRdz(self,R,*args):

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._dMdRdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        z = args[0]
        phimax = self._deltaphi_max_rad(R,z)    #radians
        return 2. * phimax * R * self._densfunc(R,z,phi=self._phicen_deg,set_outside_zero=True,consider_incompleteness=False)


    def _dMdz(self,z,ngl=20):
        
        if isinstance(z,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdz(zz,ngl=ngl) for zz in z])


        dM = (scipy.integrate.fixed_quad(self._dMdRdz,
                                         self._Rmin,self._Rmax,args=[z],n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------

    def _surfDens(self,R,phi=None,ngl=20):

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._surfDens(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        if phi is None: sys.exit("Error in SF_Sphere._surfDens(): Specify phi.")
        
        if isinstance(R,numpy.ndarray):
            #Recursion if input is array:
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(R)
            return numpy.array([self._surfDens(rr,phi=pp,ngl=ngl) for rr,pp in zip(R,phi)])
        elif isinstance(phi,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(phi)
            return numpy.array([self._surfDens(rr,phi=pp,ngl=ngl) for rr,pp in zip(R,phi)])

        zmax = self._zmaxfunc(R,phi)

        dM = 2. * (scipy.integrate.fixed_quad(lambda zz: self._densfunc(R,zz,phi=phi,set_outside_zero=True,throw_error_outside=False,consider_incompleteness=False),
                                              0.,zmax,n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------  


    def _dMdphi(self,phi,ngl=20):
        
        if isinstance(phi,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdphi(pp,ngl=ngl) for pp in phi])

        if (phi > self._pmax) or (phi < self._pmin):
            return 0. 


        #rotate x axis to go through center of sphere:
        phip = phi - self._phicen_deg

        #the small and large radius at which a secant under the angle phi 
        #intersects the circle around the sun (in the galactic plane only):
        d = self._Rcen
        cp = numpy.cos(numpy.radians(phip))
        sp = numpy.sin(numpy.radians(phip))
        r = self._dmax
        Rlow = d * cp - numpy.sqrt(r**2 - d**2 * sp**2)
        Rup  = d * cp + numpy.sqrt(r**2 - d**2 * sp**2)


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
        dM = self._fastGLint_sphere(pvfunc,xgl,wgl)
        return dM

    #-----------------------------------------------------------------

    def _spatialSampleDF_measurementErrors(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,
                                           e_radec_rad=None,e_DM_mag=None,
                                           Xsun_kpc=8.,Ysun_kpc=0.,Zsun_kpc=0.,
                                           spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.):

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._spatialSampleDF_measurementErrors(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #_____calculate size of envelope volume_____

        #radius of sphere around sun in [kpc]:
        dmax_kpc = self._dmax * spatialGalpyUnits_in_kpc
        #corresponding distance modulus: 
        DM_mag = 5. * numpy.log10(dmax_kpc * 1000.) - 5.
        #add 3*sigma to distance modulus as an envelope:
        DM_mag += 3. * e_DM_mag
        #corresponding radius of envelope sphere in galpy units:
        renv_kpc = 10.**(0.2 * DM_mag + 1.) * 10.**(-3)
        dmax_env = renv_kpc / spatialGalpyUnits_in_kpc
        #take care of possible negative radius:
        if dmax_env > self._Rcen:
            sys.exit("Error in SF_Sphere._spatialSampleDF_measurementErrors(): Envelope sphere reaches beyond the Galactic center. Change code! This should not happen!")
        print "Radius of observed volume: ",dmax_kpc," kpc, Radius of envelope volume:", renv_kpc," kpc"

        #_____create new selection function as envelope_____
        #envelope selection function:
        sf_env = SF_Sphere(dmax_env,self._Rcen,df=self._df)
        #initialize interpolated density grid:
        if not quiet: print "Initialize interpolated density grid"
        sf_env.densityGrid(
                nrs_nonfid=nrs,
                nzs_nonfid=nzs,
                ngl_vel_nonfid=ngl_vel,
                n_sigma=n_sigma,
                vT_galpy_max=vT_galpy_max,
                test_sf=test_sf,
                _multi=_multi,
                recalculate=True
                )
        
        #_____prepare loop_____
        #number of found mockdata:
        nfound       = 0
        nreject      = 0
        #output:
        R_kpc_true    = []
        z_kpc_true    = []
        phi_deg_true  = []
        R_kpc_error   = []
        z_kpc_error   = []
        phi_deg_error = []
        ra_rad_error  = []
        dec_rad_error = []
        DM_mag_error  = []
        ra_rad_true   = []
        dec_rad_true  = []
        DM_mag_true   = []

        if not quiet: print "Start sampling"

        while nfound < nmock:

            # how many more mock data points are needed:
            nmore = nmock - nfound

            #_____sample spatial mock data from envelope sphere:_____
            # (*Note:* recalc_densgrid is set explicitely to False, so 
            #          we use each time the already calculated grid.)
            R_galpy,z_galpy,phi_deg_t = sf_env.spatialSampleDF(
                                        nmock=nmore,
                                        ngl_vel=ngl_vel,    #used to calculate the peak density
                                        n_sigma=n_sigma,    #-- " --
                                        vT_galpy_max=vT_galpy_max,  #-- " --
                                        recalc_densgrid=False,
                                        quiet=True
                                        )
            R_kpc_t = R_galpy * spatialGalpyUnits_in_kpc
            z_kpc_t = z_galpy * spatialGalpyUnits_in_kpc
            # (Note: t for true.)

            #_____transform to observable coordinate system_____
            # (R,z,phi) around Galactic center --> (ra,dec,DM) around sun:
            phi_rad_t = phi_deg_t * math.pi / 180.
            radecDM = galcencyl_to_radecDM(
                                    R_kpc_t, 
                                    phi_rad_t, 
                                    z_kpc_t,
                                    quiet=True,
                                    Xsun_kpc=Xsun_kpc,
                                    Ysun_kpc=Ysun_kpc,
                                    Zsun_kpc=Zsun_kpc)
            ra_rad_t  = radecDM[0]
            dec_rad_t = radecDM[1]
            DM_mag_t  = radecDM[2]


            #_____perturb according to measurement errors_____
            if e_radec_rad is None or e_DM_mag is None:
                sys.exit("Error in SF_Sphere."+\
                         "_spatialSampleDF_measurementErrors(): errors"+\
                         " on right ascension / declination "+\
                         "(e_radec_rad) and on distance modulus "+\
                         "(e_DM_mag) have to be set.")

            # Draw random numbers for random gaussian errors:
            eta = numpy.random.randn(3,nmore)

            #Perturb data according to Gaussian distribution:
            # (Note: p for perturbed.)
            ra_rad_p      = eta[0,:] * e_radec_rad + ra_rad_t
            dec_rad_p     = eta[1,:] * e_radec_rad + dec_rad_t
            DM_mag_p      = eta[2,:] * e_DM_mag    + DM_mag_t


            #_____transform back to cylindrical coordinates_____
            # (ra,dec,DM) --> (R,z,phi):
            RphiZ = radecDM_to_galcencyl(
                         ra_rad_p,dec_rad_p,DM_mag_p,
                         quiet=True,
                         Xsun_kpc=Xsun_kpc,
                         Ysun_kpc=Ysun_kpc,
                         Zsun_kpc=Zsun_kpc
                         )
            R_kpc_p   = numpy.array([RphiZ[0]]).reshape(nmore)
            phi_rad_p = numpy.array([RphiZ[1]]).reshape(nmore)
            z_kpc_p   = numpy.array([RphiZ[2]]).reshape(nmore)
            phi_deg_p = phi_rad_p * 180. / math.pi

            #_____reject those outside of true observed volume_____
            # (DM) --> (d):
            d_kpc_p = 10.**(0.2 * DM_mag_p + 1.) * 10.**(-3)
            # reject:
            index = numpy.array(d_kpc_p <= dmax_kpc,dtype=bool)
            nfound += numpy.sum(index)
            nreject += nmore - numpy.sum(index)
            if not quiet: print "Found: ",nfound, ", Reject: ",nreject

            # add found coordinates to list:
            if numpy.sum(index) > 0:
                if index.shape[0] == 1 and index[0]:
                    R_kpc_true   .extend([R_kpc_t  ])
                    z_kpc_true   .extend([z_kpc_t  ])
                    phi_deg_true .extend([phi_deg_t])
                    R_kpc_error  .extend([R_kpc_p  ])
                    z_kpc_error  .extend([z_kpc_p  ])
                    phi_deg_error.extend([phi_deg_p])
                    ra_rad_error .extend([ra_rad_p ])
                    dec_rad_error.extend([dec_rad_p])
                    DM_mag_error .extend([DM_mag_p ])
                    ra_rad_true  .extend([ra_rad_t ])
                    dec_rad_true .extend([dec_rad_t])
                    DM_mag_true  .extend([DM_mag_t ])

                else:
                    R_kpc_true   .extend(R_kpc_t  [index])
                    z_kpc_true   .extend(z_kpc_t  [index])
                    phi_deg_true .extend(phi_deg_t[index])
                    R_kpc_error  .extend(R_kpc_p  [index])
                    z_kpc_error  .extend(z_kpc_p  [index])
                    phi_deg_error.extend(phi_deg_p[index])
                    ra_rad_error .extend(ra_rad_p [index])
                    dec_rad_error.extend(dec_rad_p[index])
                    DM_mag_error .extend(DM_mag_p [index])
                    ra_rad_true  .extend(ra_rad_t [index])
                    dec_rad_true .extend(dec_rad_t[index])
                    DM_mag_true  .extend(DM_mag_t [index])


        return (numpy.array(R_kpc_true),  numpy.array(z_kpc_true),   numpy.array(phi_deg_true), \
                numpy.array(R_kpc_error), numpy.array(z_kpc_error),  numpy.array(phi_deg_error), \
                numpy.array(ra_rad_error),numpy.array(dec_rad_error),numpy.array(DM_mag_error), \
                numpy.array(ra_rad_true),numpy.array(dec_rad_true),numpy.array(DM_mag_true))
