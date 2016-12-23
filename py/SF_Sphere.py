from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt
from coord_trafo import galcencyl_to_radecDM, radecDM_to_galcencyl, galcencyl_to_radecDMvlospmradec

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
            zcen: height of center over Galactic plane [_REFR0*ro] --> Nonzero does not work yet everywhere.
            phicen_deg: azimuth of center [deg]
        OUTPUT:
            spherical selection function object
        HISTORY:
            2015-11-30 - Started SF_Sphere.py on the basis of BovyCode/py/SF_Sphere.py - Trick (MPIA)
                       - Renamed rsun (radius of sphere) --> dmax, dsun --> Rcen, _phimax_rad() --> _deltaphi_max_rad()
        """
        SelectionFunction.__init__(self,df=df)

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

        """
            NAME:
               _contains
            PURPOSE:
                Tests whether a list of point is inside the sphere or not.
            INPUT:
            OUTPUT:
                contains - (float array) - 0 if outside of sphere, 1 if inside of sphere
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-22 - Allowed this test to account for a flexible center of the spherical survey volume. - Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._contains(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        # Recursion if input is array:
        if phi is None: sys.exit("Error in SF_Sphere._contains(): Always specify phi in case of the spherical selection function.")
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(R)
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(phi,numpy.ndarray): phi = phi + numpy.zeros_like(z)
        elif isinstance(phi,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(phi)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(phi)
        else:
            sys.exit("Error in SF_Sphere._contains(): This function is not implemented for scalar input.")
        if numpy.any(numpy.array([len(z),len(phi)]) != len(R)):
            print numpy.shape(R),numpy.shape(z),numpy.shape(phi)
            sys.exit("Error in SF_Sphere._contains(): Input arrays do not have the same length.")

      

        # rotate x axis to go through center of sphere:
        phip = phi - self._phicen_deg

        # transform from cylindrical galactocentric coordinates
        # to rectangular coordinates around center of sphere:
        xp = self._Rcen - R * numpy.cos(numpy.radians(phip))
        yp = R * numpy.sin(numpy.radians(phip))
        zp = z - self._zcen

        # Pythagorean theorem to test if this point 
        # is inside or outside of observed volume:
        index_outside_SF = (xp**2 + yp**2 + zp**2) > self._dmax**2 
        contains = numpy.ones_like(R)   #return 1 if inside
        contains[index_outside_SF] = 0. #return 0 if outside
        return contains

    #-----------------------------------------------------------------------

    def _deltaphi_max_rad(self,R,z):
        """
            NAME:
               _deltaphi_max_rad
            PURPOSE:
                largest possible phi at given radius and height
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-22 - Allowed this function to account for a flexible center of the spherical survey volume. - Trick (MPIA)
        """

        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            return numpy.array([self._deltaphi_max_rad(rr,zz) for rr,zz in zip(R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            return numpy.array([self._deltaphi_max_rad(rr,zz) for rr,zz in zip(R,z)])

        if numpy.fabs(z-self._zcen) > self._dmax:
            return 0. 

        rc = numpy.sqrt(self._dmax**2 - (z-self._zcen)**2)  #radius of circle around sphere at height z, pythagoras
       
        rmax = self._Rcen + rc
        rmin = self._Rcen - rc
        if (R < rmin) or (R > rmax):
            return 0. 

        cosphi = (R**2 - rc**2 + self._Rcen**2) / (2. * self._Rcen * R) #law of cosines
        phimax_rad = numpy.fabs(numpy.arccos(cosphi))  #[rad]
        return phimax_rad

    #-----------------------------------------------------------------------

    def _zmaxfunc(self,R,phi):
        """largest possible z at given radius and phi"""

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._zmaxfunc(): Nonzero zcen is not implemented yet.")

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

        """
            NAME:
               _densfunc
            PURPOSE:
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-19 - Allowed this function to account for a flexible center of the spherical survey volume. - Trick (MPIA)
        """

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

        #outside of observed volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_Sphere._densfunc(): "+\
               "Specify phi in case of the spherical selection function, "+\
               "when using throw_error_outside=True or set_outside_zero=True.") 

            #rotate x axis to go through center of sphere:
            phip = phi - self._phicen_deg

            xp = self._Rcen - R * numpy.cos(numpy.radians(phip))
            yp = R * numpy.sin(numpy.radians(phip))
            zp = z - self._zcen
            rp2 = (xp**2 + yp**2 + zp**2)

            outside = rp2 > self._dmax**2

            if numpy.sum(outside) > 0:
                if throw_error_outside:
                    print "x^2+y^2+z^2 = ",(xp[outside][0])**2 + (yp[outside][0])**2 + (zp[outside][0])**2
                    print "d_max^2     = ",self._dmax**2
                    print "R: ",self._Rmin," <= ",R[outside][0]," <= ",self._Rmax,"?"
                    print "z: ",self._zmin," <= ",z[outside][0]," <= ",self._zmax,"?"
                    dphi = numpy.degrees(0.5 * self._deltaphi_max_rad(R[outside][0],z[outside][0]))
                    print "phi: ",self._phicen_deg-dphi," <= ",phi[outside][0]," <= ",self._phicen_deg+dphi,"?"
                    print hahaha
                    sys.exit("Error in SF_Sphere._densfunc(): If yes, something is wrong. Testing of code is required.")
                if set_outside_zero:
                    sys.exit("Error in SF_Sphere._densfunc(): If set_outside_zero=True, taking care of array input is not implemented yet.")
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

        """
            NAME:
               _fastGLint_sphere
            PURPOSE:
                integrate the given function func(R,z,phi) over the spherical effective volume 
                by hand, analogous to Bovy, using Gauss Legendre quadrature.
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-22 - Allowed this function to account for a flexible center of the spherical survey volume. - Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._fastGLint_sphere(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        if self._zcen == 0.:

            #...original version: center of sphere is at z=0...
            #...(keep for now to conserve original code)...

            #R coordinates: R_j = 0.5 * (Rmax - Rmin) * (xgl_j + 1) + Rmin:
            Rgl_j = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin

            #account for integration limits and Jacobian in weights w_i and w_j:
            zmaxRgl_j = numpy.sqrt(self._dmax**2 - (self._Rcen - Rgl_j)**2)     #maximum height z_max of sphere at phi=0 at R=Rgl
            w_j = (self._Rmax - self._Rmin) * zmaxRgl_j * wgl    #account for integration limits; factor of 2 in (2*zmax) has cancelled with factor 0.5 from GL-prefactor of integration in z
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
            phimaxm_ij = self._deltaphi_max_rad(Rglm_j,zglm_ij) #no factor of 2, because it has cancelled with the factor 0.5 from GL-prefactor of integration in R

            #total:
            tot = numpy.sum(wm_i * wm_j * func_ij * phimaxm_ij)
            return tot

        else:

            #...new version: center of sphere can be at z != 0...

            #R coordinates: R_j = 0.5 * (Rmax - Rmin) * (xgl_j + 1) + Rmin:
            Rgl_j = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin

            #account for integration limits and Jacobian in weights w_i and w_j:
            zmaxRgl_j = self._zcen + numpy.sqrt(self._dmax**2 - (self._Rcen - Rgl_j)**2)     #maximum height z_max of sphere at phi=0 at R=Rgl
            zminRgl_j = self._zcen - numpy.sqrt(self._dmax**2 - (self._Rcen - Rgl_j)**2)     #minimum height z_max of sphere at phi=0 at R=Rgl
            w_j = 0.25 * (self._Rmax - self._Rmin) * (zmaxRgl_j - zminRgl_j) * wgl    #account for integration limits over R and z; GL-prefactors 0.5*0.5=0.25 of integration over R and z
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
                    zglm_ij[ii,jj] = 0.5 * (zmaxRgl_j[jj]-zminRgl_j[jj]) * (xgl[ii] + 1.) + zminRgl_j[jj]   #z coordinates: zgl_ij = 0.5 * [zmax(Rgl_j)-zmin(Rgl_j)] * (xgl_i + 1) + zmin(Rgl_j) = 0.5 * (zmax-zmin) * x + 0.5 * (zmax+zmin)
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
            phimaxm_ij = 2. * self._deltaphi_max_rad(Rglm_j,zglm_ij)    #delta phi is only from center of sphere to edge, therefore factor 2

            #total:
            tot = numpy.sum(wm_i * wm_j * func_ij * phimaxm_ij)
            return tot



    #-----------------------------------------------------------------

    def _Mtot_fastGL(self,xgl,wgl):

        """
            NAME:
               _Mtot_fastGL
            PURPOSE:
                integrate total mass inside effective volume by hand, analogous to Bovy, using Gauss Legendre quadrature.
                The integration accounts for integration limits - we therefore do not have to set the density outside the sphere to zero.
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
        """

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

        """
            NAME:
               _spatialSampleDF_complete
            PURPOSE:
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-15 - Allowed this sampling to have a flexible center of the spherical survey volume. - Trick (MPIA)
        """

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
            z = rc * math.sin(theta) + self._zcen   #move sphere to sit on vertical position of the center

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

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdzdR(): Nonzero zcen is not implemented yet.")

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._dMdzdR(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        R = args[0]
        phimax = self._deltaphi_max_rad(R,z)    #radians
        return 2. * phimax * R * self._densfunc(R,z,phi=self._phicen_deg,set_outside_zero=True,consider_incompleteness=False)

    #-----------------------------------------------------------------------------       

    def _dMdR(self,R,ngl=20):

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdR(): Nonzero zcen is not implemented yet.")
        
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

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdRdz(): Nonzero zcen is not implemented yet.")

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._dMdRdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        z = args[0]
        phimax = self._deltaphi_max_rad(R,z)    #radians
        return 2. * phimax * R * self._densfunc(R,z,phi=self._phicen_deg,set_outside_zero=True,consider_incompleteness=False)

    #-----------------------------------------------------------------------------     

    def _dMdz(self,z,ngl=20):

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdz(): Nonzero zcen is not implemented yet.")
        
        if isinstance(z,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdz(zz,ngl=ngl) for zz in z])


        dM = (scipy.integrate.fixed_quad(self._dMdRdz,
                                         self._Rmin,self._Rmax,args=[z],n=ngl))[0]
        return dM

    #-----------------------------------------------------------------------------

    def _surfDens(self,R,phi=None,ngl=20):

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._surfDens(): Nonzero zcen is not implemented yet.")

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

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdphi(): Nonzero zcen is not implemented yet.")
        
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

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._dMdv(): Nonzero zcen is not implemented yet.")

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

        """
            NAME:
               _spatialSampleDF_measurementErrors
            PURPOSE:
            INPUT:
            OUTPUT:
            HISTORY:
               201?-??-?? - Written. - Trick (MPIA)
               2016-12-15 - Allowed this sampling to have a flexible center of the spherical survey volume. - Trick (MPIA)
        """

        if self._zcen != 0.:
            sys.exit("Error in SF_Sphere._spatialSampleDF_measurementErrors(): Nonzero zcen is not implemented yet.")

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
        sf_env = SF_Sphere(dmax_env,self._Rcen,zcen=self._zcen,phicen_deg=self._phicen_deg,df=self._df)
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

#-----------------------------------------------------------------

    def _sampleDF_correlatedMeasurementErrors(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,_multi=None,
                                           e_ra_deg=None,e_dec_deg=None,e_d_kpc=None,e_pmra_masyr=None,e_pmdec_masyr=None,e_vlos_kms=None,
                                           corr_ra_dec=None,corr_ra_pmra=None,corr_ra_pmdec=None,corr_dec_pmra=None,corr_dec_pmdec=None,corr_pmra_pmdec=None,
                                           Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.,
                                           vXgc_sun_kms=0.,vYgc_sun_kms=230.,vZgc_sun_kms=0.,
                                           spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.):

        """
            NAME:
               _sampleDF_correlatedMeasurementErrors
            PURPOSE:
                samples mock data with correlated measurement errors. 
                So far the distance and v_los measurement errors are 
                independent of the other measurement errors - like 
                those for the TGAS red clump sample, with RAVE radial 
                velocities and photometric distances.
            INPUT:
            OUTPUT:
            HISTORY:
               2016-12-15 - Written. - Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SF_Sphere._sampleDF_correlatedMeasurementErrors(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #_____calculate size of envelope volume_____

        #radius of sphere around sun in [kpc]:
        dmax_kpc = self._dmax * spatialGalpyUnits_in_kpc
        #add 3*sigma to maximum radius as envelope:
        dmax_env_kpc = dmax_kpc + 3. * e_d_kpc
        #corresponding radius of envelope sphere in galpy units:
        dmax_env = dmax_env_kpc / spatialGalpyUnits_in_kpc
        #take care of possible negative radius:
        if dmax_env > self._Rcen:
            sys.exit("Error in SF_Sphere._sampleDF_correlatedMeasurementErrors(): Envelope sphere reaches beyond the Galactic center. Change code! This should not happen!")
        print "Radius of observed volume: ",dmax_kpc," kpc, Radius of envelope volume:", dmax_env_kpc," kpc"

        #_____create new selection function as envelope_____
        #envelope selection function:
        sf_env = SF_Sphere(dmax_env,self._Rcen,zcen=self._zcen,phicen_deg=self._phicen_deg,df=self._df)
        #initialize interpolated density grid:
        if not quiet: print "Initialize interpolated density grid"
        sf_env.densityGrid(
                nrs_nonfid=nrs,
                nzs_nonfid=nzs,
                ngl_vel_nonfid=ngl_vel,
                n_sigma=n_sigma,
                vT_galpy_max=vT_galpy_max,
                test_sf=False,
                _multi=_multi,
                recalculate=True
                )
        
        #_____prepare loop_____
        #number of found mockdata:
        nfound       = 0
        nreject      = 0
        #output:
        out_true  = numpy.zeros((6,nmock))
        out_error = numpy.zeros((6,nmock))

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

            #_____draw true velocities from df for true positions_____
            Rs = R_kpc_t / spatialGalpyUnits_in_kpc
            zs = z_kpc_t / spatialGalpyUnits_in_kpc

            # actual mock data sampling:
            vRs,vTs,vzs = self.velocitySampleDF(Rs,zs,_multi=_multi,test_sf=False) 

            vR_kms_t = vRs * velocityGalpyUnits_in_kms
            vT_kms_t = vTs * velocityGalpyUnits_in_kms
            vz_kms_t = vzs * velocityGalpyUnits_in_kms

            #_____convert to observable coordinates_____
            phi_rad_t = phi_deg_t / 180. * math.pi
            out = galcencyl_to_radecDMvlospmradec(
                                        R_kpc_t, phi_rad_t, z_kpc_t,
                                        vR_kms_t,vT_kms_t,  vz_kms_t,
                                        quiet=True,
                                        Xgc_sun_kpc=Xgc_sun_kpc,
                                        Ygc_sun_kpc=Ygc_sun_kpc,
                                        Zgc_sun_kpc=Zgc_sun_kpc,
                                        vXgc_sun_kms=vXgc_sun_kms,
                                        vYgc_sun_kms=vYgc_sun_kms,
                                        vZgc_sun_kms=vZgc_sun_kms
                                        )
            ra_rad_t       = out[0]
            dec_rad_t      = out[1]
            DM_mag_t       = out[2]
            vlos_kms_t     = out[3]
            pmra_masyr_t   = out[4]
            pmdec_masyr_t  = out[5]
            rad2deg        = 180./math.pi
            ra_deg_t       = ra_rad_t * rad2deg
            dec_deg_t      = dec_rad_t * rad2deg
            d_kpc_t        = 10.**(0.2*DM_mag_t-2.)

            #_____setup covariance matrix for measurement errors_____
            mean = numpy.zeros((nmore,6))
            std = numpy.zeros(6)
            cov = numpy.zeros((6,6))

            #6D coordinates:
            mean[:,0] = ra_deg_t            #right ascension [deg]
            mean[:,1] = dec_deg_t           #declination [deg]
            mean[:,2] = d_kpc_t             #distance [kpc]
            mean[:,3] = pmra_masyr_t        #RA proper motion [mas/yr] 
            mean[:,4] = pmdec_masyr_t       #DEC proper motion [mas/yr]
            mean[:,5] = vlos_kms_t          #heliocentric radial velocity [km/s]

            #standard deviation:
            std[0] = e_ra_deg           #[deg]
            std[1] = e_dec_deg          #[deg]
            std[2] = e_d_kpc            #[kpc]
            std[3] = e_pmra_masyr       #[mas/yr]
            std[4] = e_pmdec_masyr      #[mas/yr]
            std[5] = e_vlos_kms         #[km/s]

            #variances & correlations:
            cov[0,0] = std[0]**2
            cov[1,1] = std[1]**2
            cov[2,2] = std[2]**2
            cov[3,3] = std[3]**2
            cov[4,4] = std[4]**2
            cov[5,5] = std[5]**2

            cov[0,1] = corr_ra_dec   * std[0] * std[1]
            cov[0,2] = 0. #no correlation between RA & d
            cov[0,3] = corr_ra_pmra  * std[0] * std[3]
            cov[0,4] = corr_ra_pmdec * std[0] * std[4]
            cov[0,5] = 0. #no correlation between RA & vlos

            cov[1,2] = 0. #no correlation between DEC & d
            cov[1,3] = corr_dec_pmra  * std[1] * std[3]
            cov[1,4] = corr_dec_pmdec * std[1] * std[4]
            cov[1,5] = 0. #no correlation between DEC & vlos

            cov[2,3] = 0. #no correlation between d & pmra
            cov[2,4] = 0. #no correlation between d & pmdec
            cov[2,5] = 0. #no correlation between d & vlos

            cov[3,4] = corr_pmra_pmdec * std[3] * std[4]
            cov[3,5] = 0. #no correlation between pmra & vlos

            cov[4,5] = 0. #no correlation between pmdec & vlos

            cov[1,0] = cov[0,1]
            cov[2,0] = cov[0,2]
            cov[3,0] = cov[0,3]
            cov[4,0] = cov[0,4]
            cov[5,0] = cov[0,5]
            cov[2,1] = cov[1,2]
            cov[3,1] = cov[1,3]
            cov[4,1] = cov[1,4]
            cov[5,1] = cov[1,5]
            cov[3,2] = cov[2,3]
            cov[4,2] = cov[2,4]
            cov[5,2] = cov[2,5]
            cov[4,3] = cov[3,4]
            cov[5,3] = cov[3,5]
            cov[5,4] = cov[4,5]

            #_____perturb according to covariance error matrix_____
            #Perturb data according to multivariate Gaussian distribution:
            # (Note: p for perturbed.)
            ra_deg_p      = numpy.zeros(nmore)
            dec_deg_p     = numpy.zeros(nmore)
            d_kpc_p       = numpy.zeros(nmore)
            pmra_masyr_p  = numpy.zeros(nmore)
            pmdec_masyr_p = numpy.zeros(nmore)
            vlos_kms_p    = numpy.zeros(nmore)
            if numpy.sum(numpy.isfinite([e_ra_deg,e_dec_deg,e_d_kpc,e_pmra_masyr,e_pmdec_masyr,e_vlos_kms,corr_ra_dec,corr_ra_pmra,corr_ra_pmdec,corr_dec_pmra,corr_dec_pmdec,corr_pmra_pmdec])) != 12:
                sys.exit("Error in SF_Sphere."+\
                         "_sampleDF_correlatedMeasurementErrors(): errors"+\
                         " and error correlations have to be set.")
            for ii in range(nmore):
                err = numpy.random.multivariate_normal(mean[ii,:], cov)
                ra_deg_p[ii]      = err[0]
                dec_deg_p[ii]     = err[1]
                d_kpc_p[ii]       = err[2]
                pmra_masyr_p[ii]  = err[3]
                pmdec_masyr_p[ii] = err[4]
                vlos_kms_p[ii]    = err[5]

            #_____reject those outside of true observed volume_____
            # reject:
            index = numpy.array(d_kpc_p <= dmax_kpc,dtype=bool)
            nfound_old = nfound
            nfound += numpy.sum(index)
            nreject += nmore - numpy.sum(index)
            if not quiet: print "Found: ",nfound, ", Reject: ",nreject

            # add found coordinates to list:
            if numpy.sum(index) > 0:
                if index.shape[0] == 1 and index[0]:
                    sys.exit('blaaaaaaaaaaaaaaah')
                    out_true[0,nfound_old:nfound] = ra_deg_t
                    out_true[1,nfound_old:nfound] = dec_deg_t
                    out_true[2,nfound_old:nfound] = d_kpc_t
                    out_true[3,nfound_old:nfound] = pmra_masyr_t
                    out_true[4,nfound_old:nfound] = pmdec_masyr_t
                    out_true[5,nfound_old:nfound] = vlos_kms_t

                    out_error[0,nfound_old:nfound] = ra_deg_p
                    out_error[1,nfound_old:nfound] = dec_deg_p
                    out_error[2,nfound_old:nfound] = d_kpc_p
                    out_error[3,nfound_old:nfound] = pmra_masyr_p
                    out_error[4,nfound_old:nfound] = pmdec_masyr_p
                    out_error[5,nfound_old:nfound] = vlos_kms_p

                else:
                    out_true[0,nfound_old:nfound] = ra_deg_t      [index]
                    out_true[1,nfound_old:nfound] = dec_deg_t     [index]
                    out_true[2,nfound_old:nfound] = d_kpc_t       [index]
                    out_true[3,nfound_old:nfound] = pmra_masyr_t  [index]
                    out_true[4,nfound_old:nfound] = pmdec_masyr_t [index]
                    out_true[5,nfound_old:nfound] = vlos_kms_t    [index]

                    out_error[0,nfound_old:nfound] = ra_deg_p     [index]
                    out_error[1,nfound_old:nfound] = dec_deg_p    [index]
                    out_error[2,nfound_old:nfound] = d_kpc_p      [index]
                    out_error[3,nfound_old:nfound] = pmra_masyr_p [index]
                    out_error[4,nfound_old:nfound] = pmdec_masyr_p[index]
                    out_error[5,nfound_old:nfound] = vlos_kms_p   [index]

        return out_true, out_error
