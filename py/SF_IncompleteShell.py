#_____import packages_____
from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt
from coord_trafo import galcencyl_to_radecDM, radecDM_to_galcencyl
import healpy
from galpy.util import bovy_coords
from galpy.util import save_pickles
import matplotlib.pyplot as plt
import colormaps as cmaps

class SF_IncompleteShell(SelectionFunction):
    """
        Class that implements a spherical shell-like selection function 
        around the sun, with some spatial incompleteness substructure
    """
    def __init__(self,dmin,dmax,Rgc_Sun,
                        zgc_Sun=0.,phigc_Sun_deg=0.,
                        df=None,
                        SF_of_hpID_dkpc=None,NSIDE=None,dbin_kpc=None,galpy_to_kpc=None,  
                        SF_of_R_z=None,Rbin_kpc=None,zbin_kpc=None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a shell selection function with incompleteness
        INPUT:
            dmin -- scalar float -- radius of inner edge of shell around center (@ sun) [galpy units]
            dmax -- scalar float -- radius of outer edge of shell around center (@ sun) [galpy units]
            Rgc_Sun -- scalar float -- distance of center (@ sun) to Galactic center [galpy units]
            zgc_Sun -- scalar float -- height of center over Galactic plane [galpy units]
            phigc_Sun_deg -- scalar float -- azimuth of center [deg]
            df -- galpy object of a distribution function
            SF_of_hpID_dkpc -- float array of shape (npix,N) -- completeness selection function, function of healpix_ID (# of pixels is npix) and N bins in distance [kpc]
            NSIDE -- scalar int -- NSIDE=2**level of healpix pixelation of selection function
            dbin_kpc -- float array -- bin edges of distance bins in SF(healpix_ID,d_kpc), shape:(N+1)
            galpy_to_kpc -- scalar float -- transformation of galpy units to kpc, i.e. ro*_REFR0
            SF_of_R_z -- float array of shape (nbin_R,nbin_z) -- selection function integrated over phi and evaluated at (R,z) grid points. Will be interpolated to calculate the likelihood normalisation.
            Rbin_kpc, zbin_kpc -- 1D float arrays -- contains the grid points of the axes of the SF_of_R_z grid.
        OUTPUT:
            shell selection function object
        HISTORY:
            2016-09-20 - Started SF_IncompleteShell.py - Trick (MPIA)
        """
        SelectionFunction.__init__(self,df=df)

        #Edges of the spherical shell:
        self._dmin = dmin
        self._dmax = dmax

        #coordinates of the Sun:
        self._Rsun       = Rgc_Sun
        self._zsun       = zgc_Sun
        self._phisun_deg = phigc_Sun_deg

        #Borders:
        self._Rmin = Rgc_Sun - dmax
        self._Rmax = Rgc_Sun + dmax
        self._zmin = zgc_Sun - dmax
        self._zmax = zgc_Sun + dmax
        self._pmax_deg = phigc_Sun_deg + math.degrees(math.asin(dmax / Rgc_Sun))
        self._pmin_deg = phigc_Sun_deg - math.degrees(math.asin(dmax / Rgc_Sun))

        
        #no incompleteness:
        self._with_incompleteness     = False
        self._incompleteness_function = None
        self._incompleteness_maximum  = None
        self._incomp_SF_interpolated = None

        if SF_of_hpID_dkpc is not None:
            #incompleteness (on healpixel basis):
            self._NSIDE = NSIDE #resolution of healpixels
            self._galpy_to_kpc = galpy_to_kpc
            self._incomp_dbin_kpc  = dbin_kpc #bin edges of distance bins in SF(healpix_ID,d_kpc), shape:(N+1)
            self._incomp_SF_of_hpID_dkpc = SF_of_hpID_dkpc #function of healpix_ID and d_kpc bin, shape: (healpy.nside2npix(NSIDE),N)

            self._with_incompleteness     = True
            self._incompleteness_function = self._aux_incompleteness_function_shell #function of (R,phi_deg,z)
            self._incompleteness_maximum  = max(SF_of_hpID_dkpc.flatten())

            if numpy.shape(SF_of_hpID_dkpc)[0] != healpy.nside2npix(NSIDE):
                sys.exit("Error in SF_IncompleteShell.__init__(): NSIDE and length of 1st axis of completeness array (which should be the number of healpix) do not agree.")
            if numpy.shape(SF_of_hpID_dkpc)[1] != len(dbin_kpc)-1:
                sys.exit("_prepare_incompleteness.__init__(): length of d_dist and length of 2nd axis of completeness array (which should be the number distances - 1) do not agree.")

        if SF_of_R_z is not None:
            #incompleteness:
            self._galpy_to_kpc = galpy_to_kpc
            self._incomp_SF_of_R_z = SF_of_R_z
            self._incomp_Rbin_kpc = Rbin_kpc
            self._incomp_zbin_kpc = zbin_kpc
            self._with_incompleteness = True
            self._incomp_SF_interpolated_Rkpc_zkpc = scipy.interpolate.RectBivariateSpline(
                                        self._incomp_Rbin_kpc,self._incomp_zbin_kpc,
                                        self._incomp_SF_of_R_z,  
                                        kx=3,ky=3,
                                        s=0.
                                        )

        return None

    #-----------------------------------------------------------------------

    def _aux_incompleteness_function_shell(self,R,phi_deg,z):
        """
        NAME:
            _aux_incompleteness_function_shell
        PURPOSE:
            This function converts (R,phi,z) to (l,b,d) and reads out 
            the value of the selection function at this position on 
            basis of a table that stores the completeness as a function 
            of healpixel ID and binned distance.
        INPUT:
            R -- float scalar or array -- cylindrical radius coordinates [galpy units]
            phi_deg -- float scalar or array -- azimuth [deg]
            z -- float scalar or array -- height above the plane [galpy units]
        OUTPUT:
            completeness at (R,phi,z)
        HISTORY:
            2016-09-20 - Started. - Trick (MPIA)
        """

        #scalar vs. array input:
        if isinstance(R      ,float): R       = numpy.array([R])
        if isinstance(phi_deg,float): phi_deg = numpy.array([phi_deg])
        if isinstance(z      ,float): z       = numpy.array([z])
        ndata = numpy.max(numpy.array([len(R),len(phi_deg),len(z)]))
        if len(R)       == 1: R       = R[0]       + numpy.zeros(ndata)
        if len(z)       == 1: z       = z[0]       + numpy.zeros(ndata)
        if len(phi_deg) == 1: phi_deg = phi_deg[0] + numpy.zeros(ndata)

        # (Rsun,zsun,phisun) --> (xsun,ysun,zsun) [galpy units]
        xyz_sun = bovy_coords.cyl_to_rect(self._Rsun*self._galpy_to_kpc,self._phisun_deg/180.*math.pi,self._zsun*self._galpy_to_kpc)
        Xgc_sun_kpc = xyz_sun[0]
        Ygc_sun_kpc = xyz_sun[1]
        Zgc_sun_kpc = xyz_sun[2]

        # (R,z,phi) --> (x,y,z):
        xyz = bovy_coords.galcencyl_to_XYZ(
                    R*self._galpy_to_kpc, phi_deg/180.*math.pi, z*self._galpy_to_kpc, 
                    Xsun=Xgc_sun_kpc, Zsun=Zgc_sun_kpc
                    )
        Xs_kpc = xyz[:,0]
        Ys_kpc = xyz[:,1]
        Zs_kpc = xyz[:,2]

        # (x,y,z) --> (l,b,d):
        lbd = bovy_coords.XYZ_to_lbd(
                    Xs_kpc, Ys_kpc, Zs_kpc, 
                    degree=False
                    )
        l_rad = lbd[:,0]
        b_rad = lbd[:,1]
        d_kpc = lbd[:,2]

        # given (l,b), find the pixelID:
        phi_star_rad = l_rad
        theta_star_rad = 0.5*numpy.pi - b_rad
        pixelIDs = healpy.ang2pix(self._NSIDE,theta_star_rad, phi_star_rad)

        #return completeness:
        out = numpy.zeros(ndata)
        for ii in range(ndata):
            d_index = (d_kpc[ii] >= self._incomp_dbin_kpc[0:-1]) * (d_kpc[ii] < self._incomp_dbin_kpc[1::])
            #??remove???print d_kpc[ii],self._dmax*self._galpy_to_kpc
            out[ii] = self._incomp_SF_of_hpID_dkpc[pixelIDs[ii],d_index]
        return out


    #-----------------------------------------------------------------------

    def _prepare_and_set_SF_of_R_z(self,nbin_R=400,nbin_z=400,border=None,plotfilename='test_SF_preparation.png',savefilename=None):
        """
        NAME:
            _prepare_and_set_SF_of_R_z
        PURPOSE:
            Defines a grid in (R,z) that covers the extent of the 
            selection function. It then integrates the selection 
            function over phi at each given (R_i,z_i). Based on this 
            int sets up an interpolation object (self._incomp_SF_interpolated_Rkpc_zkpc) 
            for the selection function. It returns and also sets the class fields 
            self._incomp_SF_of_R_z, self._incomp_Rbin_kpc, 
            self._incomp_zbin_kpc, i.e. the (R,z) grid and the 
            completeness at these coordinates.
        INPUT:
            nbin_R -- int scalar -- number of grid points in radial direction covering the extent of the selection function
            nbin_z -- int scalar -- number of grid points in vertical direction covering the extent of the selection function
            border -- float scalar -- little pad to make the extent of the area in which to prepare the completeness in the (R,z) plane  slightly larger [galpy length units]
            plotfilename -- string -- .png filename to plot the integrated selection function into, to control if everything is reasonable.
            savefilename -- string -- .sav filename in which to store self._incomp_Rbin_kpc,self._incomp_zbin_kpc,self._incomp_SF_of_R_z. The contents of this file can later be used to set a selection function object without having to do the phi integration each time.
        OUTPUT:
            incomp_SF_of_R_z -- float array of shape (nbin_R,nbin_z) -- completeness at (R,z)
            incomp_Rbin_kpc, incomp_zbin_kpc -- float arrays -- coordinate axes of the (R,z) 2D grid
        HISTORY:
            2016-09-20 - Started. - Trick (MPIA)
        """
   
        #coordinates at which to evaluate the integral:
        Rs_kpc = numpy.linspace(self._Rmin-border,self._Rmax+border,nbin_R)  * self._galpy_to_kpc
        zs_kpc = numpy.linspace(self._zmin-border,self._zmax+border,nbin_z) * self._galpy_to_kpc
        Rg_kpc, zg_kpc = numpy.meshgrid(Rs_kpc,zs_kpc,indexing='ij')
        Rg_kpc = Rg_kpc.flatten()
        zg_kpc = zg_kpc.flatten()

        #evaluate integral along phi for the whole coordiante array:
        incomp_C = numpy.zeros(len(Rg_kpc))
        for ii in range(len(Rg_kpc)):
            incomp_C[ii] = self._aux_integrate_SF_overphi_in_shell(Rg_kpc[ii],zg_kpc[ii],ngl=80)

        #prepare output and set the incompleteness for this object:
        self._incomp_SF_of_R_z = numpy.reshape(incomp_C,(len(Rs_kpc),len(zs_kpc)))
        self._incomp_Rbin_kpc = Rs_kpc
        self._incomp_zbin_kpc = zs_kpc
        self._incomp_SF_interpolated_Rkpc_zkpc = scipy.interpolate.RectBivariateSpline(
                                        self._incomp_Rbin_kpc,self._incomp_zbin_kpc,
                                        self._incomp_SF_of_R_z,  
                                        kx=3,ky=3,
                                        s=0.
                                        )
        if savefilename is not None:
            save_pickles(savefilename,
              self._incomp_Rbin_kpc,self._incomp_zbin_kpc,self._incomp_SF_of_R_z)

        #plot to test integration:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(self._incomp_SF_of_R_z.T,origin='lower',cmap=cmaps.magma,extent=[min(Rs_kpc),max(Rs_kpc),min(zs_kpc),max(zs_kpc)],aspect='equal',interpolation='nearest')
        ax.set_xlabel('$R$ [kpc]')
        ax.set_ylabel('$z$ [kpc]')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$R \times \Delta \phi$ [kpc]')
        plt.tight_layout()
        plt.savefig(plotfilename,dpi=300)
        print "ATTENTION: Have a look at "+plotfilename+" and check that the SF is okay."

        #output:
        return self._incomp_SF_of_R_z, self._incomp_Rbin_kpc, self._incomp_zbin_kpc
   
    #-----------------------------------------------------------------------


    def _aux_integrate_SF_overphi_in_shell(self,R_kpc,z_kpc,ngl=None):
        """
        NAME:
            _aux_integrate_SF_overphi_in_shell
        PURPOSE:
            calculates the angular extent of the selection function at given (R,z) and integrates it along phi (taking into account the Jacobian factor R)
        INPUT:
            R_kpc, z_kpc -- float scalars -- (R,z) position at which to integrate the selection function completeness over phi [kpc]
            ngl -- int scalar -- order of gauss legendre integration over phi
        OUTPUT:
            total completeness at (R,z) integrated over phi
        HISTORY:
            2016-12-12 - Documented. - Trick (MPIA)
        """

        if not isinstance(R_kpc,float):
            sys.exit("Error in _aux_integrate_SF_overphi_in_shell(): Only for scalar input.")

        rtest_kpc = numpy.sqrt((z_kpc - self._zsun*self._galpy_to_kpc)**2 + (R_kpc - self._Rsun*self._galpy_to_kpc)**2)
        eps = 1e-15
        if rtest_kpc > (self._dmax*self._galpy_to_kpc-eps):
            return 0.
        else:

            #law of cosines: calculate the angle between R and Rsun in a triangle, where the third side is r=sqrt(rm^2+(z-z0)^2):
            phi_dmax_rad = self._deltaphi_rad(R_kpc/self._galpy_to_kpc,z_kpc/self._galpy_to_kpc,self._dmax)

            #function to integrate over:
            func = lambda phi_x_rad: self._aux_incompleteness_function_shell(R_kpc/self._galpy_to_kpc,phi_x_rad/math.pi*180.,z_kpc/self._galpy_to_kpc)  #galpy units

            jacobian = R_kpc

            phisun_rad = self._phisun_deg/180.*math.pi
            
            if rtest_kpc >= self._dmin*self._galpy_to_kpc:
                integral = scipy.integrate.fixed_quad(
                                func, 
                                phisun_rad - phi_dmax_rad, 
                                phisun_rad + phi_dmax_rad, 
                                args=(), n=ngl
                                )
                return integral[0] * jacobian

            elif rtest_kpc < self._dmin*self._galpy_to_kpc:
                phi_dmin_rad = self._deltaphi_rad(R_kpc/self._galpy_to_kpc,z_kpc/self._galpy_to_kpc,self._dmin)   #law of cosines

                integral1 = scipy.integrate.fixed_quad(
                                func, 
                                phisun_rad - phi_dmax_rad,
                                phisun_rad - phi_dmin_rad, 
                                args=(), n=ngl
                                )
                func = lambda phi_x_rad: self._aux_incompleteness_function_shell(R_kpc/self._galpy_to_kpc,phi_x_rad/math.pi*180.,z_kpc/self._galpy_to_kpc)  #galpy units
                integral2 = scipy.integrate.fixed_quad(
                                func, 
                                phisun_rad + phi_dmin_rad, 
                                phisun_rad + phi_dmax_rad, 
                                args=(), n=ngl
                                )
                return (integral1[0]+integral2[0]) * jacobian
            else:
                sys.exit("Error in _aux_integrate_SF_overphi_in_shell(). Check code!")
                

    #-----------------------------------------------------------------------

    def _contains(self,R,z,phi=None):

        """
            NAME:
               _contains
            PURPOSE:
                Tests whether a list of points is inside the shell or not.
            INPUT:
                R - galactocentric radius in galpy units
                z - height above plane in galpy units
                phi - azimuth in galactocentric coordinates in [degrees]
            OUTPUT:
                contains - (float array) - 0 if outside of sphere, 1 if inside of sphere
            HISTORY:
               2017-01-09 - Written. - Trick (MPIA)
        """

        if phi is None: sys.exit("Error in SF_IncompleteShell._contains(): Always specify phi in case of the spherical selection function.")

        # Recursion if input is array:
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
            sys.exit("Error in SF_IncompleteShell._contains(): This function is not implemented for scalar input.")
        if numpy.any(numpy.array([len(z),len(phi)]) != len(R)):
            print numpy.shape(R),numpy.shape(z),numpy.shape(phi)
            sys.exit("Error in SF_IncompleteShell._contains(): Input arrays do not have the same length.")

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

    def _deltaphi_rad(self,R,z,rmax):
        """
            NAME:
               _deltaphi_rad
            PURPOSE:
                largest possible phi at given radius and height at distance rmax from sun
            INPUT:
            OUTPUT:
            HISTORY:
               2016-09-?? - Written. - Trick (MPIA)
        """

        #scalar vs. array input:
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            return numpy.array([self._deltaphi_rad(rr,zz,rmax) for rr,zz in zip(R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            return numpy.array([self._deltaphi_rad(rr,zz,rmax) for rr,zz in zip(R,z)])

        if (z < self._zmin) or (z > self._zmax):
            return 0.

        rc = math.sqrt(rmax**2 - (z-self._zsun)**2)  #radius of circle around sphere at height z, pythagoras

        r1max = self._Rsun + rc
        r1min = self._Rsun - rc
        if (R < r1min) or (R > r1max):
            return 0.

        cosphi = (R**2 - rc**2 + self._Rsun**2) / (2. * self._Rsun * R) #law of cosines
        phimax_rad = numpy.fabs(numpy.arccos(cosphi))  #[rad]
        return phimax_rad


    #-----------------------------------------------------------------------

    def _densfunc(self,R,z,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False):

        if self._densInterp is None:
            sys.exit("Error in SF_IncompleteShell._densfunc(): "+\
                     "self._densInterp is None. Initialize density grid"+\
                     " first before calling this function.")

        if consider_incompleteness and self._with_incompleteness:
            sys.exit("Error in SF_IncompleteShell._densfunc(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #take care of array input:
        if isinstance(z,numpy.ndarray) or isinstance(R,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)

        #outside of observed volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_IncompleteShell._densfunc(): "+\
               "Specify phi in case of the spherical selection function, "+\
               "when using throw_error_outside=True or set_outside_zero=True.") 

            #rotate x axis to go through center of sphere:
            phip_deg = phi - self._phisun_deg   #deg

            xp = self._Rsun - R * numpy.cos(numpy.radians(phip_deg))
            yp = R * numpy.sin(numpy.radians(phip_deg))
            zp = z - self._zsun
            rp2 = (xp**2 + yp**2 + zp**2)

            outside = (rp2 > self._dmax**2) * (rp2 < self._dmin**2)

            if numpy.sum(outside) > 0:
                if throw_error_outside:
                    print "x^2+y^2+z^2 = ",(xp[outside][0])**2 + (yp[outside][0])**2 + (zp[outside][0])**2
                    print "d_max^2     = ",self._dmax**2
                    print "d_min^2     = ",self._dmin**2
                    print "R: ",self._Rmin," <= ",R[outside][0]," <= ",self._Rmax,"?"
                    print "z: ",self._zmin," <= ",z[outside][0]," <= ",self._zmax,"?"
                    dphi = numpy.degrees(0.5 * self._deltaphi_max_rad(R[outside][0],z[outside][0]))
                    print "phi: ",self._phisun_deg-dphi," <= ",phi[outside][0]," <= ",self._phisun_deg+dphi,"?"
                    sys.exit("Error in SF_IncompleteShell._densfunc(). If yes, something is wrong. Testing of code is required.")
                if set_outside_zero:
                    sys.exit("Error in SF_IncompleteShell._densfunc(): If set_outside_zero=True, taking care of array input is not implemented yet.")
                    return 0.

        if not isinstance(R,numpy.ndarray) or len(R) == 1:
            temp = self._densInterp.ev(R,numpy.fabs(z))     #in different versions of scipy ev() returns a result of shape () or (1,)
            temp = numpy.reshape(numpy.array([temp]),(1,)) #this line tries to circumvent the problem
            d = numpy.exp(temp[0])    #the exp of the grid, as the grid is in log
        else:
            temp = self._densInterp.ev(R,numpy.fabs(z))
            d = numpy.exp(temp)    #the exp of the grid, as the grid is in log
            if not len(R) == len(d):
                sys.exit("Error in SF_IncompleteShell._densfunc(): something with the array output is wrong.")
        return d

    #-----------------------------------------------------------------

    def _fastGLint_IncompleteShell(self,func,xgl,wgl):
        """
        NAME:
        PURPOSE:
            fast integration of func(R,z) over R and z in the regime of the shell, including the incompleteness function.
        INPUT:
        OUTPUT:
        HISTORY:
            2016-09-20 - Started. - Trick (MPIA)
        """

        #R oordinates: R_j = 0.5 * (Rmax - Rmin) * (xgl_j + 1) + Rmin:
        Rgl_j = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin

        #account for integration limits in weights w_i and w_j:
        zmaxRgl_j = self._zsun + numpy.sqrt(self._dmax**2 - (self._Rsun - Rgl_j)**2)     #maximum height z_max of sphere at phi=0 at R=Rgl
        zminRgl_j = self._zsun - numpy.sqrt(self._dmax**2 - (self._Rsun - Rgl_j)**2)
        w_j = 0.25 * (self._Rmax - self._Rmin) * (zmaxRgl_j - zminRgl_j) * wgl    #account for integration limits
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
                zglm_ij[ii,jj] = 0.5 * (zmaxRgl_j[jj] - zminRgl_j[jj]) * (xgl[ii] + 1.) + zminRgl_j[jj] #z coordinates: zgl_ij = 0.5 * (zmax(Rgl_j)-zmin(Rgl_j)) * (xgl_i + 1) + zmin(Rgl_j)
                wm_i[ii,jj] = w_i[ii]
                wm_j[ii,jj] = w_j[jj]

        #flatten everything:
        wm_i = wm_i.flatten()
        wm_j = wm_j.flatten()
        Rglm_j = Rglm_j.flatten()
        zglm_ij = zglm_ij.flatten()

        #evaluate function at each grid point:
        func_ij = func(Rglm_j,zglm_ij)

        #angular extend (including incompleteness) at each grid point:
        if self._with_incompleteness:
            #"This function only works with a pre-computed SF_of_R_z, i.e. SF(R,z) = int SF(X) R d phi, stored in self._incomp_SF_of_R_z."
            comp_ij = self._incomp_SF_interpolated_Rkpc_zkpc.ev(Rglm_j*self._galpy_to_kpc,zglm_ij*self._galpy_to_kpc)
        else:
            deltaphi_max_ij_rad = self._deltaphi_rad(Rglm_j,zglm_ij,self._dmax)
            deltaphi_min_ij_rad = self._deltaphi_rad(Rglm_j,zglm_ij,self._dmin)
            comp_ij = 2. * (deltaphi_max_ij_rad-deltaphi_min_ij_rad) * Rglm_j       #angular extent of shell times Jabobian R

        #total:
        tot = numpy.sum(wm_i * wm_j * func_ij * comp_ij)
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
               2016-09-?? - Written. - Trick (MPIA)
        """

        #define function func(R,z) to integrate:
        func = lambda rr,zz: self._densfunc(rr,zz,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False)
        
        #total mass in selection function:
        Mtot = self._fastGLint_IncompleteShell(func,xgl,wgl)
        return Mtot

    #-----------------------------------------------------------------

    def _sftot_fastGL(self,xgl,wgl):

        """
            NAME:
               _sftot_fastGL
            PURPOSE:
                integrates the selection function over the volume using Gauss Legendre quadrature.
                Needed for normalizing outlier model in dftype=12.
            INPUT:
            OUTPUT:
            HISTORY:
               2017-01-02 - Written. - Trick (MPIA)
        """

        #define function func(R,z) to integrate:
        func = lambda rr,zz: 1.
        
        #total mass in selection function:
        sf_tot = self._fastGLint_IncompleteShell(func,xgl,wgl)
        return sf_tot

    #-----------------------------------------------------------------

    def _spatialSampleDF_complete(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,recalc_densgrid=True):
        """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2016-12-12 - Corrected a bug: now the sampling properly takes into account the vertical position of the Sun. - Trick (MPIA)
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
        #print self._densfunc(self._Rmin,zprime,phi=self._phisun_deg)
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

            #reject if smaller than inner edge of shell:
            if rc < self._dmin:
                nreject += 1
                if not quiet: 
                    print nreject," rejected"
            else:

                #transformation to (R,phi,z):
                x = self._Rsun - rc * math.cos(psi) * math.cos(theta)
                y = rc * math.sin(psi) * math.cos(theta)
                z = rc * math.sin(theta) + self._zsun   #move sphere to sit on vertical position of the sun

                R = math.sqrt(x**2 + y**2)
                phi = math.atan2(y,x)   #rad
                phi = math.degrees(phi) #deg
                phi = phi + self._phisun_deg #rotate x axis to go through center of sphere

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

    #-----------------------------------------------------------------

    def _spatialSampleDF_measurementErrors(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,
                                           e_radec_rad=None,e_DM_mag=None,
                                           Xsun_kpc=8.,Ysun_kpc=0.,Zsun_kpc=0.,
                                           spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.):
 
        sys.exit("Error in SF_IncompleteShell._spatialSampleDF_measurementErrors(): This function was not written yet for the IncompleteShell Class.")

        if self._with_incompleteness:
            sys.exit("Error in SF_IncompleteShell._spatialSampleDF_measurementErrors(): "+
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
            sys.exit("Error in SF_IncompleteShell._spatialSampleDF_measurementErrors(): Envelope sphere reaches beyond the Galactic center. Change code! This should not happen!")
        print "Radius of observed volume: ",dmax_kpc," kpc, Radius of envelope volume:", renv_kpc," kpc"

        #_____create new selection function as envelope_____
        #envelope selection function:
        sf_env = SF_IncompleteShell(dmax_env,self._Rcen,df=self._df)
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
                sys.exit("Error in SF_IncompleteShell."+\
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
