from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt
from coord_trafo import galcencyl_to_radecDM, radecDM_to_galcencyl

class SF_IncompleteShell(SelectionFunction):
    """
        Class that implements a spherical shell-like selection function 
        around the sun, with some spatial incompleteness substructure
    """
    def __init__(self,dmin_kpc,dmax_kpc,Rgc_Sun_kpc,zgc_Sun_kpc=0.,phigc_Sun_deg=0.,df=None,incomp_R_kpc=None, incomp_z_kpc=None, incomp_C=None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a shell selection function with incompleteness
        INPUT:
            dmax_kpc: radius of outer edge of shell around center (@ sun) [kpc]
            dmin_kpc: radius of inner edge of shell around center (@ sun) [kpc]
            Rgc_Sun_kpc: distance of center (@ sun) to Galactic center [kpc]
            zcen_kpc: height of center over Galactic plane [kpc]
            phicen_deg: azimuth of center [deg]
            df: galpy object of a distribution function
            incomp_R: 1D array: radial coordinate of incompleteness array
            incomp_z: 1D array: vertical coordinate of incompleteness array
            incomp_C: 2D array: Incompleteness integrated over shell with given dmin and dmax
        OUTPUT:
            shell selection function object
        HISTORY:
            2016-09-20 - Started SF_IncompleteShell.py - Trick (MPIA)
        """
        SelectionFunction.__init__(self,df=df)

        if zcen != 0.:
            sys.exit("Error in SF_IncompleteShell: Nonzero zcen is not implemented yet.")

        #Edges of the spherical shell:
        self._dmin_kpc = dmin_kpc
        self._dmax_kpc = dmax_kpc

        #coordinates of the Sun:
        self._Rgc_Sun_kpc   = Rgc_Sun_kpc
        self._zgc_Sun_kpc   = zgc_Sun_kpc
        self._phigc_Sun_rad = phigc_Sun_deg * math.pi/180.

        #Borders:
        self._Rmin_kpc = Rgc_Sun_kpc - dmax
        self._Rmax_kpc = Rgc_Sun_kpc + dmax
        self._zmin_kpc = zgc_Sun_kpc - dmax
        self._zmax_kpc = zgc_Sun_kpc + dmax
        self._pmax_deg = phigc_Sun_deg + math.degrees(math.asin(dmax_kpc / Rgc_Sun_kpc))
        self._pmin_deg = phigc_Sun_deg - math.degrees(math.asin(dmax_kpc / Rgc_Sun_kpc))

        #incompleteness:
        self._incomp_C = incomp_C
        self._incomp_R_kpc = incomp_R_kpc
        self._incomp_z_kpc = incomp_z_kpc

        if incomp_C is None:
            self._with_incompleteness     = False
            self._incompleteness_function = None
            self._incompleteness_maximum  = None
        else:
            self._with_incompleteness     = True
            self._incompleteness_function = self._aux_incompleteness_function_shell
            self._incompleteness_maximum  = max(incomp_C.flatten())

        return None

    #-----------------------------------------------------------------------

    def _prepare_and_set_incompleteness(self,filename,nbin_R=400,nbin_z=400,border_kpc=0.1,plotfilename='test_SF_preparation.png'):
        """
        NAME:
        PURPOSE:
        INPUT:
            filename: *.sav file that contains int: NSIDE, 
                      1D array: dist_kpc,
                      2D array: SF(healpix_ID,dist_kpc)
        OUTPUT:
        HISTORY:
            2016-09-20 - Started. - Trick (MPIA)
        """
        
        #read selection function, which should be a function of (healpix_ID,dist):
        savefile= open(filename,'rb')
        NSIDE        = pickle.load(savefile)    #int
        dist_kpc     = pickle.load(savefile)    #1D array of lentgh N+1
        completeness = pickle.load(savefile)    #2D array of shape (hp.nside2npix(NSIDE),N)
        savefile.close()
        if numpy.shape(completeness)[0] != hp.nside2npix(NSIDE):
            sys.exit("_prepare_incompleteness(): NSIDE and length of 1st axis of completeness array (which should be the number of healpix) do not agree.")
        if numpy.shape(completeness)[1] != len(dist_kpc)-1:
            sys.exit("_prepare_incompleteness(): length of d_dist and length of 2nd axis of completeness array (which should be the number distances - 1) do not agree.")

        #coordinates at which to evaluate the integral:
        Rs_kpc = numpy.linspace(self._Rmin_kpc-border_kpc,self._Rmax_kpc+border_kpc,nbin_R)
        zs_kpc = numpy.linspace(-self._zmin_kpc-border_kpc,self._zmax_kpc+border_kpc,nbin_z)
        Rg_kpc, zg_kpc = numpy.meshgrid(Rs_kpc,zs_kpc,indexing='ij')
        Rg_kpc = Rg_kpc.flatten()
        zg_kpc = zg_kpc.flatten()

        #evaluate integral along phi for the whole coordiante array:
        incomp_C = numpy.zeros(len(Rg_kpc))
        for ii in range(len(Rg_kpc)):
            incomp_C[ii] = _aux_integrate_SF_overphi_in_shell(Rg_kpc[ii],zg_kpc[ii],completeness=completeness,ngl=80,nside=NSIDE)

        #prepare output and set the incompleteness for this object:
        self._incomp_C = numpy.reshape(incomp_C,(len(Rs_kpc),len(zs_kpc)))
        self._incomp_R_kpc = Rs_kpc
        self._incomp_z_kpc = zs_kpc
        self.set_incompleteness_function(self._aux_incompleteness_function_shell,max(incomp_C.flatten())):

        #plot to test integration:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(incomp_C.T,origin='lower',cmap=cmaps.magma,extent=[min(Rs_kpc),max(Rs_kpc),min(zs_kpc),max(zs_kpc)],aspect='equal',interpolation='nearest')
        ax.set_xlabel('$R$ [kpc]')
        ax.set_ylabel('$z$ [kpc]')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$R \times \Delta \phi$ [kpc]')
        plt.tight_layout()
        plt.savefig(plotfilename,dpi=300)
        print "ATTENTION: Have a look at "+plotfilename+" and check that the SF is okay."

        #output:
        return self._incomp_R_kpc, self._incomp_z_kpc, self._incomp_C



    #-----------------------------------------------------------------------

    def _aux_SF_at_Rphiz(self,R_kpc,phi_rad,z_kpc,completeness=None,dist_kpc=None,nside=None):
        """
        NAME:
        PURPOSE:
            This function converts (R,phi,z) to (l,b,d) and reads out the value of the selection function at this position.
        INPUT:
        OUTPUT:
        HISTORY:
            2016-09-20 - Started. - Trick (MPIA)
        """
        
        if numpy.shape(completeness)[0] != hp.nside2npix(NSIDE):
            sys.exit("_aux_prepare_incomp(): NSIDE and length of 1st axis of completeness array (which should be the number of healpix) do not agree.")
        if numpy.shape(completeness)[1] != len(dist_kpc)-1:
            sys.exit("_aux_prepare_incomp(): length of d_dist and length of 2nd axis of completeness array (which should be the number distances - 1) do not agree.")

        #scalar vs. array input:
        if isinstance(R_kpc  ,float): R_kpc   = numpy.array([R_kpc])
        if isinstance(phi_rad,float): phi_rad = numpy.array([phi_rad])
        if isinstance(z_kpc  ,float):z_kpc    = numpy.array([z_kpc])
        ndata = numpy.max(numpy.array([len(R_kpc),len(phi_rad),len(z_kpc)]))
        if len(R_kpc)   == 1: R_kpc   = R_kpc[0] + numpy.zeros(ndata)
        if len(z_kpc)   == 1: z_kpc   = z_kpc[0] + numpy.zeros(ndata)
        if len(phi_rad) == 1: phi_rad = phi_rad[0] + numpy.zeros(ndata)

        # (Rsun,zsun,phisun) --> (xsun,ysun,zsun)
        xyz_sun = bovy_coords.cyl_to_rect(self._Rg_Sun_kpc,self._phigc_Sun_rad,self._zgc_Sun_kpc)
        Xgc_sun_kpc = xyz_sun[0]
        Ygc_sun_kpc = xyz_sun[1]
        Zgc_sun_kpc = xyz_sun[2]

        # (R,z,phi) --> (x,y,z):
        xyz = bovy_coords.galcencyl_to_XYZ(
                    R_kpc, phi_rad, z_kpc, 
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
        pixelIDs = hp.ang2pix(NSIDE,theta_star_rad, phi_star_rad)

        #return completeness:
        out = numpy.zeros(ndata)
        for ii in range(ndata):
            d_index = (d_kpc > dist_kpc[0:-1]) * (d_kpc < dist_kpc[1::])
            out[ii] = completeness[pixelIDs[ii],d_index]
        return out

    #-----------------------------------------------------------------------


    def _aux_integrate_SF_overphi_in_shell(self,R_kpc,z_kpc,completeness=None,ngl=None,nside=None):

        if not isinstance(R_kpc,float):
            sys.exit("Error in _aux_integrate_SF_overphi_in_shell(): Only for scalar input.")

        rtest_kpc = numpy.sqrt((z_kpc - self._zgc_Sun_kpc)**2 + (R_kpc - self.Rgc_sun_kpc)**2)
        eps = 1e-15
        if rtest_kpc > (dmax_kpc-eps): 
            return 0.
        else:

            #law of cosines: calculate the angle between R and Rsun in a triangle, where the third side is r=sqrt(rm^2+(z-z0)^2):
            phi_dmax_rad = self._deltaphi_rad(R_kpc,z_kpc,self._dmax_kpc)

            #function to integrate over:
            func = lambda phi_x_rad: in_SF(R_kpc,phi_x_rad,z_kpc,completeness=completeness,nside=nside)

            jacobian = R_kpc
            
            if rtest_kpc >= self._dmin_kpc:
                integral = scipy.integrate.fixed_quad(
                                func, 
                                self._phigc_Sun_rad - phi_dmax_rad, 
                                self._phigc_Sun_rad + phi_dmax_rad, 
                                args=(), n=ngl
                                )
                return integral[0] * jacobian

            elif rtest_kpc < self._dmin_kpc:
                phi_dmin_rad = self._deltaphi_rad(R_kpc,z_kpc,self._dmin_kpc)   #law of cosines

                integral1 = scipy.integrate.fixed_quad(
                                func, 
                                self._phigc_Sun_rad - phi_dmax_rad,
                                self._phigc_Sun_rad - phi_dmin_rad, 
                                args=(), n=ngl
                                )
                integral2 = scipy.integrate.fixed_quad(
                                func, 
                                self._phigc_Sun_rad + phi_dmin_rad, 
                                self._phigc_Sun_rad + phi_dmax_rad, 
                                args=(), n=ngl
                                )
                return (integral1[0]+integral2[0]) * jacobian
                

    #-----------------------------------------------------------------------

    def _contains(self,R,z,phi=None):

        sys.exit("ERROR in SF_IncompleteShell._contains(): This function is not implemented yet.")


    #-----------------------------------------------------------------------

    def _deltaphi_rad(self,R_kpc,z_kpc,rmax_kpc):
        """largest possible phi at given radius and height at distance rmax from sun"""

        #scalar vs. array input:
        if isinstance(R_kpc  ,float): R_kpc   = numpy.array([R_kpc])
        if isinstance(z_kpc  ,float): z_kpc   = numpy.array([z_kpc])
        ndata = numpy.max(numpy.array([len(R_kpc),len(z_kpc)]))
        if len(R_kpc)   == 1: R_kpc   = R_kpc[0] + numpy.zeros(ndata)
        if len(z_kpc)   == 1: z_kpc   = z_kpc[0] + numpy.zeros(ndata)

        rc_kpc = math.sqrt(rmax_kpc**2 - (z_kpc-self._zgc_Sun_kpc)**2)  #radius of circle around sphere at height z, pythagoras
        cosphi = (R_kpc**2 - rc_kpc**2 + self.Rgc_sun_kpc**2) / (2. * self.Rgc_sun_kpc * R_kpc) #law of cosines
        phimax_rad = numpy.fabs(numpy.arccos(cosphi))  #rad
        return phimax_rad


    #-----------------------------------------------------------------------

    def _densfunc(self,R,z,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False):

        sys.exit("[TO DO: Rewrite for Shell]")

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

        #outside of obseved volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_IncompleteShell._densfunc(): "+\
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
                    sys.exit("Error in SF_IncompleteShell._densfunc(). If yes, something is wrong. Testing of code is required.")
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
                sys.exit("Error in SF_IncompleteShell._densfunc(): something with the array output is wrong.")
        return d

    #-----------------------------------------------------------------

    def _fastGLint_sphere(self,func,xgl,wgl):

        sys.exit("[TO DO: Rewrite for Shell]")

        """integrate the given function func(R,z,phi) over the spherical effective volume 
           by hand, analogous to Bovy, using Gauss Legendre quadrature."""

        if self._with_incompleteness:
            sys.exit("Error in SF_IncompleteShell._fastGLint_sphere(): "+
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

        sys.exit("[TO DO: Rewrite for Shell]")

        """integrate total mass inside effective volume by hand, analogous to Bovy, using Gauss Legendre quadrature.
           The integration accounts for integration limits - we therefore do not have to set the density outside the sphere to zero."""

        if self._with_incompleteness:
            sys.exit("Error in SF_IncompleteShell._Mtot_fastGL(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #define function func(R,z,phi) to integrate:
        func = lambda rr,zz,pp: self._densfunc(rr,zz,phi=pp,set_outside_zero=False,throw_error_outside=True,consider_incompleteness=False)
        
        #total mass in selection function:
        Mtot = self._fastGLint_sphere(func,xgl,wgl)
        return Mtot

    #-----------------------------------------------------------------

    def _spatialSampleDF_complete(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,recalc_densgrid=True):

        sys.exit("[TO DO: Rewrite for Shell]")

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

    #-----------------------------------------------------------------

    def _spatialSampleDF_measurementErrors(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,
                                           e_radec_rad=None,e_DM_mag=None,
                                           Xsun_kpc=8.,Ysun_kpc=0.,Zsun_kpc=0.,
                                           spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.):

        sys.exit("[TO DO: Rewrite for Shell]")

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
