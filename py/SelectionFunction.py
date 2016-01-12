import numpy
import scipy
import math
import sys
from galpy.util import multi
import multiprocessing
from coord_trafo import galcencyl_to_radecDMvlospmradec, galcencyl_to_radecDM, radecDMvlospmradec_to_galcencyl
from galpy.util import bovy_coords

class SelectionFunction:
    """Top-level class that represents a generic selection function."""
    def __init__(self,df=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initializes an object of a selection function with a yet 
            undefined shape that determines within which volume tracers 
            in a galaxy can be observed. The shape of the selection 
            function is determined by initializing one of the 
            subclasses. By 14-09-19 there exist the selection function 
            subclasses SF_Sphere, SF_Wedge and SF_Cylinder. This 
            initializer is always called from within one of the 
            subclasses. The galaxy belonging to this selection function 
            has to be specified by an action-based tracer distribution 
            function (and the (axisymmetric) potential belonging to this 
            distribution function). 

        INPUT:

            df - (distribution function object) - the distribution 
                function of this SelectionFunction, for example the 
                quasiisothermaldf from the galpy package.

        OUTPUT:

            SelectionFunction object

        HISTORY:

            2014-09-19 - Started - Wilma Trick (MPIA)
        """

        #distribution function (including potential):
        self._df = df

        # grid on which density is calculated and interpolation object, 
        # which interpolates the density on this grid:
        self._densInterp = None
        self._densgrid   = None
        
        # flag to mark if this SelectionFunction object should use 
        # fiducial actions to calculate the density, 
        # and the corresponding fiducial selection function (including 
        # potential), that is used to pre-calculate the fiducial actions:
        self._use_fiducial_actions_Bovy  = False
        self._use_fiducial_actions_Trick = False
        self._df_fid                     = None
        self._n_sigma_fiducial_fit_range = None
        self._vT_galpy_max_fiducial_fit_range = None

        # flag, if this SelectionFunction object has an incompleteness function != 1
        # so far this is not implemented
        self._with_incompleteness     = False
        self._incompleteness_function = None
        self._incompleteness_maximum  = None

        #initialize random number generator for the mock data sampling:
        #x = None
        #numpy.random.seed(seed=x)

        #TO DO: SHIFT ALL THE FIELDS TO HERE; THAT ARE ALSO USED IN THIS FILE????????????
        return None

    #----------------------------------------------------------------

    def contains(self,R,z,phi=None):

        """
        NAME:
            contains
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.contains(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._contains(R,z,phi=phi)

    #----------------------------------------------------------------

    def densityGrid(self,nrs_nonfid=16,nzs_nonfid=16,ngl_vel_nonfid=20,n_sigma=None,vT_galpy_max=None,
                    test_sf=False,test_plot=False,_multi=None,recalculate=False):
        """
        NAME:

            densityGrid

        PURPOSE:

            This function uses the potential and distribution function 
            belonging to this selection function to calculate the 
            spatial tracer density on a grid and then interpolating it. 
            The interpolation (self._densInterp) is then evoked to 
            evaluate the axisymmetric density at radius R and height z 
            by the functions dens(R,z) and _densfunc(R,z).
            This function has two different ways of calculating the 
            density:

            a) calculating the spatial density from scratch using no 
               prior information for the integration over velocities. 
               In this case nrs_nonfid, nzs_nonfid and ngl_vel_nonfid have 
               to be specified. This is the default.

            b) calculating the spatial density using pre-calculated 
               fiducial actions in the given potential for the 
               integration over the velocities. This method is auto-
               matically used, when set_fiducial_df_actions(df) was 
               previously envoked.

            This function does not take care of incompleteness. 
            Incompleteness will be included when calling 
            _densfunc(...,consider_incompleteness=True).

        INPUT:

            nrs_nonfid - (int) - number of grid points in radial 
                        direction between the smallest and largest 
                        radius within the selection function on which 
                        the density will be explicitly calculated 
                        before interpolating. This is only used, when 
                        the density grid is calculated from scratch and 
                        fudicial actions are not used. (Default: 16)

            nzs_nonfid - (int) - number of grid points in vertical 
                        direction between the smallest height (or zero) 
                        and the largest height above the plane within 
                        the selection function on which the density 
                        will be explicitly calculated before inter-
                        polating. This is only used, when the density 
                        grid is calculated from scratch and fudicial 
                        actions are not used. (Default: 16)

            ngl_vel_nonfid - (int) - order of Gauss Legendre integration 
                        used when the density grid is calculated from 
                        scratch and fiducial actions are not used. Then 
                        the distribution function is integrated with 
                        order ngl_vel_nonfid in each of the three velocity 
                        dimensions to calculate the spatial density at 
                        a given point. (Default: 20)

            test_sf - (boolean) - just for test purposes. Sets the 
                          density to 1 everywhere. (Default: False)

            test_plot - (boolean) - just for test purposes. Opens a 
                        figure which shows the interpolated density at 
                        each grid point as a color-coded scatter plot. 
                        (Default: False)

            _multi - (int) - number of cores on which to do the 
                        calculation. Used to speed up the code. In case 
                        this is not None or 1, the density at each grid 
                        point is evaluated on another core. 
                        (Default: None)

            recalculate - (boolean) - flag needed only for the special 
                        case, that the density grid of this selection 
                        function object is already calculated, but we 
                        want to re-calculate it. This flag is for 
                        example needed in case of spatial mock data     
                        sampling, to make sure, that the desired grid 
                        parameters are really applied.
                        (Default: False)

        HISTORY:

            2014-09-19 - Started - Wilma Trick (MPIA)
        """

        if self._df is None:
            sys.exit("Error in SelectionFunction.densityGrid(): "+
                     "The distribution function has to be set, " +
                     "before a density grid can be initialized.")

        # ======================================
        # calculate density on grid from scratch
        # ======================================
        if ((self._densInterp is None) or recalculate) and not self._use_fiducial_actions_Bovy and not self._use_fiducial_actions_Trick:

            #____set up grid on which the density will be calculated:____
            nrs = nrs_nonfid
            nzs = nzs_nonfid
            Rs = numpy.linspace(self._Rmin,self._Rmax,nrs)
            if self._zmax < 0.:
                zup = numpy.fabs(self._zmin)
                zlow = numpy.fabs(self._zmax)
            else:
                zup  = max(numpy.fabs(self._zmax),numpy.fabs(self._zmin))
                zlow = max(0.,self._zmin)
            zs = numpy.linspace(zlow,zup,nzs)

            #_____default integration range over velocity_____
            if n_sigma      is None: n_sigma = 4.
            if vT_galpy_max is None: vT_galpy_max = 1.5


            #____calculate density on grid in (R,z) plane:____
            
            #...test selection function by setting the density grid to 1...

            if test_sf: 
                self._densgrid= numpy.ones((nrs,nzs))

            #...calculate separately on multiple cores...
            elif _multi > 1:    
                
                #    iterate over all R's and z's
                #    ii = index in Rs
                #    jj = index in zs
                r_index = range(nrs)
                z_index = range(nzs)
                ii,jj   = numpy.meshgrid(r_index,z_index,indexing='ij')
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
                ii      = ii.flatten()
                jj      = jj.flatten()

                multOut = multi.parallel_map((
                                lambda x: self._df.density(
                                                Rs[ii[x]],zs[jj[x]],
                                                ngl=ngl_vel_nonfid,
                                                nsigma=n_sigma,
                                                vTmax=vT_galpy_max
                                                )
                                    ),
                                range(nrs*nzs),
                                numcores=numpy.amin([
                                                nrs*nzs,
                                                multiprocessing.cpu_count(),
                                                _multi
                                                ])
                                )
                # Note on multi.parallel_map(): 
                #    1. parameter: function of x to evaluate
                #    2. parameter: iterable sequence through which x will step
                #    3. parameter: number of cores

                self._densgrid= numpy.zeros((nrs,nzs))                
                for x in range(nrs*nzs):
                    self._densgrid[ii[x],jj[x]] = multOut[x]
            
            #...evaluate on only one core...
            else:          
                self._densgrid= numpy.zeros((nrs,nzs))
                for ii in range(nrs):
                    for jj in range(nzs):
                        self._densgrid[ii,jj]= self._df.density(
                                                Rs[ii],zs[jj],
                                                ngl=ngl_vel_nonfid,
                                                nsigma=n_sigma,
                                                vTmax=vT_galpy_max
                                                )

            #____set up interpolation of density on grid____

            # mirror the points around z = 0 (two rows) to get slope 
            # in midplane right when interpolating:
            if zlow == 0.:
                zlow          -= 2. * (zup - zlow) / (nzs - 1.)#new lower z
                zs             = numpy.linspace(zlow,zup,nzs+2)
                dgmirror       = numpy.zeros((nrs,nzs+2))
                dgmirror[:,1]  = self._densgrid[:,1]           #line to mirror
                dgmirror[:,0]  = self._densgrid[:,2]           #line to mirror
                dgmirror[:,2:] = self._densgrid
                self._densgrid = dgmirror                      #updated density

            # save final z and R grid on which the density is interpolated:
            self._densgrid_z = zs    
            self._densgrid_R = Rs

            # set up interpolation object:
            # this intepolate the log of the density ~linear
            self._densInterp= scipy.interpolate.RectBivariateSpline(
                                        self._densgrid_R,self._densgrid_z,
                                        numpy.log(self._densgrid),  
                                        kx=3,ky=3,
                                        s=0.
                                        )

        #=================================================================
        #calculate density on grid using fiducial actions (following Bovy)
        #=================================================================
        elif ((self._densInterp is None) or recalculate) and self._use_fiducial_actions_Bovy and not self._use_fiducial_actions_Trick:


            #____load size of density grid and order of precalculated 
            #    Gauss-Legendre integration actions____
            nrs = len(self._Rs_fid)
            nzs = len(self._zs_fid)
            ngl_fid = int(round(len(self._jr_fid[0,0,:])**(1./3.)))

            #____test, if integration range of pre-calculated actions corresponds to this integration range___
            if n_sigma      is None: n_sigma      = self._n_sigma_fiducial_fit_range
            if vT_galpy_max is None: vT_galpy_max = self._vT_galpy_max_fiducial_fit_range
            if not self._n_sigma_fiducial_fit_range == n_sigma:
                sys.exit("Error in SelectionFunction.densityGrid(): "+
                         "The integration range over the R & z velocitities "+
                         "given by n_sigma = "+str(n_sigma)+" is not "+
                         "the same as that of the pre-calculated "+
                         "fiducial actions, which is "+
                         str(self._n_sigma_fiducial_fit_range)+".")
            if not self._vT_galpy_max_fiducial_fit_range == vT_galpy_max:
                sys.exit("Error in SelectionFunction.densityGrid(): "+
                         "The integration range over the tangential velocitities "+
                         "given by vT_galpy_max = "+str(vT_galpy_max)+" is not "+
                         "the same as that of the pre-calculated "+
                         "fiducial actions, which is "+
                         str(self._vT_galpy_max_fiducial_fit_range)+".")

            #____calculate density on grid in (R,z) plane:____

            # For the integration over velocity use the precomputed 
            # actions (and guiding star radius and frequencies) on a 
            # velocity grid at each (R,z) with a range corresponding to 
            # the fiducial df's sigmas. These sigmas are usually not 
            # too different from this df's sigmas, so we can use the
            # same integration ranges. 
            # It is just important, that this df and the fiducial df 
            # have the same potential.

            self._densgrid= numpy.zeros((nrs,nzs))

            if _multi is None or _multi == 1:
                #....use only one core
                 #    and iterate over all R's and z's
                for ii in range(nrs):
                    for jj in range(nzs):
                        self._densgrid[ii,jj]= self._df.density(
                                self._Rs_fid[ii],self._zs_fid[jj],
                                ngl=ngl_fid,
                                nsigma=n_sigma,
                                vT_galpy_max=vT_galpy_max,
                                _jr     =self._jr_fid     [ii,jj,:],
                                _lz     =self._lz_fid     [ii,jj,:],
                                _jz     =self._jz_fid     [ii,jj,:],
                                _rg     =self._rg_fid     [ii,jj,:],
                                _kappa  =self._kappa_fid  [ii,jj,:],
                                _nu     =self._nu_fid     [ii,jj,:],
                                _Omega  =self._Omega_fid  [ii,jj,:],
                                _sigmaR1=self._sigmaR1_fid[ii], 
                                _sigmaz1=self._sigmaz1_fid[ii]
                                )
                    # Note: the above keywords _jr, _lz, _jz are the 
                    #         pre-computed actions, _rg the guiding star 
                    #        radii, _kappa, _nu, _Omega the frequencies,
                    #        used to integrate the df over velocity 
                    #        space.
                    #        _sigmaR1 and _sigmaz1 are the velocity 
                    #        dispersions at this R and are used to 
                    #        include the correct integration limits in 
                    #        velocity corresponding to the actions.

            elif _multi > 1:
                #....calculate on separate cores
                #    and iterate over all R's and z's: 
                #    ii = index in Rs
                #    jj = index in zs
                r_index = range(nrs)
                z_index = range(nzs)
                ii,jj   = numpy.meshgrid(r_index,z_index,indexing='ij') 
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
                ii      = ii.flatten()
                jj      = jj.flatten()
                multOut = multi.parallel_map((
                            lambda x: self._df.density(
                                     self._Rs_fid[ii[x]],
                                     self._zs_fid[jj[x]],
                                     ngl=ngl_fid,
                                     nsigma=n_sigma,
                                     vT_galpy_max=vT_galpy_max,
                                     _jr     =self._jr_fid     [ii[x],jj[x],:],
                                     _lz     =self._lz_fid     [ii[x],jj[x],:],
                                     _jz     =self._jz_fid     [ii[x],jj[x],:],
                                     _rg     =self._rg_fid     [ii[x],jj[x],:],
                                     _kappa  =self._kappa_fid  [ii[x],jj[x],:],
                                     _nu     =self._nu_fid     [ii[x],jj[x],:],
                                     _Omega  =self._Omega_fid  [ii[x],jj[x],:],
                                     _sigmaR1=self._sigmaR1_fid[ii[x]],
                                     _sigmaz1=self._sigmaz1_fid[ii[x]]
                                     )
                                ),
                            range(nrs*nzs),
                            numcores=numpy.amin([
                                        nrs*nzs,
                                        multiprocessing.cpu_count(),
                                        _multi
                                        ])
                            )
                for x in range(nrs*nzs):
                    self._densgrid[ii[x],jj[x]] = multOut[x]

            #____set up interpolation of density on grid____

            # mirror the points around z = 0 (two rows) to get slope 
            # in midplane right when interpolating:
            zlow = min(self._zs_fid)
            zup  = max(self._zs_fid)
            if zlow == 0.:
                zlow          -= 2. * (zup - zlow) / (nzs - 1.) #new lower z
                zs_finalgrid   = numpy.linspace(zlow,zup,nzs+2)
                dgmirror       = numpy.zeros((nrs,nzs+2))
                dgmirror[:,1]  = self._densgrid[:,1] #line to mirror
                dgmirror[:,0]  = self._densgrid[:,2] #line to mirror
                dgmirror[:,2:] = self._densgrid
                self._densgrid = dgmirror        #updated density
            else:
                zs_finalgrid = self._zs_fid

            #TO DO: check what's happening with self._zs_fid, zs_finalgrid and self._densgrid_z????????????????

            #save final z and R grid:
            self._densgrid_z = zs_finalgrid
            self._densgrid_R = self._Rs_fid

            # set up interpolation object:
            # this intepolate the log of the density ~linear
            self._densInterp= scipy.interpolate.RectBivariateSpline(
                                        self._densgrid_R,self._densgrid_z,
                                        numpy.log(self._densgrid),  
                                        kx=3,ky=3,
                                        s=0.
                                        )

        #=======================================================================
        #calculate density fiducial actions on a velocity integration grid (by Trick)
        #=======================================================================
        elif ((self._densInterp is None) or recalculate) and self._use_fiducial_actions_Trick and not self._use_fiducial_actions_Bovy:
            
            # shape of the 5D velocity integration grid:
            shapetuple = numpy.shape(self._R_vig)

            # *Note:* In case of the _use_fiducial_actions_Trick the 
            #         actions are a 5D grid (_jr_vig), in case of 
            #         _use_fiducial_actions_Bovy it is 3D (_jr_fid)

            #____test, if integration range of pre-calculated actions corresponds to this integration range___
            if n_sigma      is None: n_sigma      = self._n_sigma_fiducial_fit_range
            if vT_galpy_max is None: vT_galpy_max = self._vT_galpy_max_fiducial_fit_range
            if not self._n_sigma_fiducial_fit_range == n_sigma:
                print "Error in SelectionFunction.densityGrid(): "+\
                         "The integration range over the R & z velocitities "+\
                         "given by n_sigma = "+str(n_sigma)+" is not "+\
                         "the same as that of the pre-calculated "+\
                         "fiducial actions, which is "+\
                         str(self._n_sigma_fiducial_fit_range)+"."+\
                         "But as n_sigma is not used in this special "+\
                         "case the calculation is not stopped."
            if not self._vT_galpy_max_fiducial_fit_range == vT_galpy_max:
                print "Error in SelectionFunction.densityGrid(): "+\
                         "The integration range over the tangential velocitities "+\
                         "given by vT_galpy_max = "+str(vT_galpy_max)+" is not "+\
                         "the same as that of the pre-calculated "+\
                         "fiducial actions, which is "+\
                         str(self._vT_galpy_max_fiducial_fit_range)+"."+\
                         "But as vT_galpy_max is not used in this special "+\
                         "case the calculation is not stopped."

            #____evaluate df at each grid point____

            if _multi is None or _multi == 1:
                #...evaluation on one core:

                df_vig = self._df(
                        (
                            numpy.ravel(self._jr_vig),
                            numpy.ravel(self._lz_vig),
                            numpy.ravel(self._jz_vig)
                            ),
                        rg   =numpy.ravel(self._rg_vig),
                        kappa=numpy.ravel(self._kappa_vig),
                        nu   =numpy.ravel(self._nu_vig),
                        Omega=numpy.ravel(self._Omega_vig)
                        )    
 
            elif _multi > 1:
                #...evaluation on multiple cores:

                #number of data points:
                ndata = self._R_vig.size

                # number of cores to use:
                N = numpy.amin([
                            ndata,
                            multiprocessing.cpu_count(),
                            _multi
                            ])

                # data points to evaluate on one core:
                M = int(math.floor(ndata / N))

                # first evaluate arrays on each core to make use of 
                # the fast evaluation of input arrays:
                multiOut =  multi.parallel_map(
                    (lambda x: self._df(
                        (
                            numpy.ravel(self._jr_vig)[x*M:(x+1)*M],
                            numpy.ravel(self._lz_vig)[x*M:(x+1)*M],
                            numpy.ravel(self._jz_vig)[x*M:(x+1)*M]
                            ),
                        rg   =numpy.ravel(self._rg_vig)   [x*M:(x+1)*M],
                        kappa=numpy.ravel(self._kappa_vig)[x*M:(x+1)*M],
                        nu   =numpy.ravel(self._nu_vig)   [x*M:(x+1)*M],
                        Omega=numpy.ravel(self._Omega_vig)[x*M:(x+1)*M]
                        )
                    ),
                    range(N),
                    numcores=N
                    )

                #df values on vig (velocity integration grid):
                df_vig = numpy.zeros(ndata)
                for x in range(N):
                    df_vig[x*M:(x+1)*M] = multiOut[x]

                # number of data points not yet evaluated:
                K = ndata % N

                # now calculate the rest of the data:
                if K > 0:
                        
                    multiOut =  multi.parallel_map(
                        (lambda x: self._df(
                            (
                                numpy.ravel(self._jr_vig)[N*M+x],
                                numpy.ravel(self._lz_vig)[N*M+x],
                                numpy.ravel(self._jz_vig)[N*M+x]
                                ),
                            rg   =numpy.ravel(self._rg_vig)   [N*M+x],
                            kappa=numpy.ravel(self._kappa_vig)[N*M+x],
                            nu   =numpy.ravel(self._nu_vig)   [N*M+x],
                            Omega=numpy.ravel(self._Omega_vig)[N*M+x]
                            )
                        ),
                        range(K),
                        numcores=numpy.amin([
                            K,
                            multiprocessing.cpu_count(),
                            _multi
                            ])
                        )
                    for x in range(K):
                        df_vig[N*M+x] = multiOut[x]

            #____calculate density by applying GL integration:____

            #reshape to shape of velocity integration grid:
            df_vig = numpy.reshape(df_vig,shapetuple)

            #multiply with gauss legendre weights and sum over velocity axes in array:
            self._densgrid = numpy.sum(self._weights_vig * df_vig, axis=(2,3,4))

            #____set up interpolation of density on grid____

            # mirror the points around z = 0 (two rows) to get slope 
            # in midplane right when interpolating:
            zlow = min(self._zs_fid)
            zup  = max(self._zs_fid)
            if zlow == 0.:
                nrs = shapetuple[0]
                nzs = shapetuple[1]
                zlow          -= 2. * (zup - zlow) / (nzs - 1.) #new lower z
                zs_finalgrid   = numpy.linspace(zlow,zup,nzs+2)
                dgmirror       = numpy.zeros((nrs,nzs+2))
                dgmirror[:,1]  = self._densgrid[:,1] #line to mirror
                dgmirror[:,0]  = self._densgrid[:,2] #line to mirror
                dgmirror[:,2:] = self._densgrid
                self._densgrid = dgmirror        #updated density
            else:
                zs_finalgrid = self._zs_fid

            #save final z and R grid:
            self._densgrid_z = zs_finalgrid
            self._densgrid_R = self._Rs_fid

            # set up interpolation object:
            # this intepolate the log of the density ~linear
            self._densInterp= scipy.interpolate.RectBivariateSpline(
                                        self._densgrid_R,self._densgrid_z,
                                        numpy.log(self._densgrid),  
                                        kx=3,ky=3,
                                        s=0.
                                        )
 
        #======================================================================
        #plot the interpolated density at each grid point (just for test purposes)
        #=======================================================================
        elif test_plot:

            #scaling parameters to rescale galpy units to physical units:
            _REFR0 = 8.
            ro = 1.
            
            # initialize grid:
            # 2D array: 1st index is y-axis, 2nd index is x-axis
            Rgrid, zgrid = numpy.meshgrid(self._densgrid_R,self._densgrid_z,indexing='xy') 
            # (*Note:* careful with meshgrid: when using the default indexing 'xy' 
            #          arrays have to be treated as v[j,i] instead of v[i,j] later.
            #          Here it is done correctly.)
            density = numpy.empty((len(self._densgrid_z),len(self._densgrid_R)))    
                
            
            #Call interpolation object:
            for ii in range(len(self._densgrid_R)):
                for jj in range(len(self._densgrid_z)):
                    density[jj,ii]= self._densfunc(
                                        self._densgrid_R[ii],
                                        self._densgrid_z[jj],
                                        phi=0.,
                                        set_outside_zero=False)
            residual = density -self._densgrid.T

            #plot:
            import matplotlib.pyplot as plt
            plt.cla()
            plt.subplot(2,1,1)
            plt.scatter(Rgrid*_REFR0*ro,zgrid*_REFR0*ro,c=density,s=30)
            plt.xlabel("R [kpc]")
            plt.ylabel("z [kpc]")
            cbar = plt.colorbar()
            cbar.set_label("density [galpy units] (interpolated)")
            plt.subplot(2,1,2)
            plt.scatter(Rgrid*_REFR0*ro,zgrid*_REFR0*ro,c=residual,s=30)
            plt.xlabel("R [kpc]")
            plt.ylabel("z [kpc]")
            cbar = plt.colorbar()
            cbar.set_label("interpolation residuals of density [galpy units]")
            plt.show()

    #----------------------------------------------------------------

    def dens(self,R,z,phi=None):
        """
        NAME:
            densfunc
        PURPOSE:

            DO I NEED THIS ????????????????????????

        INPUT:
        OUTPUT:
        HISTORY:
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.dens(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._densfunc(R,z,phi=phi,set_outside_zero=False,throw_error_outside=True)

    #----------------------------------------------------------------

    def reset_df(self,df):
        """
        NAME:

            reset_df

        PURPOSE:
            
            This function is used to set and reset the distribution 
            function of this SelectionFunction object. If the 
            distribution function is reset, the density grid and 
            interpolation object are reset to None and have to be 
            calculated anew. In case this SelectionFunction uses 
            fiducial actions to calculate the density, the new 
            distribution function has to have the same potential as the 
            fiducial distribution function.

        INPUT:

            df - (distribution function object) - the new distribution 
                function of this SelectionFunction, for example the 
                quasiisothermaldf from the galpy package.

        HISTORY:

            2014-09-22 - Started - Wilma Trick (MPIA)
        """
        # reset distribution function and density grid:
        self._df         = df
        self._densInterp = None
        self._densgrid   = None
    
        # test, if the new df and the fiducial df have same potential:
        if self._df_fid is not None:
            if not self._df._pot == self._df_fid._pot:
                sys.exit("Error in SelectionFunction.reset_df(df): "+
                         "The new distribution function df does not "+
                         "have the same potential as the fiducial "+
                         "distribution function of this "+
                         "SelectionFunction object.")
        return None

    #----------------------------------------------------------------

    def set_fiducial_df(self,df_fid):

        # set flag and fiducial df and reset df and density:
        self._densInterp           = None
        self._densgrid             = None
        self._df_fid               = df_fid
        self._df                   = None
        self._use_fiducial_actions_Bovy = False  #They have to be calculated first.
        self._use_fiducial_actions_Trick = False
        self._n_sigma_fiducial_fit_range = None
        self._vT_galpy_max_fiducial_fit_range = None

    #----------------------------------------------------------------

    def set_fiducial_df_actions_Bovy(self,df_fid,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,_multi=None):
        """
        NAME:

            set_fiducial_df_Bovy

        PURPOSE:

            To speed up the calculation of the density on a grid in 
            densityGrid() this SelectionFunction object can use fiducial 
            actions. If this function is called, then a 3D velocity grid 
            (spaced according to Gauss Legendre integration) is set up 
            at each grid point of the (R,z) grid and the actions on this 
            grid are then pre-calculated and stored to be used by 
            densityGrid(). The fiducial distribution function is used 
            for only two things:
            a) the potential stored in the fiducial df is used for the 
               action calculation.
            b) the velocity dispersion of the fiducial df is used to 
               determine the integration ranges over velocity.
            
        INPUT:

            df_fid - (distribution function object) - the fiducial 
                        distribution function of this SelectionFunction, for 
                        example a quasiisothermaldf from the galpy package.

            nrs - (int) - number of grid points in radial 
                        direction between the smallest and largest 
                        radius within the selection function on which 
                        the density will be explicitly calculated 
                        before interpolating. (Default: 16)

            nzs - (int) - number of grid points in vertical 
                        direction between the smallest height (or zero) 
                        and the largest height above the plane within 
                        the selection function on which the density 
                        will be explicitly calculated before inter-
                        polating. (Default: 16)

            ngl_vel - (int) - order of Gauss Legendre integration in each of 
                        the three velocity dimensions to pre-calculate 
                        the actions used to later calculate the spatial 
                        density at a given (R,z) point. (Default: 20)

            _multi - (int) - number of cores on which to do the 
                        calculation. Used to speed up the code. In case 
                        this is not None or 1, the density at each grid 
                        point is evaluated on another core. 
                        (Default: None)

            n_sigma - (float) - ???

            vT_galpy_max - (float) - ???
        HISTORY:

            2014-09-22 - Started - Wilma Trick (MPIA)
        """

        # set flag and fiducial df and reset df and density:
        self.set_fiducial_df(df_fid)
        self._use_fiducial_actions_Bovy = True
        self._use_fiducial_actions_Trick = False
        self._n_sigma_fiducial_fit_range = n_sigma
        self._vT_galpy_max_fiducial_fit_range = vT_galpy_max

        # grid on which the density will be calculated using fiducial actions:
        Rs = numpy.linspace(self._Rmin,self._Rmax,nrs)
        if self._zmax < 0.:
            zup  = numpy.fabs(self._zmin)
            zlow = numpy.fabs(self._zmax)
        else:
            zup  = max(numpy.fabs(self._zmax),numpy.fabs(self._zmin))
            zlow = max(0.,self._zmin)
        zs = numpy.linspace(zlow,zup,nzs)
        self._zs_fid = zs
        self._Rs_fid = Rs

        #NOTE TO MYSELF: This could be made better! In case of a selection function that is not a wedge, here we calculate actions at R's and z's that are never actually used. When modifying this, I also could save time.
       
    
        # for each point (R,z) we want to integrate over all three velocities, 
        # using the Gauss-Legendre integration we need ngl_vel**3 actions for each 
        # point (R,z), i.e. each ngl_vel actions per velocity direction.

        # initialize arrays for pre-computed actions:
        self._jr_fid    = numpy.zeros((nrs,nzs,ngl_vel**3)) #Actions
        self._lz_fid    = numpy.zeros((nrs,nzs,ngl_vel**3))
        self._jz_fid    = numpy.zeros((nrs,nzs,ngl_vel**3))
        self._rg_fid    = numpy.zeros((nrs,nzs,ngl_vel**3)) #Guiding star radius
        self._kappa_fid = numpy.zeros((nrs,nzs,ngl_vel**3)) #Frequencies
        self._nu_fid    = numpy.zeros((nrs,nzs,ngl_vel**3))
        self._Omega_fid = numpy.zeros((nrs,nzs,ngl_vel**3))
        self._sigmaR1_fid   = numpy.zeros(nrs) #radial velocity dispersion at R
        self._sigmaz1_fid   = numpy.zeros(nrs) #vertical velocity dispersion at R
        
        #...evaluation on 1 core...
        if _multi is None or _multi == 1:

            # iterate over all R's and z's:
            for ii in range(nrs):
                for jj in range(nzs):
                    out = self._aux_actions_for_vel_int_Bovy(
                                    self._Rs_fid[ii],
                                    self._zs_fid[jj],
                                    ngl_vel=ngl_vel,
                                    n_sigma=n_sigma,
                                    vT_galpy_max=vT_galpy_max
                                    )
                    self._jr_fid   [ii,jj,:] = out[0,:] #jr
                    self._lz_fid   [ii,jj,:] = out[1,:] #lz
                    self._jz_fid   [ii,jj,:] = out[2,:] #jz
                    self._rg_fid   [ii,jj,:] = out[3,:] #rg
                    self._kappa_fid[ii,jj,:] = out[4,:] #kappa
                    self._nu_fid   [ii,jj,:] = out[5,:] #nu
                    self._Omega_fid[ii,jj,:] = out[6,:] #Omega


        # ... evaluate on multiple cores...
        elif _multi > 1:
            # iterate over all R's and z's:
            #    ii = index in Rs
            #    jj = index in zs
            r_index  = range(nrs)
            z_index = range(nzs)
            ii,jj = numpy.meshgrid(r_index,z_index,indexing='ij') 
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
            ii = ii.flatten()
            jj = jj.flatten()
            multiOut = multi.parallel_map((
                                lambda x: self._aux_actions_for_vel_int_Bovy(
                                                    self._Rs_fid[ii[x]],
                                                    self._zs_fid[jj[x]],
                                                    ngl_vel=ngl_vel,
                                                    n_sigma=n_sigma,
                                                    vT_galpy_max=vT_galpy_max
                                                    )
                                    ),
                                range(nrs*nzs),
                                numcores=numpy.amin([
                                            nrs*nzs,
                                            multiprocessing.cpu_count(),
                                            _multi
                                            ])
                                )
            for x in range(nrs*nzs):
                self._jr_fid   [ii[x],jj[x],:] = multiOut[x][0,:] #jr
                self._lz_fid   [ii[x],jj[x],:] = multiOut[x][1,:] #lz
                self._jz_fid   [ii[x],jj[x],:] = multiOut[x][2,:] #jz
                self._rg_fid   [ii[x],jj[x],:] = multiOut[x][3,:] #rg
                self._kappa_fid[ii[x],jj[x],:] = multiOut[x][4,:] #kappa
                self._nu_fid   [ii[x],jj[x],:] = multiOut[x][5,:] #nu
                self._Omega_fid[ii[x],jj[x],:] = multiOut[x][6,:] #Omega

        for ii in range(nrs):
            #radial velocity dispersion at R (cf. eq. (4) in Bovy & Rix 2013):
            self._sigmaR1_fid[ii] = self._df_fid._sr * numpy.exp(
                                                (self._df_fid._ro - self._Rs_fid[ii])/self._df_fid._hsr
                                                )
            #vertical velocity dispersion at R (cf. eq. (5) in Bovy & Rix 2013):
            self._sigmaz1_fid[ii] = self._df_fid._sz * numpy.exp(
                                                (self._df_fid._ro - self._Rs_fid[ii])/self._df_fid._hsz
                                                )

        return None

    #----------------------------------------------------------------

    def _aux_actions_for_vel_int_Bovy(self,R,z,ngl_vel=None,n_sigma=None,vT_galpy_max=None):
        """
        NAME:

            _aux_actions_for_vel_int_Bovy

        PURPOSE:

            This is an auxiliary function used in 
            set_fiducial_df_actions(). It calls a member function of the 
            distribution, that calculates the density at (R,z) by Gauss 
            Legendre integration over the velocities. For this 
            integration it calculates the actions (and corresponding 
            frequencies and guiding star radii) at the grid points of 
            the velocity grid for the given (R,z) in the potential 
            belonging to the df. This function only returns those actions 
            at the grid points for the fiducial df.

        INPUT:

            R - (float) - radius [in galpy units] at which to calculate 
                    the actions on a velocity grid.

            z - (float) - height above the plane [in galpy units] at 
                    which to calculate the actions on a velocity grid.

            df - (distribution function object) - the fiducial 
                    distribution function of which the potential is used 
                    to calculate the actions on a grid.

            ngl_vel - (int) - order of Gauss Legendre integration over the 
                    velocities. The actions which are returned, are 
                    calculated at the given (R,z) and at ngl^3 
                    velocities, which are distributed according to the 
                    points of a Gauss Legendre integration.

            n_sigma - (float) - ???

            vT_galpy_max - (float) - ???
        OUTPUT:

            out - (float-array) - array of shape (7,ngl^3) which contains 
                    for each of the three velocity directions the actions 
                    jr, lz, jz, the guiding star radius rg, and the frequencies 
                    kappa, nu, Omega.

        HISTORY:

            2014-09-22 - Started - Wilma Trick (MPIA)
        """

        if ngl_vel is None:
            sys.exit("Error in SelectionFunction._aux_actions_for_vel_int():"+
                     "ngl has to be set.")

        #...evaluate actions for Gauss-Legendre integration over velocities
        temp,jr,lz,jz,rg,kappa,nu,Omega = self._df_fid.vmomentdensity(
                                                R,z,0.,0.,0.,
                                                gl=True,ngl=ngl_vel,nsigma=n_sigma,vTmax=vT_galpy_max,
                                                _return_actions=True,
                                                _return_freqs=True)

        out = numpy.zeros((7,ngl_vel**3))
        out[0,:] = jr
        out[1,:] = lz
        out[2,:] = jz
        out[3,:] = rg
        out[4,:] = kappa
        out[5,:] = nu
        out[6,:] = Omega

        #if numpy.sum(numpy.isfinite(jr)) < jr.size:
        #    print "jr",
        #if numpy.sum(numpy.isfinite(lz)) < lz.size:
        #    print "lz",
        #if numpy.sum(numpy.isfinite(jz)) < jz.size:
        #    print "Jz",
        #if numpy.sum(numpy.isfinite(rg)) < rg.size:
        #    print "rg",
        #if numpy.sum(numpy.isfinite(kappa)) < kappa.size:
        #    print "kappa",
        #if numpy.sum(numpy.isfinite(nu)) < nu.size:
        #    print "nu",
        #if numpy.sum(numpy.isfinite(Omega)) < Omega.size:
        #    print "Omega",

        return out


    #----------------------------------------------------------------

    def spatialSampleDF(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,
                        quiet=False,test_sf=False,_multi=None,
                        recalc_densgrid=True):
        """
        NAME:
        
            spatialSampleDF

        PURPOSE:

            This is the wrapper function for the spatial mock data 
            sampling function of the subclasses. It samples (R,z,phi) 
            within an observed volume of which the shape is defined by 
            using the corresponding SelectionFunction subclass. The 
            spatial sampling works in two steps:
            1. The observed volume is first uniformly sampled in space 
               (by Inverse transform Monte Carlo sampling).
            2. Using Rejection Sampling the uniformly distributed points 
               are sampled such that they follow the spatial density as 
               given by the df.
            These steps (especially the first one) depend on the 
            explicit shape of the SelectionFunction and are therefore 
            implementen within the subclass that defines the shape.

        INPUT:

            nmock - (int) - number of mock data positions to sample and return (Default: 500)

            nrs- (int) - number of grid points in radial 
                        direction between the smallest and largest 
                        radius within the selection function on which 
                        the density will be explicitly calculated 
                        before interpolating and then sampling mock data 
                        from this density. (Default: 16)

            nzs - (int) - number of grid points in vertical 
                        direction between the smallest height (or zero) 
                        and the largest height above the plane within 
                        the selection function on which the density 
                        will be calculated. (Default: 16)

            ngl_vel - (int) - order of Gauss Legendre integration 
                        used for calculating the density. The 
                        distribution function is integrated with 
                        order ngl in each of the three velocity 
                        dimensions to calculate the spatial density at 
                        a given point. (Default: 20)

            quiet - (boolean) - flag to print (=False) or suppress (=True) 
                        status reports in the console. (Default: False)

            test_sf - (boolean) - just for test purposes. Sets the 
                        density from which spatial mock data is sampled 
                        to 1 everywhere. (Default: False)

            _multi - (int) - number of cores on which to do the 
                        calculation of spatial density. 
                        Used to speed up the code. In case 
                        this is not None or 1, the density at each grid 
                        point is evaluated on another core. 
                        (Default: None)

            recalc_densgrid - (boolean) - flag for determining, if the 
                        density grid should be re-calculated of if a 
                        previously calculated grid should be used. By 
                        setting this flag to False, the grid parameter 
                        keywords (nrs, nzs, ngl, test_sf) are not used. 
                        (ngl is however still used to determine maximum 
                        density in the observed volume.)
                        (Default: True)

            n_sigma - (float) - ???

            vT_galpy_max - (float) - ???
        

        OUTPUT:

            R - (float array) - radial positions of the mock data 
                        [in galpy units].
            z - (float array) - vertical positions of the mock data
                        [in galpy units].
            phi - (float array) - angular positions of the mock data
                        [in galpy units].

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """
        if self._with_incompleteness:
            if not quiet: print "     --> INCOMPLETE DATA SET"
            return self._spatialSampleDF_incomplete(
                    nmock=nmock,
                    nrs=nrs,nzs=nzs,
                    ngl_vel=ngl_vel,
                    n_sigma=n_sigma,
                    vT_galpy_max=vT_galpy_max,
                    quiet=quiet,test_sf=test_sf,
                    _multi=_multi,
                    recalc_densgrid=recalc_densgrid
                    )
        else:
            if not quiet: print "     --> COMPLETE DATA SET"
            return self._spatialSampleDF_complete(
                    nmock=nmock,
                    nrs=nrs,nzs=nzs,
                    ngl_vel=ngl_vel,
                    n_sigma=n_sigma,
                    vT_galpy_max=vT_galpy_max,
                    quiet=quiet,test_sf=test_sf,
                    _multi=_multi,
                    recalc_densgrid=recalc_densgrid
                    )

    #----------------------------------------------------------------


    def dMdR(self,R,ngl=20):
        """
        NAME:

            dMdR

        PURPOSE:

            The total "mass" of tracers within the selection function 
            at a given radius.
            I.e. the following integration is numerically performed:

                dM/dR (R) = \int dz dphi d\vec{v}
                                df(\vec{x},\vec{v}) * sf(\vec{x}) * R
            
            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT). The factor R 
            is the Jacobian. df the distribution function and sf the 
            selection function.
            This integral is not normalized. To normalize it the result 
            has to be divided by Mtot (see below).
            This is a wrapper function, as the explicit calculation 
            depends on the shape of the Selection Function and is 
            therefore performed in the corresponding subclass.

        INPUT:

            R - (float) - the radius [in galpy units] at which the total 
                        "mass" of tracers is calculated.

            ngl - (int) - order of Gauss Legendre integration of the 
                        spatial density over z. (Because of axisymmetry 
                        the integration over phi is trivial.) 
                        (Default: 20)
        OUTPUT:

            dMdR - (float) - total "mass" of tracers with R: dM/dR (R).

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.dMdR(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._dMdR(R,ngl=ngl)

    #----------------------------------------------------------------

    def dMdz(self,z,ngl=20):
        """
        NAME:

            dMdz

        PURPOSE:

            The total "mass" of tracers within the selection function 
            at a given vertical height above the plane.
            I.e. the following integration is numerically performed:

                dM/dz (z) = \int dR dphi d\vec{v}
                                df(\vec{x},\vec{v}) * sf(\vec{x}) * R
            
            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT). The factor R 
            is the Jacobian. df the distribution function and sf the 
            selection function.
            This integral is not normalized. To normalize it the result 
            has to be divided by Mtot (see below).
            This is a wrapper function, as the explicit calculation 
            depends on the shape of the Selection Function and is 
            therefore performed in the corresponding subclass.

        INPUT:

            z - (float) - the vertical height above the plane [in galpy 
                        units] at which the total "mass" of tracers is 
                        calculated.

            ngl - (int) - order of Gauss Legendre integration of the 
                        spatial density over R. (Because of axisymmetry 
                        the integration over phi is trivial.) 
                        (Default: 20)
        OUTPUT:

            dMdz - (float) - total "mass" of tracers with z: dM/dz (z).

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.dMdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._dMdz(z,ngl=ngl)

    #----------------------------------------------------------------

    def dMdphi(self,phi,ngl=20):
        """
        NAME:

            dMdphi

        PURPOSE:

            The total "mass" of tracers within the selection function 
            at a given angular position.
            I.e. the following integration is numerically performed:

                dM/dphi (phi) = \int dR dz d\vec{v}
                                   df(\vec{x},\vec{v}) * sf(\vec{x}) * R                
            
            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT). The factor R 
            is the Jacobian. df the distribution function and sf the 
            selection function.
            This integral is not normalized. To normalize it the result 
            has to be divided by Mtot (see below).
            This is a wrapper function, as the explicit calculation 
            depends on the shape of the Selection Function and is 
            therefore performed in the corresponding subclass.

        INPUT:

            phi - (float) - the angular position [in degrees] at which 
                        the total "mass" of tracers is calculated.

            ngl - (int) - order of Gauss Legendre integration of the 
                        spatial density over each R and z.
                        (Default: 20)
        OUTPUT:

            dMdz - (float) - total "mass" of tracers with phi: 
                        dM/dphi (phi).

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.dMdphi(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._dMdphi(phi,ngl=ngl)

   #----------------------------------------------------------------

    def Mencl(self,R,ngl=20):
        """
        NAME:

            Mencl

        PURPOSE:

            The total "mass" of tracers within the selection function 
            and enclosed within a given maximum radius.
            This is simply done by integrating dM/dR(R) from Rmin to 
            given R, i.e. the following integration is performed 
            numerically:

                M (<R) = \int_Rmin^R dR dM/dR(R)

                       = \int_Rmin^R dR \int dphi dz d\vec{v}
                             df(\vec{x},\vec{v}) * sf(\vec{x}) * R

            This integral is not normalized. To normalize it the result 
            has to be divided by Mtot (see below).
            

        INPUT:

            R - (float) - maximum radius [in galpy units] up to which 
                the total tracer density is integrated within the selection function.

            ngl - (int) - order of Gauss Legendre integration of 
                dM/dR(R) over R. (Default: 20)

        OUTPUT:

            Mencl - (float) - total "mass" of tracers enclosed within 
                given R within the selection function.

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.Mencl(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        # Recursion if input is array:
        if isinstance(R,numpy.ndarray):
            return numpy.array([self._Mencl(rr) for rr in R])

        # use scipy's Gauss Legendre integrator 
        # to integrate dMdR between Rmin and R:
        Mencl = (scipy.integrate.fixed_quad(
                        lambda rr: self._dMdR(rr,ngl),
                        self._Rmin,R,
                        n=ngl
                        )
                )[0]
        

        return Mencl

   #----------------------------------------------------------------

    def Mtot(self,ngl=20,xgl=None,wgl=None):
        """
        NAME:

            Mtot

        PURPOSE:

            The total "mass" of tracers within the selection function 
            according to the distribution function.
            I.e. the following integration is performed 
            numerically:

                M_tot = \int dR dphi dz d\vec{v}
                             df(\vec{x},\vec{v}) * sf(\vec{x}) * R

                      =  \int dR dphi dz 
                             rho(\vec{x}) * sf(\vec{x}) * R

            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT). The factor R 
            is the Jacobian. df the distribution function, sf the 
            selection function and rho the spatial tracer density.
            This total tracer "mass" can be used to normalize the 
            distribution function such that the probability of a star 
            to be within the selection function is 1.
            There are two ways to calculate it:
            a) The brute force version: Mencl(R) is evaluated for Rmax. 
               (Is only used if xgl and wgl are not provided.) 
            b) The elegant and fast version: which makes use 
               of pre-calculated Gauss Legendre points xgl and 
               weights wgl to integrate the spatial density anew 
               over R, z and phi. In this case a member function of the 
               SelectionFunctionsubclass is called to perform the 
               shape-specific integration. (To use this integration 
               version xgl and wgl have to be provided.)
            

        INPUT:

            ngl - (int) - order of Gauss Legendre integration of 
                dM/dR(R) over R in case of a). (Default: 20)

            xgl - (array of floats) - Gauss Legendre points. Has to be 
                provided in case the fast integration in b) should 
                be performed.

            wgl - (array of floats) - Gauss Legendre weights. Has to be 
                provided in case the fast integration in b) should 
                be performed.


        OUTPUT:

            Mtot - (float) - total "mass" of tracers within the 
                selection function. This can be used to normalize the 
                distribution function to the given observed volume.


        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.Mtot(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #if Gauss-Legendre points and weights are already given, choose fast way of integration:
        if xgl is not None and wgl is not None:
            return self._Mtot_fastGL(xgl,wgl)

        #otherwise integrate over each dimension separately:
        return self.Mencl(self._Rmax,ngl=ngl)


    #-------------------------------------------------------------------------------------

    def _dMdvdRdz(self,v,R,z,
                 return_vz=False,return_vR=False,return_vT=False,
                 use_galpy_pv=True,
                 ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,
                 xgl=None,wgl=None):

        """
        NAME:

            _dMdvdRdz

        PURPOSE:

            Auxiliary function used by the _dMdv() function of the 
            SelectionFunction subclasses. It returns the total 
            tracer "mass" at given v_i, R and z according to the 
            distribution function. Here v_i is either vR, vT or vz. 
            I.e. the following integral is performed numerically over 
            the other two velocity components:

            dM/(dv_i dR dz) [R,z,v_i] = \int dv_j dv_k 
                                            df(\vec{x},\vec{v})
                                      =: pv(v_i,R,z)

            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT).
            v_i, v_j, v_k \in [vR,vT,vz] and v_i != v_j != v_k.
            

        INPUT:

            v - (float) - velocity v_i [in galpy units] at which to 
                          evaluate pv(v_i,R,z).

            R - (float) - radius [in galpy units] at which to 
                          evaluate pv(v_i,R,z).

            z - (float) - height [in galpy units] at which to 
                          evaluate pv(v_i,R,z).

            return_vz - (boolean) - flag that sets v_i = vz. The df is 
                        marginalized over vR and vT. (Default: False)

            return_vR - (boolean) - flag that sets v_i = vR. The df is 
                        marginalized over vz and vT. (Default: False)

            return_vT - (boolean) - flag that sets v_i = vT. The df is
                        marginalized over vz and vR. (Default: False)

            use_galpy_pv - (boolean) - flag that forces the function to 
                        use the galpy df.pv functions pvz(), pvR() or 
                        pvT(). If set to False, the functions of this 
                        class _pvz(), _pvR() or _pvT() are used instead. 
                        (Default: True)

            xgl - (array of floats) - Gauss Legendre points. Need to be 
                        set only in case use_galpy_pv is set to False.

            wgl - (array of floats) - Gauss Legendre weights. Need to be 
                        set only in case use_galpy_pv is set to False.


        OUTPUT:

            dMdvdRdz - (float) - total "mass" of tracers at given R and z and 
                        with an velocity of v in the given direction i: 
                        dM/(dv_i dR dz)  [R,z,v_i=v].

        HISTORY:

            2014-09-23 - Started - Wilma Trick (MPIA)
        """

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction._dMdvdRdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #test if input makes sense:
        if sum([return_vz,return_vR,return_vT]) != 1:
            sys.exit("Error in SelectionFunction._dMdvdRdz(): "+
                     "Exactly one of [return_vz,return_vR,return_vT] "+
                     "has to be set to True.")

        #Recursion if input is array:
        if isinstance(z,numpy.ndarray) or isinstance(R,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            return numpy.array([
                                self._dMdvdRdz(
                                    v,rr,zz,
                                    return_vz=return_vz,
                                    return_vR=return_vR,
                                    return_vT=return_vT,
                                    ngl_vel=ngl_vel,
                                    n_sigma=n_sigma,
                                    vT_galpy_max=vT_galpy_max,
                                    use_galpy_pv=use_galpy_pv,
                                    xgl=xgl,wgl=wgl
                                    )
                                for rr,zz in zip(R,z)
                                ])

        if use_galpy_pv:
            #use galpy version of pv:
            if return_vz:
                return self._df.pvz(v,R,z,ngl=ngl_vel,nsigma=n_sigma,vTmax=vT_galpy_max)
            elif return_vR:
                return self._df.pvR(v,R,z,ngl=ngl_vel,nsigma=n_sigma,vTmax=vT_galpy_max)
            elif return_vT:
                return self._df.pvT(v,R,z,ngl=ngl_vel,nsigma=n_sigma)
        else:
            #use the version of pv in this class:
            if return_vz:
                return self._pvz(v,R,z,xgl=xgl,wgl=wgl,nsigma=n_sigma,vTmax=vT_galpy_max)
            elif return_vR:
                return self._pvR(v,R,z,xgl=xgl,wgl=wgl,nsigma=n_sigma,vTmax=vT_galpy_max)
            elif return_vT:
                return self._pvT(v,R,z,xgl=xgl,wgl=wgl,nsigma=n_sigma)

    #---------------------------------------------------------------------------------

    def dMdv(self,v,
            return_vz=False,return_vR=False,return_vT=False,
            ngl=20,_multi=None):
        """
        NAME:

            dMdv

        PURPOSE:

            The total "mass" of tracers within the selection function 
            with a given velocity in a given direction i.
            I.e. the following integration is numerically performed:

            dM/dv_i (v_i) = \int d\vec{x} dv_j dv_k 
                                            df(\vec{x},\vec{v})
                                      =: pv(v_i,R,z)

            where \vec{x} = (R,z,phi) and \vec{v} = (vR,vz,vT).
            v_i, v_j, v_k \in [vR,vT,vz] and v_i != v_j != v_k.
            
            This integral is not normalized. To normalize it the result 
            has to be divided by Mtot (see below).
            This is a wrapper function, as the explicit calculation 
            depends on the shape of the Selection Function and is 
            therefore performed in the corresponding subclass.

        INPUT:

            v - (float) - velocity v_i [in galpy units] at which to 
                          evaluate dMdv.

            return_vz - (boolean) - flag that sets v_i = vz. The df is 
                        marginalized over vR and vT. (Default: False)

            return_vR - (boolean) - flag that sets v_i = vR. The df is 
                        marginalized over vz and vT. (Default: False)

            return_vT - (boolean) - flag that sets v_i = vT. The df is
                        marginalized over vz and vR. (Default: False)

            ngl - (int) - order of Gauss Legendre integration of 
                        dM/dR(R) over R in case of a). (Default: 20)

            _multi - (int) - number of cores on which to do the 
                        calculation of ?????????. IS NOT IMPLEMENTED YET FOR THE SPHERE ?????????
                        (Default: None)


        OUTPUT:

            ?????

        HISTORY:

            2014-10-20 - Started - Wilma Trick (MPIA)
        """
        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.dMdv(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        return self._dMdv(
                    v,
                    return_vz=return_vz,
                    return_vR=return_vR,
                    return_vT=return_vT,
                    ngl=ngl,_multi=_multi
                    )


    #---------------------------------------------------------------------------------

    def velocitySampleDF(self,Rsamples,zsamples,_multi=None,test_sf=False):
        """
        NAME:
            velocitySampleDF
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
        """
        if not len(Rsamples) == len(zsamples): sys.exit("Error in SelectionFunction.velocitySampleDF: number of R and z samples is not the same.")
        
        ndata = len(Rsamples)

        vRs= []
        vTs= []
        vzs= []

        if test_sf: 
            for ii in range(ndata):
                out = self._sampleV_test_sf(n=1)
                vRs.extend(out[:,0])
                vTs.extend(out[:,1])
                vzs.extend(out[:,2])   
        else: 

            if _multi is None or _multi == 1:

                for ii in range(ndata):
                    out = self._sampleV(Rsamples[ii],zsamples[ii],n=1)
                    vRs.extend(out[:,0])
                    vTs.extend(out[:,1])
                    vzs.extend(out[:,2])   


            elif _multi > 1:
                largestClong = 4294967295
                randomseeds = numpy.random.random_integers(0,high=largestClong,size=(ndata))
                multOut= multi.parallel_map((lambda x: self._sampleV(Rsamples[x],zsamples[x],n=1,randomseed=randomseeds[x])),
                                            range(ndata),
                                            numcores=numpy.amin([ndata,multiprocessing.cpu_count(),_multi]))

                for ii in range(ndata):
                    vRs.extend(multOut[ii][:,0])
                    vTs.extend(multOut[ii][:,1])
                    vzs.extend(multOut[ii][:,2])

        vRs = numpy.array(vRs)
        vTs = numpy.array(vTs)
        vzs = numpy.array(vzs)

        return vRs,vTs,vzs    
    #---------------------------------------------------------------------------------


    def _sampleV(self,R,z,n=1,randomseed=None):
        """
        NAME:
           sampleV
        PURPOSE:
           sample a radial, azimuthal, and vertical velocity at R,z
        INPUT:

           R - Galactocentric distance [in galpy units]

           z - height [in galpy units]

           n= number of distances to sample

        OUTPUT:
           list of samples
        HISTORY:
           2012-12-17 - Written - Bovy (IAS)
        """

        #initialize random number generator
        if randomseed is not None:
            numpy.random.seed(seed=randomseed)
        
        #Determine the maximum of the velocity distribution
        
        maxVR= 0.
        maxVz= 0.
        maxVT,fopt,direc,iterations,funcalls,warnflag= scipy.optimize.fmin_powell(_aux_sampleV,1.,args=(R,z,self._df),disp=False,full_output=True)#scipy.optimize.fmin_powell(lambda x: -df(R,0.,x,z,0.,log=True),1.)
        if warnflag > 0:
            print warnflag
        logmaxVD= self._df(R,maxVR,maxVT,z,maxVz,log=True)
        #Now rejection-sample
        vRs= []
        vTs= []
        vzs= []
        while len(vRs) < n:
            nmore= n-len(vRs)+1
            #sample
            propvR= numpy.random.normal(size=nmore)*2.*self._df._sr
            propvT= numpy.random.normal(size=nmore)*2.*self._df._sr+maxVT
            propvz= numpy.random.normal(size=nmore)*2.*self._df._sz
            VDatprop= self._df(R+numpy.zeros(nmore),
                           propvR,propvT,z+numpy.zeros(nmore),
                           propvz,log=True)-logmaxVD
            VDatprop-= -0.5*(propvR**2./4./self._df._sr**2.+propvz**2./4./self._df._sz**2.\
                                 +(propvT-maxVT)**2./4./self._df._sr**2.)
            VDatprop= numpy.reshape(VDatprop,(nmore))
            indx= (VDatprop > numpy.log(numpy.random.random(size=nmore))) #accept
            vRs.extend(list(propvR[indx]))
            vTs.extend(list(propvT[indx]))
            vzs.extend(list(propvz[indx]))
        out= numpy.empty((n,3))
        out[:,0]= vRs[0:n]
        out[:,1]= vTs[0:n]
        out[:,2]= vzs[0:n]
        return out

    #---------------------------------------------------------------------------------

    def _sampleV_test_sf(self,n=1):
        """
        NAME:
           sampleV
        PURPOSE:
           sample a radial, azimuthal, and vertical velocity from a Gaussian distribution
        INPUT:
           n= number of distances to sample

        OUTPUT:
           list of samples
        HISTORY:
           2012-12-17 - Written - Bovy (IAS)
        """
        #Determine the maximum of the velocity distribution
        maxVR= 0.
        maxVz= 0.
        maxVT= 0.9  #randomly picked rotation velocity
        #Now rejection-sample
        vRs= []
        vTs= []
        vzs= []
        #randomly fixed velocity dispersion:
        sigma = 0.3
        #sample
        propvR= numpy.random.normal(size=n)*sigma
        propvT= numpy.random.normal(size=n)*sigma+maxVT
        propvz= numpy.random.normal(size=n)*sigma
        vRs.extend(list(propvR))
        vTs.extend(list(propvT))
        vzs.extend(list(propvz))
        out= numpy.empty((n,3))
        out[:,0]= vRs[0:n]
        out[:,1]= vTs[0:n]
        out[:,2]= vzs[0:n]
        return out

    #---------------------------------------------------------------------------------

    def _pvT(self,vT,R,z,xgl=None,wgl=None,n_sigma=4.):

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction._pvT(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #Recursion if input is array:
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            if not isinstance(vT,numpy.ndarray): vT = vT + numpy.zeros_like(R)
            return numpy.array([self._pvT(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma) for vv,rr,zz in zip(vT,R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(vT,numpy.ndarray): vT = vT + numpy.zeros_like(z)
            return numpy.array([self._pvT(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma) for vv,rr,pp in zip(vT,R,z)])
        elif isinstance(vT,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(vT)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(vT)
            return numpy.array([self._pvT(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma) for vv,rr,pp in zip(vT,R,z)])

        #integration limits:
        fac = n_sigma    #integration from -fac*sigma to +fac*sigma
        sigma_R = self._df._sr * math.exp(-(R-ro) / self._df._hsr) 
        sigma_z = self._df._sz * math.exp(-(R-ro) / self._df._hsz) 
        #R should actually be the guiding star radius... Let's use the sigma at R=1 instead:
        #sigma_z = self._df._sz
        #sigma_R = self._df._sr

        #integration over vz:
        wvz_i = fac * sigma_z * wgl
        xvz_i = fac * sigma_z * xgl

        #integration over vR:
        wvR_j = fac * sigma_R * wgl
        xvR_j = fac * sigma_R * xgl

        #mesh:
        wvz_ij, wvR_ij = numpy.meshgrid(wvz_i,wvR_j,indexing='ij')
        xvz_ij, xvR_ij = numpy.meshgrid(xvz_i,xvR_j,indexing='ij')
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
        wvz_ij         = wvz_ij.flatten()
        wvR_ij         = wvR_ij.flatten()
        xvz_ij         = xvz_ij.flatten()
        xvR_ij         = xvR_ij.flatten()

        #evaluate df on grid:
        df_ij = self._df(R,xvR_ij,vT,z,xvz_ij)

        #sum everything:
        tot = numpy.sum(wvz_ij * wvR_ij * df_ij)

        return tot

    #----------------------------------------------------------------------------------

    def _pvR(self,vR,R,z,xgl=None,wgl=None,n_sigma=4.,vT_galpy_max=1.5):

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction._pvR(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #Recursion if input is array:
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            if not isinstance(vR,numpy.ndarray): vR = vR + numpy.zeros_like(R)
            return numpy.array([self._pvR(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,zz in zip(vR,R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(vR,numpy.ndarray): vR = vR + numpy.zeros_like(z)
            return numpy.array([self._pvR(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,pp in zip(vR,R,z)])
        elif isinstance(vR,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(vR)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(vR)
            return numpy.array([self._pvR(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,pp in zip(vR,R,z)])

        #integration limits for vz:
        fac_z = n_sigma    #integration from -fac*sigma to +fac*sigma
        sigma_z = self._df._sz * math.exp(-(R-ro) / self._df._hsz) 
        #sigma_z = self._df._sz

        #integration limits for vT:
        #from 0 to fac_T * vT(R_sun), and vT(R_sun) ~ 1. in galpy units
        fac_T = vT_galpy_max
        max_T = fac_T * 1.

        #integration over vz:
        wvz_i = fac_z * sigma_z * wgl
        xvz_i = fac_z * sigma_z * xgl

        #integration over vT:
        wvT_j = 0.5 * max_T * wgl
        xvT_j = 0.5 * max_T * (xgl + 1.)

        #mesh:
        wvz_ij, wvT_ij = numpy.meshgrid(wvz_i,wvT_j,indexing='ij')
        xvz_ij, xvT_ij = numpy.meshgrid(xvz_i,xvT_j,indexing='ij')
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
        wvz_ij         = wvz_ij.flatten()
        wvT_ij         = wvT_ij.flatten()
        xvz_ij         = xvz_ij.flatten()
        xvT_ij         = xvT_ij.flatten()

        #evaluate df on grid:
        df_ij = self._df(R,vR,xvT_ij,z,xvz_ij)

        #sum everything:
        tot = numpy.sum(wvz_ij * wvT_ij * df_ij)

        return tot

    #----------------------------------------------------------------------------------

    def _pvz(self,vz,R,z,xgl=None,wgl=None,n_sigma=4.,vT_galpy_max=1.5):

        #Recursion if input is array:
        if isinstance(R,numpy.ndarray):
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)
            if not isinstance(vz,numpy.ndarray): vz = vz + numpy.zeros_like(R)
            return numpy.array([self._pvz(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,zz in zip(vz,R,z)])
        elif isinstance(z,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(vz,numpy.ndarray): vz = vz + numpy.zeros_like(z)
            return numpy.array([self._pvz(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,pp in zip(vz,R,z)])
        elif isinstance(vz,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(vz)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(vz)
            return numpy.array([self._pvz(self,vv,rr,zz,xgl=xgl,wgl=wgl,n_sigma=n_sigma,vT_galpy_max=vT_galpy_max) for vv,rr,pp in zip(vz,R,z)])

        #integration limits for vR:
        fac_R = n_sigma    #integration from -fac*sigma to +fac*sigma
        sigma_R = self._df._sr * math.exp(-(R-ro) / self._df._hsr) 
        #sigma_R = self._df._sr

        #integration limits for vT:
        #from 0 to fac_T * vT(R_sun), and vT(R_sun) ~ 1. in galpy units
        fac_T = vT_galpy_max
        max_T = fac_T * 1.

        #integration over vR:
        wvR_i = fac_R * sigma_R * wgl
        xvR_i = fac_R * sigma_R * xgl

        #integration over vT:
        wvT_j = 0.5 * max_T * wgl
        xvT_j = 0.5 * max_T * (xgl + 1.)

        #mesh:
        wvR_ij, wvT_ij = numpy.meshgrid(wvR_i,wvT_j,indexing='ij')
        xvR_ij, xvT_ij = numpy.meshgrid(xvR_i,xvT_j,indexing='ij')
                # (*Note:* careful with meshgrid: the default indexing 'xy' 
                #          works here as well, but when using matrix indexing, 
                #          e.g. v[i,j], later, this can lead to ugly bugs.)
        wvR_ij         = wvR_ij.flatten()
        wvT_ij         = wvT_ij.flatten()
        xvR_ij         = xvR_ij.flatten()
        xvT_ij         = xvT_ij.flatten()

        #evaluate df on grid:
        df_ij = self._df(R,xvR_ij,xvT_ij,z,vz)

        #sum everything:
        tot = numpy.sum(wvR_ij * wvT_ij * df_ij)

        return tot

    #---------------------------------------------------------------------------------

    def spatialSampleDF_measurementErrors(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,
                                           e_radec_rad=None,e_DM_mag=None,
                                           Xsun_kpc=8.,Ysun_kpc=0.,Zsun_kpc=0.,
                                           spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.):

        if self._with_incompleteness:
            sys.exit("Error in SelectionFunction.spatialSampleDF_measurementErrors(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        return self._spatialSampleDF_measurementErrors(
                                        nmock=nmock,
                                        nrs=nrs,nzs=nzs,
                                        ngl_vel=ngl_vel,
                                        n_sigma=n_sigma,
                                        vT_galpy_max=vT_galpy_max,
                                        quiet=quiet,test_sf=test_sf,_multi=_multi,
                                        e_radec_rad=e_radec_rad,
                                        e_DM_mag=e_DM_mag,
                                        Xsun_kpc=Xsun_kpc,Ysun_kpc=Ysun_kpc,Zsun_kpc=Zsun_kpc,
                                        spatialGalpyUnits_in_kpc=spatialGalpyUnits_in_kpc,
                                        velocityGalpyUnits_in_kms=velocityGalpyUnits_in_kms)

    #---------------------------------------------------------------------------------

    def velocitySampleDF_measurementErrors(self,R_kpc_true, z_kpc_true, phi_deg_true,
                                             R_kpc_error,z_kpc_error,phi_deg_error,
                                             e_vlos_kms=None,e_pm_masyr=None,
                                             spatialGalpyUnits_in_kpc=8.,velocityGalpyUnits_in_kms=230.,
                                             Xsun_kpc=8.,Ysun_kpc=0.,Zsun_kpc=0.,
                                             vXsun_kms=0.,vYsun_kms=230.,vZsun_kms=0.,
                                             _multi=None,test_sf=False):
        """
        NAME:
            velocitySampleDF_measurementErrors
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
        """
        
        #_____draw true velocities from df for true positions_____
        Rs = R_kpc_true / spatialGalpyUnits_in_kpc
        zs = z_kpc_true / spatialGalpyUnits_in_kpc

        # actual mock data sampling:
        vRs,vTs,vzs = self.velocitySampleDF(Rs,zs,_multi=_multi,test_sf=test_sf) 

        vR_kms_true = vRs * velocityGalpyUnits_in_kms
        vT_kms_true = vTs * velocityGalpyUnits_in_kms
        vz_kms_true = vzs * velocityGalpyUnits_in_kms

        #_____convert to observable coordinates_____
        phi_rad_true = phi_deg_true / 180. * math.pi
        out = galcencyl_to_radecDMvlospmradec(
                                    R_kpc_true, phi_rad_true, z_kpc_true,
                                    vR_kms_true,vT_kms_true,  vz_kms_true,
                                    quiet=True,
                                    Xsun_kpc=Xsun_kpc,
                                    Ysun_kpc=Ysun_kpc,
                                    Zsun_kpc=Zsun_kpc,
                                    vXsun_kms=vXsun_kms,
                                    vYsun_kms=vYsun_kms,
                                    vZsun_kms=vZsun_kms
                                    )
        ra_rad_t       = out[0]
        dec_rad_t      = out[1]
        DM_mag_t       = out[2]
        vlos_kms_t     = out[3]
        pm_ra_masyr_t  = out[4]
        pm_dec_masyr_t = out[5]

        phi_rad_error = phi_deg_error / 180. * math.pi
        out = galcencyl_to_radecDM(
                        R_kpc_error,phi_rad_error,z_kpc_error,
                        quiet=True,
                        Xsun_kpc=Xsun_kpc,Ysun_kpc=Ysun_kpc,Zsun_kpc=Zsun_kpc
                        )
        ra_rad_error  = out[0]
        dec_rad_error = out[1]
        DM_mag_error  = out[2]

        #_____perturb according to measurement errors_____
        if e_vlos_kms is None or e_pm_masyr is None:
            sys.exit("Error in SelectionFunction."+\
                     "velocitySampleDF_measurementErrors(): errors"+\
                     " on proper motion "+\
                     "(e_pm_masyr) and on line-of-sight velocity "+\
                     "(e_vlos_kms) have to be set.")

        # Draw random numbers for random gaussian errors:
        ndata = len(R_kpc_true)
        eta = numpy.random.randn(3,ndata)

        # Perturb data according to Gaussian distribution:
        pm_ra_masyr_error   = eta[0,:] * e_pm_masyr + pm_ra_masyr_t
        pm_dec_masyr_error  = eta[1,:] * e_pm_masyr + pm_dec_masyr_t
        vlos_kms_error      = eta[2,:] * e_vlos_kms + vlos_kms_t

        #_____transform back to cylindrical coordinates_____
        out = radecDMvlospmradec_to_galcencyl(
                                    ra_rad_error,dec_rad_error,DM_mag_error,
                                    vlos_kms_error,pm_ra_masyr_error,pm_dec_masyr_error,
                                    quiet=True,
                                    Xsun_kpc=Xsun_kpc,
                                    Ysun_kpc=Ysun_kpc,
                                    Zsun_kpc=Zsun_kpc,
                                    vXsun_kms=vXsun_kms,
                                    vYsun_kms=vYsun_kms,
                                    vZsun_kms=vZsun_kms
                                    )
        vR_kms_error  = out[3]
        vT_kms_error  = out[4]
        vz_kms_error  = out[5]

        out = galcencyl_to_radecDMvlospmradec(
                                    R_kpc_true, phi_rad_true, z_kpc_true,
                                    vR_kms_true, vT_kms_true, vz_kms_true,
                                    quiet=True,
                                    Xsun_kpc=Xsun_kpc,
                                    Ysun_kpc=Ysun_kpc,
                                    Zsun_kpc=Zsun_kpc,
                                    vXsun_kms=vXsun_kms,
                                    vYsun_kms=vYsun_kms,
                                    vZsun_kms=vZsun_kms
                                    )
        vlos_kms_true     = out[3]
        pm_ra_masyr_true  = out[4]
        pm_dec_masyr_true = out[5]

        return (numpy.array(vR_kms_true),   numpy.array(vz_kms_true),      numpy.array(vT_kms_true), \
                numpy.array(vR_kms_error),  numpy.array(vz_kms_error),     numpy.array(vT_kms_error), \
                numpy.array(vlos_kms_error),numpy.array(pm_ra_masyr_error),numpy.array(pm_dec_masyr_error), \
                numpy.array(vlos_kms_true),numpy.array(pm_ra_masyr_true),numpy.array(pm_dec_masyr_true))


#---------------------------------------------------------------------------------

    def setup_velocity_integration_grid(self,df_fid,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5):
        """
        NAME:

            setup_velocity_integration_grid

        PURPOSE:

            The fiducial distribution function is used 
            for only the following:
            *) the velocity dispersion of the fiducial df is used to 
               determine the integration ranges over velocity.
            
            TO DO: WRITE MORE DESCRIPTION????????????????

            
        INPUT:

            TO DO: UPDATE???????????????????????????


            nrs - (int) - number of grid points in radial 
                        direction between the smallest and largest 
                        radius within the selection function on which 
                        the density will be explicitly calculated 
                        before interpolating. (Default: 16)

            nzs - (int) - number of grid points in vertical 
                        direction between the smallest height (or zero) 
                        and the largest height above the plane within 
                        the selection function on which the density 
                        will be explicitly calculated before inter-
                        polating. (Default: 16)

            ngl_vel - (int) - order of Gauss Legendre integration in each of 
                        the three velocity dimensions to pre-calculate 
                        the actions used to later calculate the spatial 
                        density at a given (R,z) point. (Default: 20)

        HISTORY:

            2015-01-08 - Started - Wilma Trick (MPIA)
        """

        # set flag:
        self._use_fiducial_actions_Trick = False    #have to be calculated first
        self._n_sigma_fiducial_fit_range = n_sigma
        self._vT_galpy_fiducial_fit_range = vT_galpy_max

        #shape and dimension of the arrays:
        ngl_vR = ngl_vel
        ngl_vz = ngl_vel
        ngl_vT = ngl_vel
        shapetuple      = (nrs,nzs,ngl_vR,ngl_vT,ngl_vz)
        dim = numpy.array([nrs,nzs,ngl_vR,ngl_vT,ngl_vz])

       #_____split up integration in velocity in different intervals_____

        if (ngl_vel % 2) != 0: 
            sys.exit("Error in SelectionFunction.setup_velocity_integration_grid(): ngl_vel has to be an even number.")

        """#...how many GL points in each integration interval?
        ngl_vel_IN = int(math.floor(3./5. * ngl_vel))
        if (ngl_vel_IN % 2) != 0: ngl_vel_IN += 1
        ngl_vel_OUT = ngl_vel - ngl_vel_IN

        #...GL points and weights:
        xgl_v_IN,wgl_v_IN       = numpy.polynomial.legendre.leggauss(ngl_vel_IN)
        xgl_v_OUT12,wgl_v_OUT12 = numpy.polynomial.legendre.leggauss(ngl_vel_OUT/2)

        wgl_v = numpy.concatenate((wgl_v_OUT12,wgl_v_IN,wgl_v_OUT12))
        xgl_v = numpy.concatenate((xgl_v_OUT12,xgl_v_IN,xgl_v_OUT12))

        #...integration limits in terms of some sigma:
        vmin_unscaled = numpy.repeat([-4.,-2.,+2.],[ngl_vel_OUT/2,ngl_vel_IN,ngl_vel_OUT/2])
        vmax_unscaled = numpy.repeat([-2.,+2.,+4.],[ngl_vel_OUT/2,ngl_vel_IN,ngl_vel_OUT/2])

        #Gauss Legendre points & weights for vT integration:
        xgl_vT,wgl_vT   = numpy.polynomial.legendre.leggauss(ngl_vT)"""

        #like Bovy:
        #...GL points and weights:
        xgl_v12,wgl_v12       = numpy.polynomial.legendre.leggauss(ngl_vel/2)
        wgl_v = numpy.concatenate((wgl_v12,wgl_v12))
        xgl_v = numpy.concatenate((xgl_v12,xgl_v12))

        #...integration limits in terms of some sigma:
        #like Bovy: vmin_unscaled = numpy.repeat([-4.,0.],[ngl_vel/2,ngl_vel/2]) 
        #like Bovy: vmax_unscaled = numpy.repeat([0.,+4.],[ngl_vel/2,ngl_vel/2]) 
        vmin_unscaled = numpy.repeat([-n_sigma,       0.],[ngl_vel/2,ngl_vel/2]) 
        vmax_unscaled = numpy.repeat([       0.,+n_sigma],[ngl_vel/2,ngl_vel/2])  

        #Gauss Legendre points & weights for vT integration:
        xgl_vT,wgl_vT   = numpy.polynomial.legendre.leggauss(ngl_vT)

        """#???
        #...GL points and weights:
        xgl_v,wgl_v       = numpy.polynomial.legendre.leggauss(ngl_vel)

        #...integration limits in terms of some sigma:
        vmin_unscaled = numpy.repeat([-4.],[ngl_vel])
        vmax_unscaled = numpy.repeat([+4.],[ngl_vel])

        #Gauss Legendre points & weights for vT integration:
        xgl_vT,wgl_vT   = numpy.polynomial.legendre.leggauss(ngl_vT)
        #???"""

        #_____setup grid_____

        # for each point (R,z) we want to integrate over all three velocities, 
        # using the Gauss-Legendre integration we need ngl_vel**3 actions for each 
        # point (R,z), i.e. each ngl_vel actions per velocity direction.

        # rectangular (R,z) grid:
        Rn = numpy.zeros(shapetuple)
        zm = numpy.zeros(shapetuple)

        # Gauss Legendre grid for velocities:
        vzk = numpy.zeros(shapetuple)
        vTj = numpy.zeros(shapetuple)
        vRi = numpy.zeros(shapetuple)

        # Gauss Legendre weights for velocity integration:
        Wk = numpy.zeros(shapetuple)
        Wj = numpy.zeros(shapetuple)
        Wi = numpy.zeros(shapetuple)

        #-----R-----
        #grid limits:
        Rs = numpy.linspace(self._Rmin,self._Rmax,nrs)
        self._Rs_fid = Rs
        #-----z-----
        #grid limits:
        if self._zmax < 0.:
            zup  = numpy.fabs(self._zmin)
            zlow = numpy.fabs(self._zmax)
        else:
            zup  = max(numpy.fabs(self._zmax),numpy.fabs(self._zmin))
            zlow = max(0.,self._zmin)
        #grid points:
        zs                  = numpy.linspace(zlow,zup,nzs)
        self._zs_fid        = zs
        #extend to proper shape:
        for mm in range(nzs):
            zm [:,mm,:,:,:] = zs[mm]

        #NOTE TO MYSELF: This could be made better! In case of a selection function that is not a wedge, here we calculate actions at R's and z's that are never actually used. When modifying this, I also could save time.

        
        #-----R-----
        for nn in range(nrs):
            Rnt = Rs[nn]
                    
            #-----vz-----
            #integration limits (the fiducial df is only used to set the grid):
            ro = df_fid._ro
            sigmaZn = df_fid._sz * numpy.exp((ro-Rnt)/df_fid._hsz)
            vZmaxn = vmax_unscaled * sigmaZn
            vZminn = vmin_unscaled * sigmaZn
            for kk in range(ngl_vz):
                #calculate GL points/weights:
                Wkt  = 0.5*(vZmaxn[kk]-vZminn[kk])*wgl_v[kk]
                vzkt = 0.5*(vZmaxn[kk]-vZminn[kk])*xgl_v[kk] + 0.5*(vZmaxn[kk]+vZminn[kk])

                #-----vT-----
                #integration limits:
                vTmin = 0.
                vTmax = vT_galpy_max #default: 1.5
                for jj in range(ngl_vT):
                    #calculate GL points/weights:
                    Wjt  = 0.5*(vTmax-vTmin)*wgl_vT[jj]
                    vTjt = 0.5*(vTmax-vTmin)*xgl_vT[jj] + 0.5*(vTmax+vTmin)
                    
                    #-----vR-----
                    #integration limits (the fiducial df is only used to set the grid):
                    sigmaRn = df_fid._sr * numpy.exp((ro-Rnt)/df_fid._hsr)
                    vRmaxn  = vmax_unscaled * sigmaRn
                    vRminn  = vmin_unscaled * sigmaRn
                    for ii in range(ngl_vR):
                        #calculate GL points/weights:
                        Wit  = 0.5*(vRmaxn[ii]-vRminn[ii])*wgl_v[ii]
                        vRit = 0.5*(vRmaxn[ii]-vRminn[ii])*xgl_v[ii] + 0.5*(vRmaxn[ii]+vRminn[ii])

                        #assign to grid:
                        Rn [nn,:,ii,jj,kk] = Rnt
                        vRi[nn,:,ii,jj,kk] = vRit
                        vTj[nn,:,ii,jj,kk] = vTjt
                        vzk[nn,:,ii,jj,kk] = vzkt
                        Wi [nn,:,ii,jj,kk] = Wit
                        Wj [nn,:,ii,jj,kk] = Wjt
                        Wk [nn,:,ii,jj,kk] = Wkt



        #_____store grid points_____
        # **Note:** vig = Velocity Integration Grid
        self._R_vig = Rn
        self._z_vig = zm
        self._vR_vig = vRi
        self._vT_vig = vTj
        self._vz_vig = vzk

        #Gauss Legendre weights:
        self._weights_vig = Wi * Wj * Wk

    #-------------------------------------------------------------------

    def calculate_actions_on_vig_using_fid_pot(self,df_fid,_multi=None):

        if self._R_vig is None:
            sys.exit("Error in SelectionFunction.calculate_actions_on_vig_using_fid_pot(): The velocity integration grid has to be setupo before this function can be called. Use SelectionFunction.setup_velocity_integration_grid().")

        # set flag and fiducial df and reset df and density:
        self.set_fiducial_df(df_fid)
        self._use_fiducial_actions_Bovy = False
        self._use_fiducial_actions_Trick = True

        """#use the action-angle object of the fiducial distribution function / potential object:
        if self._df_fid is None:
            sys.exit("Error in SelectionFunction.calculate_fiducial_actions_of_velocity_integration_grid(): This function uses "+\
                    "the action-angle-object of the fiducial df (and its "+\
                    "potential). _fid_df has to be set (with a given "+\
                    "potential and action-angle object) before this "+\
                    "function can be evoked.")"""

        #input & output shape:
        shapetuple = numpy.shape(self._R_vig)


        if _multi is None or _multi == 1:
            #...evaluation on one core:
            import time
            start = time.time()
            out = self._df_fid(
                            numpy.ravel(self._R_vig),
                            numpy.ravel(self._vR_vig),
                            numpy.ravel(self._vT_vig),
                            numpy.ravel(self._z_vig),
                            numpy.ravel(self._vz_vig),
                            log=True,
                            _return_actions=True,
                            _return_freqs=True
                            )
            print 'total calculation:',time.time()-start
            jr_data    = out[1]
            lz_data    = out[2]
            jz_data    = out[3]
            rg_data    = out[4]
            kappa_data = out[5]
            nu_data    = out[6]
            Omega_data = out[7]
            
 
        elif _multi > 1:
            #...evaluation on multiple cores:

            #number of data points:
            ndata = self._R_vig.size

            #prepare output for actions and frequencies:
            jr_data    = numpy.zeros(ndata)
            lz_data    = numpy.zeros(ndata)
            jz_data    = numpy.zeros(ndata)
            rg_data    = numpy.zeros(ndata)
            kappa_data = numpy.zeros(ndata)
            nu_data    = numpy.zeros(ndata)
            Omega_data = numpy.zeros(ndata)

            # number of cores to use: N
            N = numpy.amin([
                    ndata,
                    multiprocessing.cpu_count(),
                    _multi
                    ])
            # data points to evaluate on one core:
            M = int(math.floor(ndata / N))

            # first evaluate arrays on each core to make use of 
            # the fast evaluation of input arrays:
            multiOut =  multi.parallel_map(
                                (lambda x: self._df_fid(
                                    numpy.ravel(self._R_vig) [x*M:(x+1)*M],
                                    numpy.ravel(self._vR_vig)[x*M:(x+1)*M],
                                    numpy.ravel(self._vT_vig)[x*M:(x+1)*M],
                                    numpy.ravel(self._z_vig) [x*M:(x+1)*M],
                                    numpy.ravel(self._vz_vig)[x*M:(x+1)*M],
                                    log=True,
                                    _return_actions=True,
                                    _return_freqs=True
                                    )),
                                range(N),
                                numcores=N
                                )
            for x in range(N):
                jr_data   [x*M:(x+1)*M] = multiOut[x][1]
                lz_data   [x*M:(x+1)*M] = multiOut[x][2]
                jz_data   [x*M:(x+1)*M] = multiOut[x][3]
                rg_data   [x*M:(x+1)*M] = multiOut[x][4]
                kappa_data[x*M:(x+1)*M] = multiOut[x][5]
                nu_data   [x*M:(x+1)*M] = multiOut[x][6]
                Omega_data[x*M:(x+1)*M] = multiOut[x][7]

            # number of data points not yet evaluated:
            K = ndata % N

            # now calculate the rest of the data:
            if K > 0:
                    
                multiOut =  multi.parallel_map(
                                    (lambda x: self._df_fid(
                                        numpy.ravel(self._R_vig) [N*M+x],
                                        numpy.ravel(self._vR_vig)[N*M+x],
                                        numpy.ravel(self._vT_vig)[N*M+x],
                                        numpy.ravel(self._z_vig) [N*M+x],
                                        numpy.ravel(self._vz_vig)[N*M+x],
                                        log=True,
                                        _return_actions=True,
                                        _return_freqs=True
                                        )),
                                    range(K),
                                    numcores=numpy.amin([
                                        K,
                                        multiprocessing.cpu_count(),
                                        _multi
                                        ])
                                    )
                for x in range(K):
                    jr_data   [N*M+x] = multiOut[x][1]
                    lz_data   [N*M+x] = multiOut[x][2]
                    jz_data   [N*M+x] = multiOut[x][3]
                    rg_data   [N*M+x] = multiOut[x][4]
                    kappa_data[N*M+x] = multiOut[x][5]
                    nu_data   [N*M+x] = multiOut[x][6]
                    Omega_data[N*M+x] = multiOut[x][7]

        #back to original shape:
        self._jr_vig    = numpy.reshape(jr_data   ,shapetuple)
        self._lz_vig    = numpy.reshape(lz_data   ,shapetuple)
        self._jz_vig    = numpy.reshape(jz_data   ,shapetuple)
        self._rg_vig    = numpy.reshape(rg_data   ,shapetuple)
        self._kappa_vig = numpy.reshape(kappa_data,shapetuple)
        self._nu_vig    = numpy.reshape(nu_data   ,shapetuple)
        self._Omega_vig = numpy.reshape(Omega_data,shapetuple)

   #---------------------------------------------------------------------------------------------

    """def set_VIG_actions(self,jr_vig,lz_vig,jz_vig,rg_vig,kappa_vig,nu_vig,Omega_vig,R_vig,z_vig,GL_weights_vig):

        nrs = len(R_vig[:,0,0,0,0])
        nzs = len(R_vig[0,:,0,0,0])

        if (self._Rmin is None) or (self._Rmax is None) or (self._zmin is None) or (self._zmax is None):
            sys.exit("Error in SelectionFunction.set_VIG_actions(): This selection function has to be initialized with a volume shape first. Rmin, Rmax, zmin and zmax are not known.")

        #grid limits in R:
        self._Rs_fid = numpy.linspace(self._Rmin,self._Rmax,nrs)
        testR = (R_vig[:,0,0,0,0] != self._Rs_fid)
        if numpy.sum(testR) > 0: sys.exit("Error in SelectionFunction.set_VIG_actions(): The R-component of the velocity integration grid (R_vig) is not the same as the grid given by the volume shape of this selection function.")

        #grid limits in z:
        if self._zmax < 0.:
            zup  = numpy.fabs(self._zmin)
            zlow = numpy.fabs(self._zmax)
        else:
            zup  = max(numpy.fabs(self._zmax),numpy.fabs(self._zmin))
            zlow = max(0.,self._zmin)
        self._zs_fid = numpy.linspace(zlow,zup,nzs)
        testZ = (z_vig[0,:,0,0,0] != self._zs_fid)
        if numpy.sum(testZ) > 0: sys.exit("Error in SelectionFunction.set_VIG_actions(): The z-component of the velocity integration grid (z_vig) is not the same as the grid given by the volume shape of this selection function.")

        #actions on velocity integration grid (VIG):
        self._jr_vig    = jr_vig
        self._lz_vig    = lz_vig
        self._jz_vig    = jz_vig
        self._rg_vig    = rg_vig
        self._kappa_vig = kappa_vig
        self._nu_vig    = nu_vig
        self._Omega_vig = Omega_vig
        self._R_vig     = R_vig
        self._z_vig     = z_vig
        self._weights_vig = GL_weights_vig"""

    #----------------------------------------------------------------

    def set_incompleteness_function(self,incompleteness_function,incompleteness_maximum):

        self._with_incompleteness     = True
        self._incompleteness_function = incompleteness_function
        self._incompleteness_maximum  = incompleteness_maximum

   #---------------------------------------------------------------------------------------------

    def _spatialSampleDF_incomplete(self,nmock=500,nrs=16,nzs=16,ngl_vel=20,n_sigma=4.,vT_galpy_max=1.5,quiet=False,test_sf=False,_multi=None,recalc_densgrid=True):

        #_____is incompleteness function initializeed?_____
        if self._incompleteness_function is None or self._incompleteness_maximum is None:
            sys.exit("Error in SelectionFunction._spatialSampleDF_incomplete(): "+
                     "There is no completeness function and/or peak. "+
                     "The completeness function should be a function of distance "+
                     "from sun and height above the plane")
        if not recalc_densgrid:
            sys.exit("Error in SelectionFunction._spatialSampleDF_incomplete(): "+
                     "Keyword recalc_densgrid is not set to True. "+
                     "This function will always recalculate the density grid.")


        #_____initialize interpolated density grid_____
        if not quiet: print "Initialize (complete) interpolated density grid"
        self.densityGrid(
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
        Rarr = []
        zarr = []
        phiarr = []

        if not quiet: print "Start sampling"

        while nfound < nmock:

            # how many more mock data points are needed:
            nmore = nmock - nfound

            #____sample spatial mock data_____
            Rs, zs, phis_deg = self._spatialSampleDF_complete(
                                   nmock=nmore,
                                   ngl_vel=ngl_vel,    #used to calculate the peak density
                                   n_sigma=n_sigma,    #-- " --
                                   vT_galpy_max=vT_galpy_max, #-- " --
                                   recalc_densgrid=False,
                                   quiet=True
                                   )
            phis_rad = math.pi * phis_deg / 180.

            #_____calculate distance from sun_____ 
            xyz = bovy_coords.galcencyl_to_XYZ(Rs,phis_rad,zs,Xsun=self._dsun,Ysun=0.,Zsun=0.)
            Xs = xyz[0]
            Ys = xyz[1]
            Zs = xyz[2]
            ds = numpy.sqrt(Xs**2 + Ys**2 + Zs**2)

            #_____Apply incmpleteness using the rejection method:_____
            #random number:
            etas = numpy.random.rand(nmore)
            #max completeness at distances:
            comp_at_dz = self._incompleteness_function(ds,zs)
            #max completeness at peak * random number < 1:
            comptest = self._incompleteness_maximum * etas
            #accepted data points:
            index = (comptest <= comp_at_dz)
            if numpy.sum(index) == 0:
                pass
            elif numpy.sum(index) == 1:
                Rarr.extend([Rs[index]])
                zarr.extend([zs[index]])
                phiarr.extend([phis_deg[index]])
            else:
                Rarr.extend(Rs[index])
                zarr.extend(zs[index])
                phiarr.extend(phis_deg[index])
            #update counters:
            nfound += numpy.sum(index)
            nreject += nmore - numpy.sum(index)
            if not quiet: print "Found: ",nfound, ", Reject: ",nreject

        return numpy.array(Rarr),numpy.array(zarr),numpy.array(phiarr)


#---------------------------------------------------------------------------------

def _aux_sampleV(vT,*args):
    R = args[0]
    z = args[1]
    df = args[2]
    if isinstance(vT,numpy.ndarray): 
        return numpy.array([_aux_sampleV(vv,*args) for vv in vT])
    res = -df(R,0.,vT,z,0.,log=True)
    return res





