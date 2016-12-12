from SelectionFunction import SelectionFunction
import sys
import numpy
import math
import scipy
import matplotlib.pyplot as plt
import time
from galpy.util import multi
import multiprocessing

class SF_Wedge(SelectionFunction):
    """Class that implements a wedge-shaped selection function"""
    def __init__(self,Rmin,Rmax,zmin,zmax,pmin,pmax,df=None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a wedge-shaped selection function
        INPUT:
            Rmin, Rmax: minimum and maximum galactocentric radius [_REFR0*ro]
            zmin, zmax: minimum and maximum height above the plane [_REFR0*ro]
            pmin, pmax: minimum and maximum azimuth angle [degrees]
        OUTPUT:
            wedge-shaped selection function object
        HISTORY:
        """
        SelectionFunction.__init__(self,df=df)

        if zmin > zmax:
            sys.exit("Error in SF_Wedge.__init__"+\
                     "(Rmin,Rmax,zmin,zmax,pmin,pmax): "+\
                     "zmin has to be smaller than zmax.")
        if Rmin > Rmax:
            sys.exit("Error in SF_Wedge.__init__"+\
                     "(Rmin,Rmax,zmin,zmax,pmin,pmax): "+\
                     "Rmin has to be smaller than Rmax.")
        if pmin > pmax:
            sys.exit("Error in SF_Wedge.__init__"+\
                     "(Rmin,Rmax,zmin,zmax,pmin,pmax): "+\
                     "phimin has to be smaller than phimax.")

        #Borders of the wedge-shaped selection function:
        self._Rmin = Rmin
        self._Rmax = Rmax
        self._zmin = zmin
        self._zmax = zmax
        self._pmin = pmin
        self._pmax = pmax

        return None


    #---------------------------------------------------------------------

    def _contains(self,R,z,phi=None):

        sys.exit("Error in SF_Wedge._contains(): This method is not implemented yet. TO DO!!!!!!!!!!!!!!!!!!!!!")

        if self._with_incompleteness:
            sys.exit("Error in SF_Wedge._contains(): "+
                     "Function not yet implemented to take care of imcompleteness.")

    #---------------------------------------------------------------------

    def _densfunc(self,R,z,phi=None,set_outside_zero=False,throw_error_outside=False,consider_incompleteness=False):

        if self._densInterp is None:
            sys.exit("Error in SF_Wedge._densfunc(): "+\
                     "self._densInterp is None. Initialize density grid"+\
                     " first before calling this function.")

        if consider_incompleteness and self._with_incompleteness:
            sys.exit("Error in SF_Wedge._densfunc(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        #take care of array input:
        if isinstance(z,numpy.ndarray) or isinstance(R,numpy.ndarray):
            if not isinstance(R,numpy.ndarray): R = R + numpy.zeros_like(z)
            if not isinstance(z,numpy.ndarray): z = z + numpy.zeros_like(R)

        #in case we are outside of observed volume:
        if throw_error_outside or set_outside_zero:

            if phi is None: sys.exit("Error in SF_Wedge._densfunc(): "+\
               "Specify phi in case of the wedge-shaped selection function, "+\
               "when using throw_error_outside=True or set_outside_zero=True.")

            outside = (R < self._Rmin) + (R > self._Rmax) +\
                      (z < self._zmin) + (z > self._zmax) +\
                      (phi < self._pmin) + (phi > self._pmax)

            if numpy.sum(outside) > 0:
                if throw_error_outside:
                    print "R: ",self._Rmin," <= ",R[outside]," <= ",self._Rmax,"?"
                    print "z: ",self._zmin," <= ",z[outside]," <= ",self._zmax,"?"
                    print "phi: ",self._pmin," <= ",phi[outside]," <= ",self._pmax,"?"
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

    #-----------------------------------------------------------------

    def _fastGLint_wedge(self,func,xgl,wgl):

        """integrate the given function func(R,z,phi) over the wedge-shaped effective volume 
           by hand, analogous to Bovy, using Gauss Legendre quadrature."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Wedge._fastGLint_wedge(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        
        #transform gauss legendre points and weights to correct integration limits:
        Rgl = 0.5 * (self._Rmax - self._Rmin) * (xgl + 1.) + self._Rmin
        zgl = 0.5 * (self._zmax - self._zmin) * (xgl + 1.) + self._zmin
        wRgl = 0.5 * (self._Rmax - self._Rmin) * wgl
        wzgl = 0.5 * (self._zmax - self._zmin) * wgl
        
        #multiply with Jacobian R, because dV = R dR dphi dz:
        wRgl = wRgl * Rgl
        
        #integration factor for integration over phi:
        int_phi = (self._pmax - self._pmin) * math.pi / 180.    #[rad]
        
        #make grid for integration over 2D:
        Rglarr, zglarr   = numpy.meshgrid(Rgl,zgl,indexing='ij')
        wRglarr, wzglarr = numpy.meshgrid(wRgl,wzgl,indexing='ij') 
        Rglarr  = Rglarr.flatten()
        zglarr  = zglarr.flatten()
        wRglarr = wRglarr.flatten()
        wzglarr = wzglarr.flatten()
        # (*Note:* careful with meshgrid: the default indexing 'xy' 
        #          works here as well, but when using matrix indexing, 
        #          e.g. v[i,j], later, this can lead to ugly bugs.)
        
        #calculate density at each grid point:
        phi = 0.
        densarr = func(Rglarr,zglarr,phi)
        
        #total mass in selection function:
        tot = int_phi * numpy.sum(wRglarr * wzglarr * densarr)
        return tot

    #-----------------------------------------------------------------

    def _Mtot_fastGL(self,xgl,wgl):

        """integrate total mass inside effective volume by hand, analogous to Bovy, using Gauss Legendre quadrature.
           The integration accounts for integration limits - we therefore do not have to set the density outside the wedge to zero."""

        if self._with_incompleteness:
            sys.exit("Error in SF_Wedge._Mtot_fastGL(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        
        #define function func(R,z,phi) to integrate:
        func = lambda rr,zz,pp: self._densfunc(rr,zz,phi=pp,set_outside_zero=False,throw_error_outside=True,consider_incompleteness=False)
        
        #total mass in selection function:
        Mtot = self._fastGLint_wedge(func,xgl,wgl)
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
        densmax = self._df.density(self._Rmin,zprime,ngl=ngl_vel,nsigma=n_sigma,vTmax=vT_galpy_max)
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
            
            #Wedge selection function:
            phi = (self._pmax - self._pmin) * eta[0] + self._pmin  #uniform distributed
            z = (self._zmax - self._zmin) * eta[1] + self._zmin   #uniform distributed
            R = math.sqrt((self._Rmax**2 - self._Rmin**2) * eta[2] + self._Rmin**2)    #p(R) ~ R for wedge, i.e. P(<R) ~ R^2 --> R ~ sqrt(y) (according to inverse function method)

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
        
        if isinstance(R,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdR(rr,ngl=ngl) for rr in R])

        a = math.radians(numpy.fabs(self._pmax - self._pmin)) #taking into account the angular width of the wedge

        dM = a * R * self._surfDens(R,ngl=ngl)
        return dM 

    #-----------------------------------------------------------------------------


    def _dMdz(self,z,ngl=20):

        if self._with_incompleteness:
            sys.exit("Error in SF_Wedge._dMdz(): "+
                     "Function not yet implemented to take care of imcompleteness.")
        
        if isinstance(z,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdz(zz,ngl=ngl) for zz in z])

        a = math.radians(numpy.fabs(self._pmax - self._pmin)) #taking into account the angular width of the wedge

        dM = a * (scipy.integrate.fixed_quad(lambda rr: rr * self._densfunc(rr,z,set_outside_zero=True,throw_error_outside=False,consider_incompleteness=False),
                                       self._Rmin,self._Rmax,n=ngl))[0]
        return dM 


    #-----------------------------------------------------------------------------

    def _surfDens(self,R,phi=None,ngl=20):

        if self._with_incompleteness:
            sys.exit("Error in SF_Wedge._surfDens(): "+
                     "Function not yet implemented to take care of imcompleteness.")

        if phi is None: sys.exit("Error in SF_Wedge._surfDens(): Specify phi.")
        
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

    def _dMdphi(self,phi,ngl=20):
        
        if isinstance(phi,numpy.ndarray):
            #Recursion if input is array:
            return numpy.array([self._dMdphi(pp,ngl=ngl) for pp in phi])

        if phi < self._pmin or phi > self._pmax:
            #out of bounds
            return 0.

        dM = (scipy.integrate.fixed_quad(lambda rr: rr * self._surfDens(rr,phi=phi,ngl=ngl),
                                         self._Rmin,self._Rmax,n=ngl))[0]
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
        dM = self._fastGLint_wedge(pvfunc,xgl,wgl)
        return dM

    #-------------------------------------------------------------------------------------

    """def _set_5D_GLgrid(self,nRs=20,nzs=20,ngl_vel=20,ngl_vT=20):
        
        #Gauss Legendre points & weights for vT integration:
        xgl_vT,wgl_vT   = numpy.polynomial.legendre.leggauss(ngl_vT)
        
        
        #_____split up integration in velocity in different intervals_____
        #...how many GL points in each integration interval?
        if (ngl_vel % 2) != 0: sys.exit("Error in SF_Wedge._set_5D_GLgrid(): ngl_vel has to be an even number.")
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

        #shape and dimension of the arrays:
        ngl_vz = ngl_vel
        ngl_vR = ngl_vel
        tupleshape      = (ngl_vz,ngl_vT,ngl_vR,nzs,nRs)
        dim = numpy.array([ngl_vz,ngl_vT,ngl_vR,nzs,nRs])
        
        # rectangular (R,z) grid:
        Rn = numpy.zeros(tupleshape)
        zm = numpy.zeros(tupleshape)

        # Gauss Legendre grid for velocities:
        vzk = numpy.zeros(tupleshape)
        vTj = numpy.zeros(tupleshape)
        vRi = numpy.zeros(tupleshape)

        # Gauss Legendre weights for velocity integration:
        Wk = numpy.zeros(tupleshape)
        Wj = numpy.zeros(tupleshape)
        Wi = numpy.zeros(tupleshape)
        
        #_____R_____
        #grid limits:
        Rmax = self._Rmax
        Rmin = self._Rmin
        Rs = numpy.linspace(Rmin,Rmax,nRs)    
        for nn in range(nRs):
            Rnt = Rs[nn]
            
            #_____z_____
            #integration limits:
            zmax = self._zmax
            zmin = self._zmin
            zs = numpy.linspace(zmin,zmax,nzs) 
            for mm in range(nzs):
                zmt = zs[mm]
                    
                #_____vz_____
                #integration limits (the fiducial df is only used to set the grid):
                ro = self._df_fid._ro
                sigmaZn = self._df_fid._sz * numpy.exp((ro-Rnt)/self._df_fid._hsz)
                vZmaxn = vmax_unscaled * sigmaZn
                vZminn = vmin_unscaled * sigmaZn
                for kk in range(ngl_vz):
                    #calculate GL points/weights:
                    Wkt  = 0.5*(vZmaxn[kk]-vZminn[kk])*wgl_v[kk]
                    vzkt = 0.5*(vZmaxn[kk]-vZminn[kk])*xgl_v[kk] + 0.5*(vZmaxn[kk]+vZminn[kk])

                    #_____vT_____
                    #integration limits:
                    vTmin = 0.
                    vTmax = 1.5
                    for jj in range(ngl_vT):
                        #calculate GL points/weights:
                        Wjt  = 0.5*(vTmax-vTmin)*wgl_vT[jj]
                        vTjt = 0.5*(vTmax-vTmin)*xgl_vT[jj] + 0.5*(vTmax+vTmin)
                    
                        #_____vR_____
                        #integration limits (the fiducial df is only used to set the grid):
                        sigmaRn = self._df_fid._sr * numpy.exp((ro-Rnt)/self._df_fid._hsr)
                        vRmaxn  = vmax_unscaled * sigmaRn
                        vRminn  = vmin_unscaled * sigmaRn
                        for ii in range(ngl_vR):
                            #calculate GL points/weights:
                            Wit  = 0.5*(vRmaxn[ii]-vRminn[ii])*wgl_v[ii]
                            vRit = 0.5*(vRmaxn[ii]-vRminn[ii])*xgl_v[ii] + 0.5*(vRmaxn[ii]+vRminn[ii])

                            Rn [ii,jj,kk,mm,nn] = Rnt
                            
                            zm [ii,jj,kk,mm,nn] = zmt
                            
                            vzk[ii,jj,kk,mm,nn] = vzkt
                            Wk [ii,jj,kk,mm,nn] = Wkt
                            
                            vTj[ii,jj,kk,mm,nn] = vTjt
                            Wj [ii,jj,kk,mm,nn] = Wjt
                            
                            vRi[ii,jj,kk,mm,nn] = vRit
                            Wi [ii,jj,kk,mm,nn] = Wit
                            
                            

        #_____store grid points_____
        self._R_gl5D = Rn
        self._z_gl5D = zm
        self._vR_gl5D = vRi
        self._vT_gl5D = vTj
        self._vz_gl5D = vzk

        #Gauss Legendre weights:
        self._weights_gl5D = Wi * Wj * Wk * Wm * Wndensity 

    #-------------------------------------------------------------------------------------

    def _add_phi_6D_GLgrid(self,ngl_phi):

        sys.exit("TO DO: This should be rewritten in the following way: It makes an GL grid in (R,z,phi) but uses the interpolated density and also calls the completeness function that still has to be implemented")

        xgl_phi,wgl_phi = numpy.polynomial.legendre.leggauss(ngl_phi)

        ts = numpy.shape(self._R_glGrid)
        tupleshape      = (ngl_phi,ts[0],ts[1],ts[2],ts[3],ts[4],ts[5])
        dim = numpy.array([ngl_phi,ts[0],ts[1],ts[2],ts[3],ts[4],ts[5]])

        #_____phi_____
        #integration limits:
        phimax = self._pmax
        phimin = self._pmin
        #calculate GL points/weights:
        Wlt   = 0.5*(phimax-phimin)*wgl_phi
        philt = 0.5*(phimax-phimin)*xgl_phi + 0.5*(phimaxmn+phiminmn)
        #correct shape:
        Wl   = numpy.resize(numpy.repeat(Wlt  ,numpy.prod(dim[1::])),tupleshape)
        phil = numpy.resize(numpy.repeat(philt,numpy.prod(dim[1::])),tupleshape)
        #store:
        self._phi_gl6D = phil

        #correct shape of all grids:
        self._R_gl6D  = numpy.resize(self._R_gl5D,tupleshape)
        self._z_gl6D  = numpy.resize(self._z_gl5D,tupleshape)
        self._vR_gl6D = numpy.resize(self._vR_gl5D,tupleshape)
        self._vT_gl6D = numpy.resize(self._vT_gl5D,tupleshape)
        self._vz_gl6D = numpy.resize(self._vz_gl5D,tupleshape)

        #Gauss Legendre weights:
        self._weights_gl6D = self._weights_gl5D * Wl

    #-------------------------------------------------------------------------------------

    def _densityGrid_from_5DGLgrid(self,_multi=None):
        
        #shape of the grid:
        shapetuple = numpy.shape(self._R_gl5D)

        #evaluate THIS df (not the fiducial) at all the actions on the grid:
        print "          evaluate df"

        if _multi is None or _multi == 1:
            #...evaluation on one core:

            qdf = self._df(
                        (
                            self._jr_gl5D.flatten(),
                            self._lz_gl5D.flatten(),
                            self._jz_gl5D.flatten()
                            ),
                        rg   =self._rg_gl5D.flatten(),
                        kappa=self._kappa_gl5D.flatten(),
                        nu   =self._nu_gl5D.flatten(),
                        Omega=self._Omega_gl5D.flatten()
                        )    
 
        elif _multi > 1:
            #...evaluation on multiple cores:

            # The following works only, because          
            # the selection function is independent of velocities.
            # i.e.: at one given point (vR,vz,vT) the corresponding (R,z) 
            # grid is always the same as at any other velocity point by 
            # construction in _set_5D_GLgrid().

            nRs  = shapetuple[4]
            nzs  = shapetuple[3]
            ngl_vR = shapetuple[2]
            ngl_vT = shapetuple[1]
            ngl_vz = shapetuple[0]

            # The actions for each (R,z) are calculated separately:
            #    ii = index in R
            #    kk = index in z
            R_index = range(nRs)
            z_index = range(nzs)
            ii,kk = numpy.meshgrid(R_index,z_index,indexing='ij')
            # Note: here the indexing of the meshgrid ('ij') is very important, 
            # as we want to reshape the result later back to the input shape
            ii = ii.flatten()
            kk = kk.flatten()

            multiOut =  multi.parallel_map(
                    (lambda x: self._df(
                        (
                            numpy.ravel(self._jr_gl5D[:,:,:,kk[x],ii[x]]),
                            numpy.ravel(self._lz_gl5D[:,:,:,kk[x],ii[x]]),
                            numpy.ravel(self._jz_gl5D[:,:,:,kk[x],ii[x]]),
                            ),
                        rg   =numpy.ravel(self._rg_gl5D   [:,:,:,kk[x],ii[x]]),
                        kappa=numpy.ravel(self._kappa_gl5D[:,:,:,kk[x],ii[x]]),
                        nu   =numpy.ravel(self._nu_gl5D   [:,:,:,kk[x],ii[x]]),
                        Omega=numpy.ravel(self._Omega_gl5D[:,:,:,kk[x],ii[x]])
                        )
                    ),
                    range(nRs*nzs),
                    numcores=numpy.amin([
                                nrs*nzs,
                                multiprocessing.cpu_count(),
                                _multi
                                ])
                    )

            qdf = numpy.zeros((ngl_vz*ngl_vT*ngl_vR,nzs,nRs))
            for x in range(nrs*nzs):
                qdf[:,kk[x],ii[x]] = multiOut[x]

        #back to original shape:
        self._dfgrid_gl5D = numpy.reshape(qdf,shapetuple)

        #calculate density at each grid point by summing over velocities:
        print "          sum over velocities"
        dens = numpy.sum(self._weights_gl5D * self._dfgrid_gl5D, axis=(0,1,2))

        #assign the grid to normal densfunc grid:
        self._densgrid = numpy.transpose(dens) #(transposed such that the first axis is R and the second axis is z)
        self._densgrid_R = self._R_gl6D[0,0,0,0,:]
        self._densgrid_z = self._z_gl6D[0,0,0,:,0]

        # set up interpolation object:
        # this intepolate the log of the density ~linear
        self._densInterp= scipy.interpolate.RectBivariateSpline(
                                        self._densgrid_R,self._densgrid_z,
                                        numpy.log(self._densgrid),  
                                        kx=3,ky=3,
                                        s=0.
                                        )
        

    #---------------------------------------------

    def _density_from_GLgrid_test(self):

        #shape of the grid:
        shapetuple = numpy.shape(self._R_gl5D)

        ngl_R  = shapetuple[4]
        ngl_z  = shapetuple[3]
        ngl_vR = shapetuple[2]
        ngl_vT = shapetuple[1]
        ngl_vz = shapetuple[0]
    
        for ii in range(ngl_R):
            for kk in range(ngl_z):
                print 'R = ',self._R_gl5D[0,0,0,kk,ii],
                print ', z = ', self._z_gl5D[0,0,0,kk,ii],
                dens = numpy.sum(self._weights_gl5D_vel[:,:,:,kk,ii] * self._dfgrid_gl5D[:,:,:,kk,ii])
                print ', qdf = ',dens

        return self._R_gl5D[0,0,0,:,:].flatten(),self._z_gl5D[0,0,0,:,:].flatten()"""


