#_____import packages_____
import math
import multiprocessing
import numpy
import os
import pickle
import time
import sys
from galpy.df import quasiisothermaldf
from galpy.util import multi
from precalc_actions import precalc_pot_actions_sf

#---------------------------------------------------------------------

#____load shared memory_____
data_shared = None
info_MCMC = None
#read current path
current_path_filename = "temp/path_of_current_analysis.sav"
if os.path.exists(current_path_filename):
    savefile    = open(current_path_filename,'rb')
    current_path = pickle.load(savefile)
    savefile.close()
    

    #load shared data
    shared_data_filename = current_path+'_shared_data.npy'
    if os.path.exists(shared_data_filename):
        data_shared = numpy.load(shared_data_filename)

    #load MCMC info:
    info_MCMC_filename   = current_path+'_info_MCMC.sav'
    if os.path.exists(info_MCMC_filename):
        savefile    = open(info_MCMC_filename,'rb')
        info_MCMC   = pickle.load(savefile)
        savefile.close()

#---------------------------------------------------------------------

def logprob_MCMC(
            p,
            def_param=(data_shared,info_MCMC)
            ):

    """
    NAME:
        logprob_MCMC
    PURPOSE:
    INPUT:
        p - (float numpy array)
          - current walker position in potential and qdf parameter space [potPar,dfPar]
    OUTPUT:
    HISTORY:
    """
    #zeit_start = time.time()

    #_____Reference scales_____
    _REFR0 = 8.                 #[kpc]
    _REFV0 = 220.               #[km/s]

    #_____read additional MCMC information_____
    chainfilename   = info_MCMC['chainfilename']
    potParEst_phys  = numpy.array(  info_MCMC['potParEst_phys'],dtype='float64')
    dfParEst_fit    = numpy.array(  info_MCMC['dfParEst_fit'],dtype='float64')
    dfParFid_fit    = numpy.array(  info_MCMC['dfParFid_fit'],dtype='float64')
    sfParEst_phys   = numpy.array(  info_MCMC['sfParEst_phys'],dtype='float64')
    potParFitBool   = numpy.array(  info_MCMC['potParFitBool'],dtype='bool')
    dfParFitBool    = numpy.array(  info_MCMC['dfParFitBool'],dtype='bool')
    ro_known        = float(        info_MCMC['ro_known'])
    _N_SPATIAL_R    = int(          info_MCMC['_N_SPATIAL_R'])
    _N_SPATIAL_Z    = int(          info_MCMC['_N_SPATIAL_Z'])
    _NGL_VELOCITY   = int(          info_MCMC['_NGL_VELOCITY'])
    _N_SIGMA        = float(        info_MCMC['_N_SIGMA'])
    _VT_GALPY_MAX   = float(        info_MCMC['_VT_GALPY_MAX'])                      
    _XGL            = numpy.array(  info_MCMC['_XGL'],dtype='float64')
    _WGL            = numpy.array(  info_MCMC['_WGL'],dtype='float64')
    datatype        = int(          info_MCMC['datatype'])
    pottype         = int(          info_MCMC['pottype'])
    sftype          = int(          info_MCMC['sftype'])
    noStars         = int(          info_MCMC['noStars'])
    marginal_coord  = int(          info_MCMC['marginal_coord'])
    xgl_marginal    = numpy.array(  info_MCMC['xgl_marginal'],dtype='float64')
    wgl_marginal    = numpy.array(  info_MCMC['wgl_marginal'],dtype='float64')
    _N_ERROR_SAMPLES= int(          info_MCMC['_N_ERROR_SAMPLES'])
    in_sf_data      = numpy.array(  info_MCMC['in_sf_data'],dtype='float64')
    MCMC_use_fidDF  = bool(         info_MCMC['MCMC_use_fidDF'])

    #_____separate coordinates into free potential and free df coordinates_____
    npotpar = numpy.sum(potParFitBool)
    potPar_phys_MCMC = p[0:npotpar]    #physical units
    dfPar_fit_MCMC   = p[npotpar::]    #logarithmic fit units, not yet galpy scaled


    #_____p_Phi parameters_____
    #load estimates for fixed potential parameters
    potPar_phys = potParEst_phys
    #overwrite the free parameters with current parameter set
    potPar_phys[potParFitBool] = potPar_phys_MCMC
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0

    #_____p_DF parameters in galpy units_____
    #load estimates for fixed DF parameters
    dfPar_fit               = dfParEst_fit
    #overwrite the free parameters with current parameter set
    dfPar_fit[dfParFitBool] = dfPar_fit_MCMC
    traf                    = numpy.array([ro,vo,vo,ro,ro])
    dfPar_galpy             = numpy.exp(dfPar_fit) / traf

    #_____rescale data to galpy units_____
    ndata = noStars*_N_ERROR_SAMPLES
    R_galpy   = data_shared[0,:]/ro # R_data/ro
    vR_galpy  = data_shared[1,:]/vo #vR_data/vo
    vT_galpy  = data_shared[2,:]/vo #vT_data/vo
    z_galpy   = data_shared[3,:]/ro # z_data/ro
    vz_galpy  = data_shared[4,:]/vo #vz_data/vo
    weights_marginal = None
    if datatype == 3: #perfect mock data, marginalization over one coordinate
        if marginal_coord == 4: #marginalize over vT
            vTmin = 0.  #integration limits
            vTmax = _VT_GALPY_MAX
            weights_marginal = 0.5 * (vTmax - vTmin) * wgl_marginal
            vT_marginal = 0.5 * (vTmax - vTmin) * xgl_marginal + 0.5 * (vTmax + vTmin)
            vT_galpy = numpy.tile(vT_marginal,noStars)
        else:
            sys.exit("Error in logprob_MCMC(): The "+\
                     'marginalization over "'+marginal_coord+'" is not yet '+
                     "implemented. So far only the marginalization over 'vT' is "+
                     "implemented.")

    #_____calculate actions: of data and for density calculation in SF_____
    if MCMC_use_fidDF:
        # the integration range for the density is set by the fiducial QDF!
        pot,aA,sf,actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,
                        potPar_phys,
                        dfParFid_fit, # <-- !!!
                        sfParEst_phys,
                        R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                        ro_known,
                        _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,None)
    else:
        # the integration range for the density is set by the current QDF!
        pot,aA,sf,actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,
                        potPar_phys,
                        dfPar_fit, # <-- !!!
                        sfParEst_phys,
                        R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                        ro_known,
                        _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,None)

    if not pot_physical:
        #_____if potential is unphysical, return log(0)_____
        loglike = -numpy.inf
    else:

        jr_data    = actions[0,:]
        lz_data    = actions[1,:]
        jz_data    = actions[2,:]
        rg_data    = actions[3,:]
        kappa_data = actions[4,:]
        nu_data    = actions[5,:]
        Omega_data = actions[6,:]

        #_____calculate likelihood_____
        loglike = loglikelihood_dfPar(
                                pot,
                                aA,
                                sf,
                                dfPar_galpy[0], #hr
                                dfPar_galpy[1], #sr
                                dfPar_galpy[2], #sz
                                dfPar_galpy[3], #hsr
                                dfPar_galpy[4], #hsz
                                ro,vo,
                                jr_data,
                                lz_data,
                                jz_data,
                                rg_data,
                                kappa_data,
                                nu_data,
                                Omega_data,
                                _XGL,_WGL,
                                datatype,noStars,
                                marginal_coord=marginal_coord,
                                weights_marginal=weights_marginal,
                                _N_ERROR_SAMPLES=_N_ERROR_SAMPLES,
                                in_sf_data=in_sf_data
                                )

    if numpy.isnan(loglike):
        loglike = -numpy.inf
    elif loglike < -1.7e+308:
        loglike = -numpy.inf

    #_____priors_____
    #potential parameters:
    #       flat prior
    #       zero outside the given grid limits and in unphysical regions (see above)
    #df parameters:
    #       logarithmically flat priors
    #       (because the fit parameters are actually log(df parameters))
    logprior = 0.

    #_____measure time_____
    #zeit_t = time.time() - zeit_start
    #if zeit_t > 120.: 
    #    print p
    #    print round(zeit_t,2)," sec"

    #_____print current walker position to file_____
    if not chainfilename is None:
        f = open(chainfilename, "a")
        for k in range(len(p)):
            f.write("%3.5f " % (p[k]))
        f.write("%3.5f \n" % (loglike + logprior))
        f.close()

    return loglike + logprior


#------------------------------------------------------------------------


def loglikelihood_potPar(pot,aA,sf,
                         _N_SPATIAL_R,_N_SPATIAL_Z,
                         ro,vo,
                         jr_data,lz_data,jz_data,
                         rg_data,
                         kappa_data,nu_data,Omega_data,
                         dfParArr_fit,
                         dfParFid_galpy,
                         _NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_XGL,_WGL,_MULTI,
                         datatype,noStars,
                         marginal_coord=None,weights_marginal=None,
                         _N_ERROR_SAMPLES=None,in_sf_data=None):

    """
    NAME:
        loglikelihood_potPar
    PURPOSE:
    INPUT:
        pot                 - (Potential object)
                            - Potential object, corresponding to the 
                              current potential of which we want to 
                              calculate the likelihhod

        aA                  - (ActionAngle object)
                            - ActionAngle object, corresponding to the 
                              current potential

        sf                  - (SelectionFunction object) 
                            - Selection function object, corresponding 
                              to the mock data selection function

        _N_SPATIAL_R, N_SPATIAL_Z  - (int,int) 
                            - number of grid points in R and z on which 
                              the density for a given DF is calculated 
                              and then interpolated

        ro, vo              - (float,float) 
                            - normalisation of scales and velocities to 
                              galpy units with respect to _REFR0 and 
                              _REFV0

        jr_data,lz_data,jz_data,rg_data,kappa_data,nu_data,Omega_data 
                            - (float array, float array, float array, etc.) 
                            - actions, guiding star radii and frequencies
                              of the data


        lnhrs,lnszs,lnhszs  - (float array, float array, float array) 
                            - grid of p_DF parameters ln(h_r), 
                              ln(sigma_z), ln(h_sigma_z) to investigate 
                              the likelihood

        dfParFid_galpy      - (float array) 
                            - parameters of the fiducial qdf in galpy units

    OUTPUT:
    HISTORY:
    """

    #_____initialize likelihood grid_____
    ndfs = len(dfParArr_fit[:,0])  #number of qdfs to test
    loglike_out = numpy.zeros(ndfs)

    #_____setup fiducial qdf and calculate actions on velocity grid_____
    # velocity grid that corresponds to the sigmas of the fiducial qdf.
    #print "   * calculate fiducial actions"   
    qdf_fid = quasiisothermaldf(
                dfParFid_galpy[0],
                dfParFid_galpy[1],
                dfParFid_galpy[2],
                dfParFid_galpy[3],
                dfParFid_galpy[4],
                pot=pot,aA=aA,
                cutcounter=True, 
                ro=ro
                )
    # (*Note:* if cutcounter=True, set counter-rotating stars' 
    #          DF to zero)
    sf.set_fiducial_df_actions_Bovy(
               qdf_fid,
               nrs=_N_SPATIAL_R,nzs=_N_SPATIAL_Z,
               ngl_vel=_NGL_VELOCITY,
               n_sigma=_N_SIGMA,
               vT_galpy_max=_VT_GALPY_MAX,
               _multi=_MULTI
               )
    """#setup grid in (R,z,vR,vT,vz) for integration over velocity at (R,z):
    sf.setup_velocity_integration_grid(
                qdf_fid,
                nrs=_N_SPATIAL_R,nzs=_N_SPATIAL_Z,
                ngl_vel=_NGL_VELOCITY,
                n_sigma=_N_SIGMA,
                vT_galpy_max=_VT_GALPY_MAX
                )
    #calculate actions at each grid point on the grid above:
    sf.calculate_actions_on_vig_using_fid_pot(qdf_fid,_multi=_MULTI)"""

    #_____p_DF parameters in galpy units_____
    traf = numpy.array([ro,vo,vo,ro,ro])
    dfParArr_galpy = numpy.exp(dfParArr_fit) / traf

    #_____start iteration through distribution functions_____
    #print "   * iterate through qDFs"   
    if _MULTI is None or _MULTI == 1:
        for jj in range(ndfs):

            #_____likelihood_____
            loglike_out[jj] = loglikelihood_dfPar(
                                        pot,aA,sf,
                                        dfParArr_galpy[jj,0], #hr
                                        dfParArr_galpy[jj,1], #sr
                                        dfParArr_galpy[jj,2], #sz
                                        dfParArr_galpy[jj,3], #hsr
                                        dfParArr_galpy[jj,4], #hsz
                                        ro,vo,
                                        jr_data,lz_data,jz_data,
                                        rg_data,
                                        kappa_data,nu_data,Omega_data,
                                        _XGL,_WGL,
                                        datatype,noStars,
                                        marginal_coord=marginal_coord,
                                        weights_marginal=weights_marginal,
                                        _N_ERROR_SAMPLES=_N_ERROR_SAMPLES,
                                        in_sf_data=in_sf_data
                                        )

    elif _MULTI > 1:
        
        #....iterate over all sets of qdf parameters
        #    and calculate each on a separate core:
        loglike_out = multi.parallel_map(
                            (lambda x: loglikelihood_dfPar(
                                            pot,aA,sf,
                                            dfParArr_galpy[x,0], #hr
                                            dfParArr_galpy[x,1], #sr
                                            dfParArr_galpy[x,2], #sz
                                            dfParArr_galpy[x,3], #hsr
                                            dfParArr_galpy[x,4], #hsz
                                            ro,vo,
                                            jr_data,lz_data,jz_data,
                                            rg_data,
                                            kappa_data,nu_data,Omega_data,
                                            _XGL,_WGL,
                                            datatype,noStars,
                                            marginal_coord=marginal_coord,
                                            weights_marginal=weights_marginal,
                                            _N_ERROR_SAMPLES=_N_ERROR_SAMPLES,
                                            in_sf_data=in_sf_data
                                            )),
                            range(ndfs),
                            numcores=numpy.amin([
                                               ndfs,
                                               multiprocessing.cpu_count(),
                                               _MULTI
                                               ])
                            )
              
    return loglike_out

#-------------------------------------------------------------------

def loglikelihood_dfPar(pot,aA,sf,
                        hr,sr,sz,hsr,hsz,
                        ro,vo,
                        jr_data,lz_data,jz_data,
                        rg_data,
                        kappa_data,nu_data,Omega_data,
                        _XGL,_WGL,
                        datatype,noStars,
                        marginal_coord=None,weights_marginal=None,
                        _N_ERROR_SAMPLES=None,in_sf_data=None,use_outlier_model=True):
    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2015-??-?? - Written - Trick (MPIA)
            2015-12-27 - Added simple outlier model. - Trick (MPIA)
            2016-02-16 - Made outlier model optional. 
        """
    
    #_____initialize qdf_____
    # set parameters of distribution function:
    qdf = quasiisothermaldf(
                            hr,sr,sz,hsr,hsz,   
                            pot=pot,aA=aA,
                            cutcounter=True,
                            ro=ro)
    # (*Note:* if cutcounter=True, set counter-rotating stars' 
    #          DF to zero)

    #_____integrate density over effective volume_____
    # set distribution function in selection function:
    sf.reset_df(qdf)
    # setup density grid for interpolating the density 
    # (*Note 1:* we do not use evaluation on multiple cores here, 
    #            because we rather evaluate each qdf on a separate core)
    # (*Note 2:* The order of GL integration is set by ngl when setting 
    #            up the fiducial qdf.)
    sf.densityGrid(_multi=None)
    # integrate density over effective volume, i.e. the selection 
    # function. Use a fast GL integration analogous to Bovy:
    dens_norm = sf.Mtot(xgl=_XGL,wgl=_WGL)
    
    
    #_____calculate likelihood for given data type_____
    if datatype == 1 or datatype == 4:   
        #1: perfect mock data
        #4: perfect mock data (mix of 2 sets)

        #_____calculate loglikelihood_____
        # log(likelihood) of all data points given p_DF and p_Phi:
        lnL_i = qdf(
                    (jr_data,lz_data,jz_data),
                    rg   =rg_data,
                    kappa=kappa_data,
                    nu   =nu_data,
                    Omega=Omega_data,
                    log  =True
                    )

        #_____normalize likelihood_____
        # for current p_DF and p_Phi:
        lnL_i -= math.log(dens_norm)

        #_____units of the likelihood_____
        # [L_i dx^3 dv^3] = 1 --> [L_i] = [xv]^{-3}
        logunits = 3. * numpy.log(ro*vo)
        lnL_i -= logunits

    elif datatype == 2:
        #2: mock data with measurement errors

        #_____calculate loglikelihood for each error sample_____
        # (*Note:* the likelihood of each of errors samples around the 
        #          observed data points. Which are, by the way, not in the 
        #          list anymore. Only Gaussian samples around them.)

        # log(likelihood) of all data points given p_DF and p_Phi:
        lnL_err = qdf(
                    (jr_data,lz_data,jz_data),
                    rg   =rg_data,
                    kappa=kappa_data,
                    nu   =nu_data,
                    Omega=Omega_data,
                    log=True
                    )

        #_____normalize likelihood_____
        # for current p_DF and p_Phi:
        lnL_err -= math.log(dens_norm)

        #_____units of the likelihood_____
        # [L_i dx^3 dv^3] = 1 --> [L_i] = [xv]^{-3}
        logunits = 3. * numpy.log(ro*vo)
        lnL_err -= logunits 

        #_____calculate likelihoods belonging to one real (!) data point____
        # by taking the mean for each data point we sum up the likelihoods, 
        # not the loglikelihoods:
        L_err = numpy.exp(lnL_err)
        # reshaping such that the error samples belonging to one real 
        # observed data point are again in the same row:
        L_err       = numpy.reshape(L_err     ,(_N_ERROR_SAMPLES,noStars))
        in_sf_data  = numpy.reshape(in_sf_data,(_N_ERROR_SAMPLES,noStars))
        # the convolution integral of the likelihood with the error gaussian
        # is calculated in the following way:
        #       int L(x|p) * N[x_i,e_x](x) dx'
        #               ~ 1/M * sum_j=1^M qdf(x_j) * sf(x_j) / norm
        # where x_j sample N[x_i,e_x](x_j), 
        # and M = _NERRSAMPLE,
        # and L(x|p) = qdf(x) * sf(x) / norm = L_err * insf_data
        # and in_sf_data = sf(x) is 0/1 when outside/inside of observed volume.
        # (*Note:* axis=0 means summing only over rows.)
        L_i = numpy.sum(L_err * in_sf_data,axis=0) / float(_N_ERROR_SAMPLES)
        # back to log likelihood:
        lnL_i = numpy.log(L_i)
        if len(lnL_i) != noStars: 
            sys.exit("Error in loglikelihood_dfPar: "+\
                     "calculating mean likelihood does not give "+\
                     "right number of elements.")


    elif datatype == 3:
        #3: perfect mock data, marginalization over one coordinate

        #_____calculate loglikelihood_____
        # log(likelihood) of all data points given p_DF and p_Phi:
        # This is the log(likelihood) at each data point, including the data points 
        # used to marginalize over one of the velocities
        lnL_j_all = qdf(
                    (jr_data,lz_data,jz_data),
                    rg   =rg_data,
                    kappa=kappa_data,
                    nu   =nu_data,
                    Omega=Omega_data,
                    log  =True
                    )

        #_____normalize likelihood_____
        # for current p_DF and p_Phi:
        lnL_j_all -= math.log(dens_norm)

        #_____marginalize over one of the coordinates_____
        if len(weights_marginal.shape) > 1:
            sys.exit("Error in loglikelihood_dfPar(): "+\
                     "marginalization over coordinates is only "+
                     "implemented for vT so far.")
        ngl_marginal = len(weights_marginal)
        # reshape the likelihood such that in each row (e.g. L_j[0,:])
        # we have all the ngl_marginal points belonging to one original data point:
        L_j_all = numpy.reshape(numpy.exp(lnL_j_all),(noStars,ngl_marginal))
        # now apply the Gauss-Legendre integration for each real data point:
        L_i = numpy.sum(L_j_all*weights_marginal,axis=1)
        #back to log(likelihood):
        lnL_i = numpy.log(L_i)

        #_____units of the likelihood_____
        if marginal_coord == 4 or marginal_coord == 2 or marginal_coord == 6:
            #marginalization over one velocity component:            
            # [L_i dx^3 dv^2] = 1 --> [L_i] = [x^{-3} v^{-2}]
            logunits = 3. * numpy.log(ro) + 2. * numpy.log(vo)
            lnL_i -= logunits 
        else: 
            sys.exit("ERROR in loglikelihood_dfPar(): "+\
                 "marginal_coord "+str(marginal_coord)+" is not defined.")  
    else:
        sys.exit("ERROR in loglikelihood_dfPar(): "+\
                 "data type "+str(datatype)+" is not defined.")    


    if use_outlier_model:
        #_____simple outlier model_____
        #all likelihoods have to be bigger than $\epsilon \cdot \bar{\mathscr{L}}$
        L_i      = numpy.exp(lnL_i)
        median_L = numpy.median(L_i)
        epsilon  = 0.001 # = 0.1 %
        L_i      = numpy.maximum(L_i,epsilon*median_L)
        lnL_i    = numpy.log(L_i)

    #_____sum logL for final loglikelihood_____
    # sum up contributions for all data points: 
    loglike_out = numpy.sum(lnL_i)  

    return loglike_out


