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
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object, setup_SelectionFunction_object
from prior import calculate_logprior
from outlier_model import calculate_outlier_model, scale_df_fit_to_galpy, scale_df_galpy_to_phys

#---------------------------------------------------------------------

#____load shared memory_____
data_shared               = None
info_MCMC                 = None
data_actions_shared       = None
fiducial_actions_shared   = None
incomp_shared             = None

#read current path
current_path_filename = "temp/path_of_current_analysis.sav"
if os.path.exists(current_path_filename):
    savefile    = open(current_path_filename,'rb')
    current_path = pickle.load(savefile)
    savefile.close()
    

    #load shared data:
    shared_data_filename = current_path+'_shared_data.npy'
    if os.path.exists(shared_data_filename):
        data_shared = numpy.load(shared_data_filename)

    #load MCMC info:
    info_MCMC_filename   = current_path+'_info_MCMC.sav'
    if os.path.exists(info_MCMC_filename):
        savefile    = open(info_MCMC_filename,'rb')
        info_MCMC   = pickle.load(savefile)
        savefile.close()

    #special case: DF fit only. --> load shared data action data:
    shared_action_data_filename = current_path+'_shared_data_actions.npy'
    if os.path.exists(shared_action_data_filename):
        data_actions_shared = numpy.load(shared_action_data_filename)

    #special case: DF fit only. --> load shared fiducial action data:
    shared_action_fiducial_filename = current_path+'_shared_fiducial_actions.npy'
    if os.path.exists(shared_action_fiducial_filename):
        fiducial_actions_shared = numpy.load(shared_action_fiducial_filename)

    #special case: sftype = 4, SF_incompleteShell
    shared_data_SF_incompleteShell_filename = current_path+'_shared_SF_incompleteShell.npy'
    if os.path.exists(shared_data_SF_incompleteShell_filename):
        incomp_shared = numpy.load(shared_data_SF_incompleteShell_filename)

#---------------------------------------------------------------------

def logprob_MCMC(
            p,
            def_param=(data_shared,info_MCMC,incomp_shared)
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
        2016-04-15 - Added the parameters governing the actionAngle Delta and accuracy to precalc_pot_actions_sf().
        2016-09-25 - Added shared data parameter incomp_shared, that is used in precalc_pot_actions_sf(). - Trick (MPIA)
        2016-12-27 - Now makes use of function that calculates for a given priortype a non-flat prior for the model parameters. - Trick (MPIA)
        2017-01-02 - Now uses different dftypes with flexible number of parameters. Removed in_sf_data. - Trick (MPIA)
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
    dftype          = int(          info_MCMC['dftype'])
    priortype       = int(          info_MCMC['priortype'])
    noStars         = int(          info_MCMC['noStars'])
    norm_outlier    = float(        info_MCMC['norm_outlier'])
    marginal_coord  = int(          info_MCMC['marginal_coord'])
    xgl_marginal    = numpy.array(  info_MCMC['xgl_marginal'],dtype='float64')
    wgl_marginal    = numpy.array(  info_MCMC['wgl_marginal'],dtype='float64')
    _N_ERROR_SAMPLES= int(          info_MCMC['_N_ERROR_SAMPLES'])
    MCMC_use_fidDF  = bool(         info_MCMC['MCMC_use_fidDF'])
    aASG_accuracy   = numpy.array(  info_MCMC['aASG_accuracy'],dtype='float64')
    use_default_Delta = bool(       info_MCMC['use_default_Delta'])
    estimate_Delta  = bool(         info_MCMC['estimate_Delta'])
    Delta_fixed     = float(        info_MCMC['Delta_fixed'])

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
    dfPar_galpy = scale_df_fit_to_galpy(dftype,ro,vo,dfPar_fit)

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

    #_____prepare data for outlier model_____
    data_for_outlier_model_galpy = None
    if dftype == 12:
        data_for_outlier_model_galpy = numpy.zeros((4,len(R_galpy)))
        data_for_outlier_model_galpy[0,:] = R_galpy
        data_for_outlier_model_galpy[1,:] = vR_galpy
        data_for_outlier_model_galpy[2,:] = vT_galpy
        data_for_outlier_model_galpy[3,:] = vz_galpy

    #_____calculate actions: of data and for density calculation in SF_____
    if MCMC_use_fidDF:
        # the integration range for the density is set by the fiducial QDF!
        pot,aA,sf,actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,dftype,
                        potPar_phys,
                        dfParFid_fit, # <-- !!!
                        sfParEst_phys,
                        R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                        ro_known,
                        _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,
                        None,   #<-- _MULTI
                        aASG_accuracy,use_default_Delta,estimate_Delta,Delta_fixed,
                        incomp_shared=incomp_shared)
    else:
        # the integration range for the density is set by the current QDF!
        pot,aA,sf,actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,dftype,
                        potPar_phys,
                        dfPar_fit, # <-- !!!
                        sfParEst_phys,
                        R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                        ro_known,
                        _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,
                        None,   #<-- _MULTI
                        aASG_accuracy,use_default_Delta,estimate_Delta,Delta_fixed,
                        incomp_shared=incomp_shared)

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
                                dftype,
                                dfPar_galpy,
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
                                data_for_outlier_model_galpy=data_for_outlier_model_galpy,
                                normalisation_for_outlier_model=norm_outlier,
                                marginal_coord=marginal_coord,
                                weights_marginal=weights_marginal,
                                _N_ERROR_SAMPLES=_N_ERROR_SAMPLES
                                )

    if numpy.isnan(loglike):
        loglike = -numpy.inf
    elif loglike < -1.7e+308:
        loglike = -numpy.inf

    #_____priors on the model parameters_____
    #priortype = 0: flat priors in potential parameters and 
    #               logarithmically flat  priors in DF parameters
    #priortype = 1: additionally: prior on flat rotation curve               
    logprior = calculate_logprior(priortype,pottype,potPar_phys,pot_physical,pot=pot)

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

def logprob_MCMC_fitDF_only(
            p,
            def_param=(data_actions_shared,fiducial_actions_shared,info_MCMC,data_shared)
            ):

    """
    NAME:
        logprob_MCMC_fitDF_only
    PURPOSE:
    INPUT:
        p - (float numpy array)
          - current walker position in potential and qdf parameter space [potPar,dfPar]
    OUTPUT:
    HISTORY:
        16-02-18 - Written (based on logprob_MCMC()). - Trick (MPIA)
        16-04-15 - Added the parameter aAS_Delta to setup_potential_and_ActionAngle_object().
        16-12-27 - Now makes use of function that calculates for a given priortype a non-flat prior for the model parameters. - Trick (MPIA)
        2017-01-02 - Now uses different dftypes with flexible number of parameters. Removed in_sf_data. - Trick (MPIA)
    """

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
    pottype         = int(          info_MCMC['pottype'])   #this might be a slim version of the pottype only
    sftype          = int(          info_MCMC['sftype'])
    dftype          = int(          info_MCMC['dftype'])
    priortype       = int(          info_MCMC['priortype'])
    noStars         = int(          info_MCMC['noStars'])
    norm_outlier    = float(        info_MCMC['norm_outlier'])
    marginal_coord  = int(          info_MCMC['marginal_coord'])
    xgl_marginal    = numpy.array(  info_MCMC['xgl_marginal'],dtype='float64')
    wgl_marginal    = numpy.array(  info_MCMC['wgl_marginal'],dtype='float64')
    _N_ERROR_SAMPLES= int(          info_MCMC['_N_ERROR_SAMPLES'])
    MCMC_use_fidDF  = bool(         info_MCMC['MCMC_use_fidDF'])
    aAS_Delta_DFonly = float(       info_MCMC['aAS_Delta_DFfitonly'])

    #_____separate coordinates into free potential and free df coordinates_____
    npotpar = numpy.sum(potParFitBool)
    if npotpar > 0:
        sys.exit("Error in logprob_MCMC_fitDF_only(): "+\
                 "This MCMC likelihood function should be used only "+\
                 "if the potential is NOT fitted. However, some "+\
                 "potential parameter seem to be fit parameters.")
    dfPar_fit_MCMC   = p    #logarithmic fit units, not yet galpy scaled


    #_____p_Phi parameters_____
    #load estimates for fixed potential parameters
    potPar_phys = potParEst_phys
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0

    #_____p_DF parameters in galpy units_____
    #load estimates for fixed DF parameters
    dfPar_fit               = dfParEst_fit
    #overwrite the free parameters with current parameter set
    dfPar_fit[dfParFitBool] = dfPar_fit_MCMC
    dfPar_galpy             = scale_df_fit_to_galpy(dftype,ro,vo,dfPar_fit)
    #fiducial DF parameters in galpy units:
    dfParFid_galpy          = scale_df_fit_to_galpy(dftype,ro,vo,dfParFid_fit)

    #_____load precalculated actions_____
    #data actions:
    jr_data    = data_actions_shared[0,:]
    lz_data    = data_actions_shared[1,:]
    jz_data    = data_actions_shared[2,:]
    rg_data    = data_actions_shared[3,:]
    kappa_data = data_actions_shared[4,:]
    nu_data    = data_actions_shared[5,:]
    Omega_data = data_actions_shared[6,:]
    #fiducial actions for normalization:
    jr_fiducial    = fiducial_actions_shared[0,:,:,:]
    lz_fiducial    = fiducial_actions_shared[1,:,:,:]
    jz_fiducial    = fiducial_actions_shared[2,:,:,:]
    rg_fiducial    = fiducial_actions_shared[3,:,:,:]
    kappa_fiducial = fiducial_actions_shared[4,:,:,:]
    nu_fiducial    = fiducial_actions_shared[5,:,:,:]
    Omega_fiducial = fiducial_actions_shared[6,:,:,:]

    #_____marginalization over a coordinate?_____
    weights_marginal = None
    if datatype == 3: #perfect mock data, marginalization over one coordinate
        sys.exit("Error in logprob_MCM_fitDF_only(): So far the "+\
                 "marginalization for this special case is not yet "+\
                 "implemented.")

    #_____prepare data for outlier model_____
    data_for_outlier_model_galpy = None
    if dftype == 12:
        data_for_outlier_model_galpy = numpy.zeros((4,len(jr_data)))
        data_for_outlier_model_galpy[0,:] = data_shared[0,:]/ro # R_data/ro
        data_for_outlier_model_galpy[1,:] = data_shared[1,:]/vo #vR_data/vo
        data_for_outlier_model_galpy[2,:] = data_shared[2,:]/vo #vT_data/vo
        data_for_outlier_model_galpy[3,:] = data_shared[4,:]/vo #vz_data/vo
    

    #_____setup potential and actionAngle object_____
    #Attention: In case the actual potential uses a StaeckelFudgeGrid
    #           to interpolate the actions, we do not set it up here 
    #           (because it is too expensive) but rather the slim version
    #           with StaeckelFudge. The pottype was changed in 
    #           analyze_mockdata_RoadMapping.py after setting up the shared data.    
    try:
        pot, aA = setup_Potential_and_ActionAngle_object(
                        pottype,
                        potPar_phys,
                        #Staeckel Delta:
                        aAS_Delta=aAS_Delta_DFonly
                        )
        pot_physical = True
    except RuntimeError as e:
        pot_physical = False
        loglike = -numpy.inf

    if pot_physical:

        #_____initialize selection function_____
        if ro_known:
            sf = setup_SelectionFunction_object(
                            sftype,
                            sfParEst_phys,
                            ro
                            )
        else:
            sys.exit("Error in logprob_MCMC_fitDF_only(): "+\
                     "How to deal with the selection function if ro is not known? To do: Implement.")

        #_____setup fiducial qdf and calculate actions on velocity grid_____
        # fiducial qDF is only needed to re-derive the integration limits
        # over vR and vz. (The actions at the grid points in between are
        # already precalculated, but for the pre-factor of the Gauss-Legendre
        # integration we need the integration limits again.)
        if dftype in [0,11,12]:
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
        else:
            sys.exit("Error in logprob_MCMC_fitDF_only(): dftype = "+\
                     str(dftype)+" is not defined.")

        # Now use the precalculated fiducial actions to set up the density
        # interpolation grid:
        sf.set_fiducial_df_actions_Bovy_from_precalculated_actions(
                    qdf_fid,
                    _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,
                    jr_fiducial,lz_fiducial,jz_fiducial,rg_fiducial,
                    kappa_fiducial,nu_fiducial,Omega_fiducial
                    )

        #_____calculate likelihood_____
        loglike = loglikelihood_dfPar(
                                pot,
                                aA,
                                sf,
                                dftype,
                                dfPar_galpy,
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
                                data_for_outlier_model_galpy=data_for_outlier_model_galpy,
                                normalisation_for_outlier_model=norm_outlier,  
                                marginal_coord=marginal_coord,
                                weights_marginal=weights_marginal,
                                _N_ERROR_SAMPLES=_N_ERROR_SAMPLES
                                )

    if numpy.isnan(loglike):
        loglike = -numpy.inf
    elif loglike < -1.7e+308:
        loglike = -numpy.inf

    #_____priors on the model parameters_____
    #priortype = 0: flat priors in potential parameters and 
    #               logarithmically flat  priors in DF parameters
    #priortype = 1: additionally: prior on flat rotation curve 
    #               (makes no sense to use it in a fit of the DF only)
    if priortype in [0]:             
        logprior = calculate_logprior(priortype,pottype,potPar_phys,pot_physical,pot=pot)
    else:
        sys.exit("Error in logprob_MCMC_fitDF_only(): It makes no sense "+\
                 "to use priortype = "+str(priortype)+" because the "+\
                 "potential is not fitted anyway.")

    #_____print current walker position to file_____
    if not chainfilename is None:
        f = open(chainfilename, "a")
        for k in range(len(p)):
            f.write("%3.5f " % (p[k]))
        f.write("%3.5f \n" % (loglike + logprior))
        f.close()

    return loglike + logprior

#------------------------------------------------------------------------


def loglikelihood_potPar(pot,aA,sf,dftype,
                         _N_SPATIAL_R,_N_SPATIAL_Z,
                         ro,vo,
                         jr_data,lz_data,jz_data,
                         rg_data,
                         kappa_data,nu_data,Omega_data,
                         dfParArr_fit,
                         dfParFid_galpy,
                         _NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_XGL,_WGL,_MULTI,
                         datatype,noStars,
                         pottype,priortype,potPar_phys, #needed for calculating the prior
                         data_for_outlier_model_galpy=None,normalisation_for_outlier_model=None, #needed for outlier model
                         marginal_coord=None,weights_marginal=None,
                         _N_ERROR_SAMPLES=None):

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
        2013-??-?? - First version by Jo Bovy.
        2016-12-27 - Now makes use of function that calculates for a given priortype a non-flat prior for the model parameters. - Trick (MPIA)
        2017-01-02 - Now uses different dftypes with flexible number of parameters. Removed in_sf_data. - Trick (MPIA)
    """

    #_____initialize likelihood grid_____
    ndfs = len(dfParArr_fit[:,0])  #number of qdfs to test
    loglike_out = numpy.zeros(ndfs)

    #_____setup fiducial qdf and calculate actions on velocity grid_____
    # velocity grid that corresponds to the sigmas of the fiducial qdf.
    #print "   * calculate fiducial actions"
    if dftype in [0,11,12]: 
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
    else:
        sys.exit("Error in loglikelihood_potPar(): dftype = "+\
                 str(dftype)+" is not defined.")

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
    dfParArr_galpy = scale_df_fit_to_galpy(dftype,ro,vo,dfParArr_fit)


    #_____start iteration through distribution functions_____
    #print "   * iterate through qDFs"   
    if _MULTI is None or _MULTI == 1:
        for jj in range(ndfs):

            #_____likelihood_____
            loglike_out[jj] = loglikelihood_dfPar(
                                        pot,aA,sf,dftype,
                                        dfParArr_galpy[jj,:],#df and outlier parameters
                                        ro,vo,
                                        jr_data,lz_data,jz_data,
                                        rg_data,
                                        kappa_data,nu_data,Omega_data,
                                        _XGL,_WGL,
                                        datatype,noStars,
                                        data_for_outlier_model_galpy=data_for_outlier_model_galpy,
                                        normalisation_for_outlier_model=normalisation_for_outlier_model,
                                        marginal_coord=marginal_coord,
                                        weights_marginal=weights_marginal,
                                        _N_ERROR_SAMPLES=_N_ERROR_SAMPLES
                                        )

    elif _MULTI > 1:
        
        #....iterate over all sets of qdf parameters
        #    and calculate each on a separate core:
        loglike_out = multi.parallel_map(
                            (lambda x: loglikelihood_dfPar(
                                            pot,aA,sf,dftype,
                                            dfParArr_galpy[x,:], #df and outlier parameters
                                            ro,vo,
                                            jr_data,lz_data,jz_data,
                                            rg_data,
                                            kappa_data,nu_data,Omega_data,
                                            _XGL,_WGL,
                                            datatype,noStars,
                                            data_for_outlier_model_galpy=data_for_outlier_model_galpy,
                                            normalisation_for_outlier_model=normalisation_for_outlier_model,
                                            marginal_coord=marginal_coord,
                                            weights_marginal=weights_marginal,
                                            _N_ERROR_SAMPLES=_N_ERROR_SAMPLES
                                            )),
                            range(ndfs),
                            numcores=numpy.amin([
                                               ndfs,
                                               multiprocessing.cpu_count(),
                                               _MULTI
                                               ])
                            )
    loglike_out = numpy.array(loglike_out)

    #_____priors on the model parameters_____
    #priortype = 0: flat priors in potential parameters and 
    #               logarithmically flat  priors in DF parameters, i.e. logprior = 0.
    #priortype = 1: additionally: prior on flat rotation curve
    if priortype in [0,1]:
        pot_physical = True #any potential which is passed to this function is physical
        logprior = calculate_logprior(priortype,pottype,potPar_phys,pot_physical,pot=pot)
    else:
        sys.exit("Error in loglikelihood_potPar(): For priortype=[0,1] "+\
                 "the prior is independent of the values of the DF "+\
                 "parameters. If priortype="+str(priortype)+" contains "+\
                 "priors on the DF parameters, the code needs to be "+\
                 "modified accordingly. Otherwise make sure that the "+\
                 "code takes properly care of the new priortype.")

              
    return loglike_out + logprior

#-------------------------------------------------------------------

def loglikelihood_dfPar(pot,aA,sf,dftype,
                        dfPar_galpy,#[hr,sr,sz,hsr,hsz,outlier model parameters]
                        ro,vo,
                        jr_data,lz_data,jz_data,
                        rg_data,
                        kappa_data,nu_data,Omega_data,
                        _XGL,_WGL,
                        datatype,noStars,
                        data_for_outlier_model_galpy=None,normalisation_for_outlier_model=None,   #used for outlier model
                        marginal_coord=None,weights_marginal=None,  #used for marginalization over one coordinate
                        _N_ERROR_SAMPLES=None,return_likelihoods_of_stars=False):
    """
        NAME:
        PURPOSE:
        INPUT:
            dftype - int scalar - defines the DF (and outlier) model to use
            dfPar_galpy - float array - contains the qDF parameters, and the parameters used for the outlier model
        OUTPUT:
        HISTORY:
            2015-??-?? - Written - Trick (MPIA)
            2015-12-27 - Added simple outlier model. - Trick (MPIA)
            2016-02-16 - Made outlier model optional. - Trick (MPIA)
            2016-12-13 - Added datatype 5, which uses TGAS/RAVE data and a covariance error matrix. - Trick (MPIA)
            2016-12-31 - Allowed for different outlier models. Restructured the function concerning the different data types to make the code slimmer. - Trick (MPIA)
            2017-01-02 - Removed in_sf_data keyword. Finished implementing outlier model dftype == 12. - Trick (MPIA)
    """
    
    #_____initialize df_____
    # set parameters of distribution function:
    if dftype in [0,11,12]:
        #0:  single qDF, no outlier model
        #11: single qDF, robust likelihood
        #12: single qDF, mixture model for outliers
        hr  = dfPar_galpy[0]
        sr  = dfPar_galpy[1]
        sz  = dfPar_galpy[2]
        hsr = dfPar_galpy[3]
        hsz = dfPar_galpy[4]
        df = quasiisothermaldf(
                                hr,sr,sz,hsr,hsz,   
                                pot=pot,aA=aA,
                                cutcounter=True,
                                ro=ro)
        # (*Note:* if cutcounter=True, set counter-rotating stars' 
        #          DF to zero)
    else:
        sys.exit("Error in loglikelihood_dfPar(): dftype = "+str(dftype)+" is not defined.")

    #_____integrate density over effective volume_____
    # set distribution function in selection function:
    sf.reset_df(df)
    # setup density grid for interpolating the density 
    # (*Note 1:* we do not use evaluation on multiple cores here, 
    #            because we rather evaluate each qdf on a separate core)
    # (*Note 2:* The order of GL integration is set by ngl when setting 
    #            up the fiducial qdf.)
    sf.densityGrid(_multi=None)
    # integrate density over effective volume, i.e. the selection 
    # function. Use a fast GL integration analogous to Bovy:
    dens_norm = sf.Mtot(xgl=_XGL,wgl=_WGL)

    #_____evaluate DF for each data point_____
    #this is log(likelihood) of all data points given p_DF and p_Phi:
    lnL_i = df(
                (jr_data,lz_data,jz_data),
                rg   =rg_data,
                kappa=kappa_data,
                nu   =nu_data,
                Omega=Omega_data,
                log  =True
                )

    #_____normalize likelihood_____
    # for current p_DF and p_Phi:
    if dens_norm > 0.: 
        lnL_i -= math.log(dens_norm)
    else:              
        lnL_i = numpy.zeros_like(lnL_i) - numpy.inf
        print "Warning in loglikelihood_dfPar(): normalization is <= 0"+\
              " for df parameters: "
        print scale_df_galpy_to_phys(dftype,ro,vo,dfPar_galpy)
        print "normalization = ",dens_norm,"\n"

    # (*Note:* We ignore the model-independent prefactor Sum_i sf(x_i) 
    #          in the likelihood 
    #          L({x,v}|p) = Sum_i sf(x_i) * df(x_i,v_v) / norm 
    #          with norm = int sf(x) * df(x,v) dx dv 
    #          and only evaluate L({x,v}|p) = Sum_i df(x_i,v_i) / norm)

    #_____units of the likelihood_____
    # [L_i dx^3 dv^3] = 1 --> [L_i] = [xv]^{-3}
    logunits = 3. * numpy.log(ro*vo)
    lnL_i -= logunits

    if dftype == 12:
        #_____apply outlier model: mixture model_____
        if data_for_outlier_model_galpy is None:
            sys.exit("Error in loglikelihood_dfPar: "+\
                 "To use the outlier mixture model (oltype = 2), the keyword "+\
                 "data_for_outlier_model_galpy needs to be set.")
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out]

        #the outlier fraction p_out is most of the time also a fit parameter:
        p_out     = dfPar_galpy[5]
        #test, if outlier fraction is physical:
        if p_out < 0. or p_out > 1.:
            #if outlier fraction is unphysical, return log(0)
            lnL_i = numpy.zeros_like(lnL_i) - numpy.inf
        else:
            #outlier likelihood, properly normalized and in correct units:
            lnL_i_outl = calculate_outlier_model(
                                dftype,dfPar_galpy,
                                ro,vo,
                                data_galpy=data_for_outlier_model_galpy,
                                sf=sf,
                                norm_out=normalisation_for_outlier_model
                                )
            #likelihood according to physical model, units: [xv]^{-3}:
            L_i_phys = numpy.exp(lnL_i)
            #likelihood according to outlier model, units: [xv]^{-3}:
            L_i_outl = numpy.exp(lnL_i_outl)
            #mixture model (analogous to eq. 17 in Hogg, Bovy & Lang 2010, eq. 26 in Bovy & Rix (2013)):
            L_i_tot  = (1.-p_out) * L_i_phys + p_out * L_i_outl
            #back to log:
            lnL_i    = numpy.log(L_i_tot)

  
    #_____calculate likelihood for given data type_____
    if datatype in [1,4]:   
        #1: perfect mock data
        #4: perfect mock data (mix of 2 sets)

        pass    #nothing special happens
        
    elif datatype in [2,5]:
        #2: mock data with measurement errors
        #5: TGAS data with covariance error matrix

        #_____loglikelihood for each error sample_____
        lnL_err = lnL_i
        # *Note:* this is the likelihood of each of errors samples around the 
        #         observed data points. Which are, by the way, not in the 
        #         list anymore. Only Gaussian samples around them.

        #_____calculate likelihoods belonging to one real (!) data point____
        # by taking the mean for each data point we sum up the likelihoods, 
        # not the loglikelihoods:
        L_err = numpy.exp(lnL_err)
        # reshaping such that the error samples belonging to one real 
        # observed data point are again in the same row:
        L_err       = numpy.reshape(L_err     ,(_N_ERROR_SAMPLES,noStars))
        # the convolution integral of the likelihood with the error gaussian
        # is calculated in the following way:
        #       int L(x|p) * N[x_i,e_x](x) dx'
        #               ~ 1/M * sum_j=1^M df(x_j) * sf(x_j) / norm
        # where x_j sample N[x_i,e_x](x_j), 
        # and M = _NERRSAMPLE,
        # and L(x|p) = df(x) * sf(x) / norm = L_err * sf(x)
        # (*Note:* Our error approximation assumes the spatial position 
        #          to be perfectly known and inside the survey volume. 
        #          sf(x) is > 0 for each error sample.)
        # (*Note:* We ignore the model-independent prefactor Sum_i sf(x_i) 
        #          in the likelihood and only evaluate L(x|p) = df(x) / norm)
        # (*Note:* axis=0 means summing only over rows.)
        L_i = numpy.sum(L_err,axis=0) / float(_N_ERROR_SAMPLES)
        # back to log likelihood:
        lnL_i = numpy.log(L_i)
        if len(lnL_i) != noStars: 
            sys.exit("Error in loglikelihood_dfPar: "+\
                     "calculating mean likelihood does not give "+\
                     "right number of elements.")


    elif datatype == 3:
        #3: perfect mock data, marginalization over one coordinate

        sys.exit("Error in loglikelihood_dfPar(): "+\
                 "Not sure if code for datatype = 3 is still doing "+\
                 "what it is supposed to. It might not work together "+\
                 "with dftype=12. Check!")

        #_____loglikelihood for each error sample_____
        lnL_j_all = lnL_i + logunits
        # *Note:* This is the log(likelihood) at each data point, 
        #         including the data points 
        #         used to marginalize over one of the velocities
        # *Note:* This likelihood does not yet have the proper units.
        #         We add logunits here, because we have subtracted 
        #         logunits earlier and we effectively want to undo this. 
        #         The proper units will be applied later. 
        #         (Could be done more elegantly. Well... Keep it here for later.)

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


    if dftype == 11:
        #_____simple outlier model: robust likelihood_____
        #all likelihoods have to be bigger than $\epsilon \cdot \bar{\mathscr{L}}$
        #analogous to what was used in Trick, Bovy, D'Onghia & Rix (2017)
        L_i      = numpy.exp(lnL_i)
        median_L = numpy.median(L_i)
        epsilon  = 0.001 # = 0.1 %
        L_i      = numpy.maximum(L_i,epsilon*median_L)
        lnL_i    = numpy.log(L_i)

    #_____sum logL for final loglikelihood_____
    # sum up contributions for all data points: 
    loglike_out = numpy.sum(lnL_i)  

    if return_likelihoods_of_stars:
        return lnL_i
    else:
        return loglike_out


