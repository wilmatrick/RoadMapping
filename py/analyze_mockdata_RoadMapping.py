#______________________________
#_____import packages__________
#______________________________
#from __past__ import division
#from __future__ import print_function
import emcee
import importlib
import math
import multiprocessing
import numpy
import os
import pickle
import sys
import time
import scipy.optimize
import galpy
from galpy.potential import Potential
from galpy.actionAngle import actionAngle, estimateDeltaStaeckel
from galpy.util import save_pickles, multi
from SelectionFunction import SelectionFunction
from read_RoadMapping_parameters import read_RoadMapping_parameters
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object,setup_SelectionFunction_object
from setup_parameter_fit import setup_parameter_fit
from read_and_setup_data_set import read_and_setup_data_set
import likelihood
from likelihood import loglikelihood_potPar
from emcee.interruptible_pool import InterruptiblePool as Pool
from precalc_actions import setup_data_actions
from setup_shared_data import shared_data_MCMC, shared_data_DFfit_only_MCMC, shared_data_incompleteShell
from outlier_model import scale_df_fit_to_galpy

def analyze_mockdata_RoadMapping(datasetname,testname=None,multicores=63,mockdatapath='../data/',redo_analysis=True,method='GRID'):

    """
        NAME:
           analyze_mockdata_RoadMapping
        PURPOSE:
            Run the RoadMapping action-based dynamical modeling machinery.
        INPUT:
            datasetname   - name of the data set to analyze. Expects the data to be at
                            mockdatapath+"/"+datasetname+"/"+ datasetname+"_mockdata.sav"
            testname      - name of this specific RoadMapping analysis. Expects a analysis parameter file at
                            mockdatapath+"/"+datasetname+"/"+datasetname+"_"+testname+"_analysis_parameters.txt"
            multicores    - number of cores to use in parallel processing. 
                            Note, this code runs on a single node only, with multiple cores. 
                            Usually, RoadMapping would run analyses for several MAPs, 
                            so I analyse different MAPs on different nodes, but each MAP only on one node.
                            Ideally, the number of cores available is an integer multiple of the 
                            number of walkers used in the MCMC (= 64).
            mockdatapath  - Folder where the data can be found (see above).
            redo_analysis - If False, pick off the analysis where it was interrupted previously. 
                            If True, overwrite any existing analysis output files.
            method        - Options are 'GRID' and 'MCMC'. I recommend performing a nested 'GRID' search 
                            a few times first to find the peak and width of the pdf, and then launch the 
                            'MCMC' at the peak to sample the exact shape of the pdf more smoothly.
                       
        OUTPUT:
            None. The analysis output will be stored in a folder '../out/'.
        HISTORY:
           2015-11-27 - Started analyze_mockdata_RoadMapping.py on the basis of 
                        BovyCode/py/analyze_mockdata_RoadMapping.py - Trick (MPIA)
           2016-02-16 - Added _MULTI keyword to setup_Potential_and_ActionAngle_object() 
                        to speed up the ActionAngleStaeckelGrid. - Trick (MPIA)
           2016-04-15 - Added keywords to setup_Potential_and_ActionAngle_object() and 
                        MCMC_info that take care of choosing different Staeckel Deltas.
           2016-05-02 - Changed MCMC from 100 walkers to 64 walkers, to account for actual number of CPUs on my cluster.
           2016-12-13 - Added datatype = 5, which uses TGAS/RAVE data and a covariance error matrix. - Trick (MPIA)
           2016-12-27 - The likelihood now takes also care of priors on the potential parameters. - Trick (MPIA)
           2017-01-03 - Added calculation of normalisation and data for oulier model (dftype = 12). 
                        Removed in_sf_data. - Trick (MPIA)
           2017-01-09 - Test now for all datatypes if all stars are inside the selection function. - Trick (MPIA)
           2021-10-20 - Added an Exception if any datatype other than 1 is used, 
                        and a warning about galpy versions other than 1.2. - Trick (MPA
    """

    #if galpy.__version__ != '1.2':
    #    sys.exit("Error in analyze_mockdata_RoadMapping(): RoadMapping works currently with galpy.__version__ = 1.2 only.")
    print("DEBUGGING OUTPUT in analyze_mockdata_RoadMapping(): Switched off warning about only using galpy's version 1.2 in October 2021. Make sure this does not lead to unexpected problems when using different galpy versions. Especially for Python 3.")

    print("______________________________________________________________________")
    print("_____Analyse mock data set: ",datasetname)
    if testname is not None: print("_____test: ",testname)
    print("_____with: ",method)
    print("______________________________________________________________________")

    #_____read analysis parameters and write them to screen_____
    ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            mockdatapath=mockdatapath,
            print_to_screen=True)

    #_______________________________
    #_____set global variables______
    #_______________________________
    
    #...Number of cores on which the evaluation is done:
    _MULTI = multicores 

    #...Reference scales:
    _REFR0 = 8.                 #[kpc], Reference radius 
    _REFV0 = 220.               #[km/s], Reference velocity

    #...Density grid for effective volume calculation:
    _N_SPATIAL_R = ANALYSIS['N_spatial'] #Default: 16
    _N_SPATIAL_Z = ANALYSIS['N_spatial'] #Default: 16
    # (*Note:* the predicted spatial density (for a given DF) is 
    #          calculated on a 16x16 grid in (R,z), 
    #          cf. Appendix in Bovy & Rix (2013))
    _NGL_VELOCITY = ANALYSIS['N_velocity'] #in Bovy & Rix (2013): 20. Must be at least 20! important! 
    # (*Note:* Gauss Legendre order for integration of velocities at 
    #          each (R,z) grid point when setting up the density 
    #          interpolation grid)
    _N_SIGMA      = ANALYSIS['N_sigma']
    _VT_GALPY_MAX = ANALYSIS['vT_galpy_max']

    #...prepare Gauss Legendre integration...
    #...of density over effective volume  ...
    _effVol_ngl = 40    #order of Gauss Legendre integration
    _XGL,_WGL = numpy.polynomial.legendre.leggauss(_effVol_ngl)    
    # (*Note:* sample points (x) and weights (w) for 
    #          Gauss-Legendre (gl) quadrature)
    #...of likelihood over coordinates    ...
    xgl_marginal, wgl_marginal = numpy.nan, numpy.nan
    if ANALYSIS['datatype'] == 3:
        ngl_marginal = ANALYSIS['ngl_marginal']
        xgl_marginal, wgl_marginal = numpy.polynomial.legendre.leggauss(ngl_marginal) 

    #...for measurement errors: number of MC samples to marginalize over errors
    _N_ERROR_SAMPLES = 1
    if ANALYSIS['datatype'] in [2,5]:
        _N_ERROR_SAMPLES = ANALYSIS['N_error_samples']

    #...number of stars in the data set:
    if ANALYSIS['datatype'] == 4:   #mixed populations
        noStars =  ANALYSIS['noStars'][0]
    else:
        noStars = ANALYSIS['noStars']
        
    #...abort analysis to warn about possible bug:
    if ANALYSIS['datatype'] != 1:
        raise Exception('----- ABORT ROADMAPPING FOR ANY DATATYPE OTHER THAN 1 (perfect mock data) -----\n'+\
                       'Wilma suspects that there might be still a little bug in the code \n'+\
                       'that does not account for v_circ(R_Sun) as a free parameter when converting from ra,dec etc. \n'+\
                       'to Galactocentric coordinates and then to actions.\n'+
                       'Check first and fix if necessary before doing anything else.')

    #_____start timer_____
    zeit_start = time.time()

    #________________________
    #_____mock data set______
    #________________________

    #_____read & setup mock data_____
    dataset = read_and_setup_data_set(
                datasetname,
                testname=testname,
                mockdatapath=mockdatapath
                )
    R_data   = dataset[0]   #[kpc  /_REFR0]
    vR_data  = dataset[1]   #[km/s /_REFV0]
    phi_data = dataset[2]   #[deg]
    vT_data  = dataset[3]   #[km/s /_REFV0]
    z_data   = dataset[4]   #[kpc  /_REFR0]
    vz_data  = dataset[5]   #[km/s /_REFV0]

    #_____initialize selection function_____
    if ANALYSIS['ro_known']:
        sf = setup_SelectionFunction_object(
                        ANALYSIS['sftype'],
                        ANALYSIS['sfParEst_phys'],
                        ANALYSIS['ro']
                        )

        #Test if all stars are inside the survey volume and in regions 
        #with completeness > 0:
        in_sf_data = sf.contains(R_data/ANALYSIS['ro'],z_data/ANALYSIS['ro'],phi=phi_data)
        if numpy.sum(in_sf_data) < len(in_sf_data):
            sys.exit("Error in analyze_mockdata_RoadMapping(): "+\
                     "There are measured positions that are outside of "+\
                     "the observed volume. This should not be the case.")
        #In case of measurement errors (datatype in [2,5]), our error 
        #approximation assumes the positions are perfectly known,
        #therefore nothing should be outside of the survey volume.
    else:
        sys.exit("Error in analyze_mockdata_RoadMapping(): "+\
                 "How to deal with the selection function if ro is not known? To do: Implement.")

    #_____prepare data for outlier model_____
    normalisation_for_outlier_model = None
    data_for_outlier_model = None
    if ANALYSIS['dftype'] == 12:
        #Integrate the f(x)=1 over the selection function.
        #Needed as normalisation for the outlier model.
        normalisation_for_outlier_model = sf.sftot(xgl=_XGL,wgl=_WGL)
        #store data later used for outlier model.
        data_for_outlier_model = numpy.zeros((4,len(R_data)))
        data_for_outlier_model[0,:] = R_data
        data_for_outlier_model[1,:] = vR_data
        data_for_outlier_model[2,:] = vT_data
        data_for_outlier_model[3,:] = vz_data
        


    if method == 'GRID':
        #_____________________________________
        #_____Prepare analysis on Grid________
        #_____________________________________

        #_____initialize parameter grid_____
        print("Parameter intervals to investigate ")
        FITGRID = setup_parameter_fit(
                    datasetname,testname=testname,
                    mockdatapath=mockdatapath,
                    print_to_screen=True
                    )
        potParArr_phys = FITGRID['potParArr_phys']
        dfParArr_fit   = FITGRID['dfParArr_fit']

        #_____initialize likelihood grid_____
        likelihood_shapetuple = FITGRID['potShape']+FITGRID['dfShape']
        npots = len(potParArr_phys[:,0])
        ndfs  = len( dfParArr_fit[:,0])
        loglike_out = numpy.zeros((npots,ndfs))
        
        #_____load current state of analysis from file_____
        if testname is None: savefilename = "../out/"+datasetname+"_analysis_output_GRID.sav"
        else:                savefilename = "../out/"+datasetname+"_"+testname+"_analysis_output_GRID.sav"
        # test if input file exists already, 
        # if yes, load previous iteration indexes etc.   
        if os.path.exists(savefilename) and not redo_analysis:                                          
            print("Loading state from file "+savefilename)
            savefile    = open(savefilename,'rb')
            loglike_out = pickle.load(savefile)
            ii          = pickle.load(savefile)
            savefile.close()
        else:
            ii = 0                         #initialize analysis

        #___________________________________
        #_____Start analysis on Grid________
        #___________________________________

        print("\n")

        print(" *** Start running the GRID analysis *** ")
            
        #_____start iteration through potentials_____
        start = time.time()
        while ii < npots:
            if True:
                #text  = "\r"
                #text += "Working on potential "+str(ii+1)+" / "+str(npots)
                #text += "time of one potential: "+str(time.time() - start)
                #start = time.time()
        
                #_____setup potential / Action object_____
                #print("   * setup potential")
                potPar_phys = potParArr_phys[ii,:]
                try:
                    if ANALYSIS['use_default_Delta']:
                        pot, aA = setup_Potential_and_ActionAngle_object(
                                        ANALYSIS['pottype'],
                                        potPar_phys,
                                        #Setting up actionAngleStaeckelGrid:
                                        _MULTI           =_MULTI,
                                        aASG_accuracy    =ANALYSIS['aASG_accuracy']
                                        )
                    else:
                        if ANALYSIS['estimate_Delta']:
                            pot_temp= setup_Potential_and_ActionAngle_object(
                                        ANALYSIS['pottype'],
                                        potPar_phys,
                                        return_only_potential=True
                                        )
                            aAS_Delta = estimateDeltaStaeckel(
                                        R_data,
                                        z_data,
                                        pot=pot_temp
                                        )
                        else:
                            aAS_Delta = ANALYSIS['Delta_fixed']
                        pot, aA = setup_Potential_and_ActionAngle_object(
                                        ANALYSIS['pottype'],
                                        potPar_phys,
                                        #Setting up actionAngleStaeckelGrid:
                                        _MULTI           =_MULTI,
                                        aASG_accuracy    =ANALYSIS['aASG_accuracy'],
                                        #Staeckel Delta:
                                        aAS_Delta        =aAS_Delta
                                        )
                except RuntimeError as e:
                    # if this set of parameters gives a nonsense potential, 
                    # fix output to maximum negative number...
                    loglike_out[ii,:] = -numpy.finfo(numpy.dtype(numpy.float64)).max
                    #... and continue with loop.
                    ii += 1
                    continue
                ro = potPar_phys[0] / _REFR0
                vo = potPar_phys[1] / _REFV0

                #_____rescale fiducial qdf parameters to galpy units_____
                #these parameters are used to set the integration grid 
                #of the df over velocities to get the density.
                dfParFid_fit = ANALYSIS['dfParFid_fit']
                dfParFid_galpy = scale_df_fit_to_galpy(ANALYSIS['dftype'],ro,vo,dfParFid_fit)

                #_____rescale data for outlier model to galpy units_____
                data_for_outlier_model_galpy = None
                if ANALYSIS['dftype'] == 12:
                    traf = numpy.array([ro,vo,vo,vo]).reshape((4,1))
                    data_for_outlier_model_galpy = data_for_outlier_model / traf

                #print("   * calculate data actions")
                #_____calculate actions and frequencies of the data_____
                #before calculating the actions, the data is properly 
                #scaled to galpy units with the current potentials vo and ro
                weights_marginal = None #only used in datatype = 4
                R_galpy   =  R_data/ro
                vR_galpy  = vR_data/vo
                vT_galpy  = vT_data/vo
                z_galpy   =  z_data/ro
                vz_galpy  = vz_data/vo
                if ANALYSIS['pottype'] == 1 and ANALYSIS['datatype'] == 1:
                    #isochrone potential, perfect mock data: 1 core only
                    actions = setup_data_actions(pot,aA,
                           ANALYSIS['dftype'],
                           R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                           dfParFid_galpy,ro,
                           1)
                elif ANALYSIS['datatype'] == 3:
                    #perfect mock data, marginalization over one coordinate
                    if ANALYSIS['marginal_coord'] == 4: #marginalize over vT
                        vTmin = 0.  #integration limits
                        vTmax = _VT_GALPY_MAX
                        weights_marginal = 0.5 * (vTmax - vTmin) * wgl_marginal
                        vT_marginal = 0.5 * (vTmax - vTmin) * xgl_marginal + 0.5 * (vTmax + vTmin)
                        vT_galpy = numpy.tile(vT_marginal,noStars)
                    else:
                        sys.exit("Error in analyze_mockdata_RoadMapping(): The "+\
                                 'marginalization over "'+ANALYSIS['marginal_coord']+'" is not yet '+
                                 "implemented. So far only the marginalization over 'vT' is "+
                                 "implemented.")
                    actions = setup_data_actions(pot,aA,
                           ANALYSIS['dftype'],
                           R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                           dfParFid_galpy,ro,
                           _MULTI)
                else:
                    actions = setup_data_actions(pot,aA,
                           ANALYSIS['dftype'],
                           R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                           dfParFid_galpy,ro,
                           _MULTI)
                jr_data    = actions[0,:]
                lz_data    = actions[1,:]
                jz_data    = actions[2,:]
                rg_data    = actions[3,:]
                kappa_data = actions[4,:]
                nu_data    = actions[5,:]
                Omega_data = actions[6,:]
      
                #_____calculate the likelihoods for all p_DF parameters_____
                # for the given p_Phi parameters
                #print("   * calculate likelihood")
                loglike_out[ii,:] = loglikelihood_potPar(
                                        pot,aA,sf,ANALYSIS['dftype'],
                                        _N_SPATIAL_R,_N_SPATIAL_Z,
                                        ro,vo,
                                        jr_data,lz_data,jz_data,
                                        rg_data,
                                        kappa_data,nu_data,Omega_data,
                                        dfParArr_fit,
                                        dfParFid_galpy,
                                        _NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_XGL,_WGL,
                                        _MULTI,
                                        ANALYSIS['datatype'],noStars,
                                        ANALYSIS['pottype'],ANALYSIS['priortype'],potPar_phys, #needed for calculating the prior
                                        data_for_outlier_model_galpy=data_for_outlier_model_galpy, #needed for outlier model
                                        normalisation_for_outlier_model=normalisation_for_outlier_model, #needed for outlier model
                                        marginal_coord=ANALYSIS['marginal_coord'],
                                        weights_marginal=weights_marginal,
                                        _N_ERROR_SAMPLES=_N_ERROR_SAMPLES
                                        )



                #_____save output, current iteration state, and the grid axes_____
                save_pickles(
                    savefilename,
                    loglike_out.reshape(likelihood_shapetuple), #likelihood in correct grid shape
                    ii+1,                                       #next potential to calculate
                    FITGRID['fitParNamesLatex'],                #names of axes in Latex
                    FITGRID['gridPointNo'],                     #number of points along each axis
                    FITGRID['gridAxesPoints'],                  #all axes in one flattened array
                    FITGRID['gridAxesIndex'],                   #indices of start and end of each axis in above array
                    ANALYSIS['potParFitBool'],                  #boolean array that indicates which of all potential parameters are fitted
                    ANALYSIS['dfParFitBool'],                   #boolean array that indicates which of all DF parameters are fitted
                    ANALYSIS['potParTrue_phys'],                #true potential parameters in physical units
                    ANALYSIS['dfParTrue_fit']                   #true df parameters in logarithmic fit units
                    )
            ii += 1

    elif method == 'MCMC':
        #_____________________________________
        #_____Prepare analysis with MCMC________
        #_____________________________________

        #_____initialize parameter grid_____
        print("Parameter intervals to investigate with MCMC:")
        print("     * potential: central grid point only used for initial walker positions")
        print("     * DF:        central grid point used for initial walker positions & fiducial qDF / fitting range")
        FITGRID = setup_parameter_fit(
                    datasetname,testname=testname,
                    mockdatapath=mockdatapath,
                    print_to_screen=True
                    )
    

        #_____load other parameters used for logprob_____
        #df parameters:
        dfParEst_fit  = ANALYSIS['dfParEst_fit'] #df parameter estimate used for the fixed df parameters
        dfParFid_fit  = ANALYSIS['dfParFid_fit']
        dfParFitBool  = ANALYSIS['dfParFitBool']
        #pot parameters:
        potParEst_phys= ANALYSIS['potParEst_phys']
        potParFitBool = ANALYSIS['potParFitBool']
        sfParEst_phys = ANALYSIS['sfParEst_phys']
        #model & data type:
        datatype      = ANALYSIS['datatype']
        pottype       = ANALYSIS['pottype']
        sftype        = ANALYSIS['sftype']
        dftype        = ANALYSIS['dftype']
        ro_known      = ANALYSIS['ro_known']
        priortype     = ANALYSIS['priortype']
        #walker initialization:
        min_walkerpos = FITGRID['min_walkerpos']
        max_walkerpos = FITGRID['max_walkerpos']
        noMCMCsteps   = ANALYSIS['noMCMCsteps']
        #setup MCMC:
        noMCMCburnin   = ANALYSIS['noMCMCburnin']
        MCMC_use_fidDF = ANALYSIS['MCMC_use_fidDF']
        #parameters used for marginalization (datatype == 3):
        marginal_coord = ANALYSIS['marginal_coord']
        #actionAngleStaeckel(Grid):
        use_default_Delta = ANALYSIS['use_default_Delta']
        estimate_Delta = ANALYSIS['estimate_Delta']
        Delta_fixed    = ANALYSIS['Delta_fixed']
        aASG_accuracy  = ANALYSIS['aASG_accuracy']

        #___________________________________
        #_____Start analysis with MCMC______
        #___________________________________

        print(" *** Save & read shared data *** ")

        #_____prepare output for current iteration state_____
        if testname is None: chainfilename = "../out/"+datasetname+"_chain_MCMC.dat"
        else:                chainfilename = "../out/"+datasetname+"_"+testname+"_chain_MCMC.dat"               
        f = open(chainfilename, "w")
        f.write("# ")
        for item in FITGRID['fitParNamesScreen']:
            f.write(item+"\t")
        f.write("log(prob)\n")
        f.close()

        #_____setup dictionary with additional MCMC information_____
        info_MCMC = {   'chainfilename':chainfilename,
                        'potParEst_phys':potParEst_phys,
                        'dfParEst_fit':dfParEst_fit,
                        'dfParFid_fit':dfParFid_fit,
                        'sfParEst_phys':sfParEst_phys,
                        'potParFitBool':potParFitBool,
                        'dfParFitBool':dfParFitBool,
                        'ro_known':ro_known,
                        '_N_SPATIAL_R':_N_SPATIAL_R,
                        '_N_SPATIAL_Z':_N_SPATIAL_Z,
                        '_NGL_VELOCITY':_NGL_VELOCITY,
                        '_N_SIGMA':_N_SIGMA,
                        '_VT_GALPY_MAX':_VT_GALPY_MAX,                      
                        '_XGL':_XGL,
                        '_WGL':_WGL,
                        'datatype':datatype,
                        'pottype':pottype,
                        'sftype':sftype,
                        'dftype':dftype,
                        'priortype':priortype,
                        'noStars':noStars,
                        'norm_outlier':normalisation_for_outlier_model,
                        'marginal_coord':marginal_coord,
                        'xgl_marginal':xgl_marginal,
                        'wgl_marginal':wgl_marginal,
                        '_N_ERROR_SAMPLES':_N_ERROR_SAMPLES,
                        'MCMC_use_fidDF':MCMC_use_fidDF,
                        'use_default_Delta':use_default_Delta,
                        'estimate_Delta':estimate_Delta,
                        'Delta_fixed':Delta_fixed,
                        'aASG_accuracy':aASG_accuracy,
                        'aAS_Delta_DFfitonly': None
                         }

        #_____test if only the DF is fitted_____
        DF_fit_only = (numpy.sum(potParFitBool) == 0)
        if DF_fit_only:
            #Setting up the StaeckelFudge ActionGrid is very slow. 
            #As we precalculate all actions anyway, we use the standard StaeckelFudge
            #to set up the potential in the MCMC chain.
            if pottype in numpy.array([21,31,41,421,51,61,71,81]): 
                pottype_slim = (pottype-1)//10
                info_MCMC['pottype'] = pottype_slim
            #Set the Staeckel Fudge Delta once and for all:
            if use_default_Delta: 
                if not ro_known: 
                    sys.exit("Error in analyze_mockdata_RoadMapping:"+\
                             "ro is not known. How to deal with the "+\
                             "Staeckel Fudge Delta?")
                info_MCMC['aAS_Delta_DFfitonly'] = 0.45
            elif estimate_Delta:
                sys.exit("Error in analyze_mockdata_RoadMapping:"+\
                         "Estimating Delta for the DF fit only is "+\
                         "not yet implemented.")
            elif (not use_default_Delta) and (not estimate_Delta): 
                info_MCMC['aAS_Delta_DFfitonly'] = Delta_fixed

        #_____1. save shared data_____
        #write current path into file:
        if testname is None: current_path = mockdatapath+datasetname+"/"+datasetname
        else:                current_path = mockdatapath+datasetname+"/"+datasetname+"_"+testname
        current_path_filename = "temp/path_of_current_analysis.sav"
        save_pickles(
            current_path_filename,
            current_path
            )
        #write MCMC info into sav file:
        info_MCMC_filename = current_path+'_info_MCMC.sav'
        save_pickles(
            info_MCMC_filename,
            info_MCMC
            )
        #write shared data into .npy binary files:
        #write (R,vR,vT,z,vz) into file...
        shared_data_MCMC(
                 R_data,vR_data,vT_data,z_data,vz_data,
                 current_path)
        #file names:
        shared_data_filename = current_path+'_shared_data.npy'
        if DF_fit_only:   
            #...special case: potential is kept fixed: 
            #                 write all pre-calculated actions into file...
            shared_data_DFfit_only_MCMC(pottype,sftype,datatype,dftype,
                            potParEst_phys,dfParFid_fit,sfParEst_phys,
                            R_data,vR_data,vT_data,z_data,vz_data,
                            ro_known,
                            _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_MULTI,
                            aASG_accuracy,use_default_Delta,estimate_Delta,Delta_fixed,
                            current_path)
            #file names:
            shared_data_actions_filename = current_path+'_shared_data_actions.npy'
            shared_fiducial_actions_filename = current_path+'_shared_fiducial_actions.npy'

        #...special case: Selection Function: Incompleteness shell
        if sftype == 4 and not DF_fit_only: 
            if not ro_known: sys.exit("Error in analyze_mockdata_RoadMapping(): ro is not known. Not yet clear how to deal with that.")
            shared_data_incompleteShell(sftype,sfParEst_phys,ANALYSIS['ro'],current_path)
            shared_data_SF_incompleteShell_filename = current_path+'_shared_SF_incompleteShell.npy'

        #_____2. load shared data into global variables_____
        importlib.reload(likelihood)
        if not DF_fit_only: #standard case
            from likelihood import logprob_MCMC as log_likelihood_MCMC
        else:   #special case: potential is kept fixed
            from likelihood import logprob_MCMC_fitDF_only as log_likelihood_MCMC
            

        #_____3. delete files immediately_____
        os.remove(current_path_filename)
        os.remove(info_MCMC_filename)
        os.remove(shared_data_filename)
        if DF_fit_only:   #special case: potential is kept fixed
            os.remove(shared_data_actions_filename)
            os.remove(shared_fiducial_actions_filename)
        if sftype == 4 and not DF_fit_only: #special case: SF_IncompleteShell
            os.remove(shared_data_SF_incompleteShell_filename)
        

        print(" *** Start running the MCMC *** ")


        #_____randomize random number generator_____
        numpy.random.seed(seed=None)

        #_____initialize walkers_____
        #number of free fit parameters:
        ndim     = numpy.sum(potParFitBool) + numpy.sum(dfParFitBool)
        #number of walkers:
        nwalkers = 64
        #initial walker positions:
        p0 = numpy.random.randn(ndim * nwalkers).reshape((nwalkers, ndim))
        p0 = p0 * 0.25 * (max_walkerpos - min_walkerpos) #central pixel corresponds to +/- 2 sigma
        p0 += 0.5 * (max_walkerpos + min_walkerpos)

        #_____Setup pool_____
        if _MULTI is not None and _MULTI > 1:
            #pool = Pool(numpy.amin([multiprocessing.cpu_count(),_MULTI]))
            pool = emcee.interruptible_pool.InterruptiblePool(processes=numpy.amin([multiprocessing.cpu_count(),_MULTI]))

        #_____initialize emcee MCMC sampler_____
        if _MULTI is not None and _MULTI > 1:
            sampler = emcee.EnsembleSampler(
                        nwalkers, 
                        ndim, 
                        log_likelihood_MCMC, #logprob_MCMC,
                        #threads=numpy.amin([multiprocessing.cpu_count(),_MULTI]),
                        pool=pool
                        )
        else:
            sampler = emcee.EnsembleSampler(
                        nwalkers, 
                        ndim, 
                        log_likelihood_MCMC #logprob_MCMC
                        )


        #_____grid axes_____
        if testname is None: savefilename = "../out/"+datasetname+"_parameters_MCMC.sav"
        else:                savefilename = "../out/"+datasetname+"_"+testname+"_parameters_MCMC.sav"           
        save_pickles(
            savefilename,
            FITGRID['fitParNamesLatex'],                #names of axes in Latex
            FITGRID['gridPointNo'],                     #number of points along each axis
            FITGRID['gridAxesPoints'],                  #all axes in one flattened array
            FITGRID['gridAxesIndex'],                   #indices of start and end of each axis in above array
            ANALYSIS['potParFitBool'],                  #boolean array that indicates which of all potential parameters are fitted
            ANALYSIS['dfParFitBool'],                   #boolean array that indicates which of all DF parameters are fitted
            ANALYSIS['potParTrue_phys'],                #true potential parameters in physical units
            ANALYSIS['dfParTrue_fit']                   #true df parameters in logarithmic fit units
            )

        #_____run MCMC_____
        pos, prob, state = sampler.run_mcmc(p0, noMCMCsteps)


        #____Close pool_____
        if _MULTI is not None and _MULTI > 1:
            pool.close()

        #____output MCMC chain_____
        #chain_out = numpy.zeros((nwalkers,nsteps,ndim))
        chain_out = sampler.chain

        #_____save output and the grid axes_____
        if testname is None: savefilename = "../out/"+datasetname+"_analysis_output_MCMC.sav"
        else:                savefilename = "../out/"+datasetname+"_"+testname+"_analysis_output_MCMC.sav"       
        save_pickles(
            savefilename,
            chain_out,                                  #MCMC chain
            FITGRID['fitParNamesLatex'],                #names of axes in Latex
            FITGRID['gridPointNo'],                     #number of points along each axis
            FITGRID['gridAxesPoints'],                  #all axes in one flattened array
            FITGRID['gridAxesIndex'],                   #indices of start and end of each axis in above array
            ANALYSIS['potParFitBool'],                  #boolean array that indicates which of all potential parameters are fitted
            ANALYSIS['dfParFitBool'],                   #boolean array that indicates which of all DF parameters are fitted
            ANALYSIS['potParTrue_phys'],                #true potential parameters in physical units
            ANALYSIS['dfParTrue_fit']                   #true df parameters in logarithmic fit units
            )

    #_____time taken_____
    zeit_t = time.time() - zeit_start
    zeit_th = int(zeit_t/3600.)
    zeit_tm = int((zeit_t - zeit_th*3600.0)/60.0)
    zeit_ts = zeit_t % 60.
    print("")
    print("Finished analysis of: ",savefilename)
    print("time:", zeit_th, "hours", zeit_tm, "minutes",int(zeit_ts), "seconds")
   
    return None

#------------------------------------------------------------------------------------------
