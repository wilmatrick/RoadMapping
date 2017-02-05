#_____import packages_____
import numpy
import os
import pickle
from write_RoadMapping_parameters import write_RoadMapping_parameters
from read_RoadMapping_parameters import read_RoadMapping_parameters
from get_N_MCMC_models import get_MCMC_mean_SE, get_GRID_midpoint
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object,setup_SelectionFunction_object
from galpy.df import quasiisothermaldf
from galpy.util import save_pickles
from precalc_actions import setup_data_actions
import sys
from outlier_model import scale_df_fit_to_galpy, scale_df_galpy_to_phys


def create_bestfit_reference_mockdata(datasetname_original,datasetname_reference,
                                      _DATATYPE,_NSTARS,
                                      testname_original=None,
                                      output_path='../out/',
                                      mockdata_path='../data/',
                                      _MULTI=None,
                                      _N_SPAT=20,_NGL_VEL=40,_N_SIGMA=5.,
                                      with_actions=False,
                                      redo_everything=False,
                                      method='MCMC_median',
                                      originalmockdatafilename=None
                                      ):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            201?-??-?? - Written. - Trick (MPIA)
            2017-01-03 - Allowed for flexible number of df parameters, i.e. added dftype. - Trick (MPIA)
            2017-01-18 - Added optional keyword originalmockdatafilename. - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.     #spatial scaling
    _REFV0 = 220.   #velocity scaling

   #_____get best fit values_____
    print "* Get best fit values for ",datasetname_original," *"


    #MCMC output filename:
    if testname_original is None: 
        analysis_original = output_path+datasetname_original+"_analysis_output_MCMC.sav"
        analysis_original_grid = output_path+datasetname_original+"_analysis_output_GRID.sav"
    else: 
        analysis_original = output_path+datasetname_original+"_"+testname_original+"_analysis_output_MCMC.sav"
        analysis_original_grid = output_path+datasetname_original+"_"+testname_original+"_analysis_output_GRID.sav"

    #get best fit values from previous analysis:
    if method == 'MCMC_median':
        means, stddevs, medians = get_MCMC_mean_SE(
                            datasetname_original,
                            testname=testname_original,
                            analysis_output_filename=analysis_original,
                            mockdatapath=mockdata_path,
                            quantities_to_calculate=None,
                            Gaussian_fit=False
                            )
        print "best fit values (median of MC distribution): ", medians
        bestfits = medians
    elif method == 'GRID_midpoint':
        ParFit_GRID_phys = get_GRID_midpoint(
                            datasetname_original,
                            testname=testname_original,
                            analysis_output_filename=analysis_original_grid,
                            mockdatapath=mockdata_path,
                            fulldatapath=None,
                            quantities_to_calculate=None
                            )
        bestfits = ParFit_GRID_phys

    #read analysis parameters:
    ANALYSIS = read_RoadMapping_parameters(
            datasetname_original,
            testname=testname_original,
            mockdatapath=mockdata_path
            )

    #separate coordinates into free potential and free df coordinates
    npotpar = numpy.sum(ANALYSIS['potParFitBool'])
    potPar_phys_MCMC = bestfits[0:npotpar]    #physical units
    dfPar_fit_MCMC   = bestfits[npotpar::]    #logarithmic fit units, not yet galpy scaled

    #p_Phi parameters
    #load estimates for fixed potential parameters
    potPar_phys = ANALYSIS['potParEst_phys']
    #overwrite the free parameters with current parameter set
    potPar_phys[ANALYSIS['potParFitBool']] = potPar_phys_MCMC
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0
    print "potential parameters: ",potPar_phys

    #p_DF parameters in galpy units
    #load estimates for fixed DF parameters
    dfPar_fit               = ANALYSIS['dfParEst_fit']
    #overwrite the free parameters with current parameter set
    dfPar_fit[ANALYSIS['dfParFitBool']] = dfPar_fit_MCMC
    dfPar_galpy             = scale_df_fit_to_galpy( ANALYSIS['dftype'],ro,vo,dfPar_fit)
    dfPar_phys              = scale_df_galpy_to_phys(ANALYSIS['dftype'],ro,vo,dfPar_galpy)
    #only use qDF parameters and ignore outlier model parameters:
    if ANALYSIS['dftype'] in [0,11,12]:
        dfPar_phys = dfPar_phys[0:5]
    else:
        sys.exit("Error in create_bestfit_reference_mockdata(): dftype = "+\
                 str(ANALYSIS['dftype'])+" is not defined.")
    print "qDF parameters: ",dfPar_phys

    #fiducial p_DF parameters in galpy units
    #load fiducial DF parameters
    dfParFid_fit   = ANALYSIS['dfParFid_fit']
    dfParFid_galpy = scale_df_fit_to_galpy( ANALYSIS['dftype'],ro,vo,dfParFid_fit)
    dfParFid_phys  = scale_df_galpy_to_phys(ANALYSIS['dftype'],ro,vo,dfParFid_galpy)
    #only use qDF parameters and ignore outlier model parameters:
    if ANALYSIS['dftype'] in [0,11,12]:
        dfParFid_phys = dfParFid_phys[0:5]
    else:
        sys.exit("Error in create_bestfit_reference_mockdata(): dftype = "+\
                 str(ANALYSIS['dftype'])+" is not defined.")
    print "Fiducial qDF: ",dfParFid_phys

    #only use qDF parameters and ignore outlier model parameters:
    dfParFitBool = ANALYSIS['dfParFitBool']
    if ANALYSIS['dftype'] in [0,11,12]:
        dfParFitBool = dfParFitBool[0:5]
    else:
        sys.exit("Error in create_bestfit_reference_mockdata(): dftype = "+\
                 str(ANALYSIS['dftype'])+" is not defined.")

    #_____setup galaxy_____
    print "* Setup Galaxy *"

    #potential:
    if not ANALYSIS['use_default_Delta'] or ANALYSIS['estimate_Delta']:
        sys.exit("Call to setup_Potential_and_ActionAngle_object needs to be account for Delta changes before proceeding.")
    pot, aA = setup_Potential_and_ActionAngle_object(ANALYSIS['pottype'],potPar_phys)

    mockdatafilename = mockdata_path+datasetname_reference+"/"+datasetname_reference+"_mockdata.sav"
    if (not os.path.exists(mockdatafilename)) or redo_everything:


        #set up distribution function (galpy units):
        if ANALYSIS['dftype'] in [0,11,12]:
            hr  = dfPar_galpy[0]
            sr  = dfPar_galpy[1]
            sz  = dfPar_galpy[2]
            hsr = dfPar_galpy[3]
            hsz = dfPar_galpy[4]
            qdf = quasiisothermaldf(
                    hr,sr,sz,hsr,hsz,
                    pot=pot,aA=aA,
                    cutcounter=True, 
                    ro=ro
                    )
        else:
            sys.exit("Error in create_bestfit_reference_mockdata(): dftype = "+\
                     str(ANALYSIS['dftype'])+" is not defined.")

        #selection function (physical units):
        sf = setup_SelectionFunction_object(ANALYSIS['sftype'],ANALYSIS['sfParEst_phys'],ro,df=qdf)


        #_____sample mock data_____

        if _DATATYPE == 1: #perfect mock data                                                                             

            #sample data:                                                                                         
            print "* Sample coordinates *"
            rs,zs,phis = sf.spatialSampleDF(nmock=_NSTARS,nrs=_N_SPAT,nzs=_N_SPAT,ngl_vel=_NGL_VEL,n_sigma=_N_SIGMA,_multi=_MULTI,quiet=False) #galpy units                                                                                         
            print "* Sample velocities *"
            vRs,vTs,vzs = sf.velocitySampleDF(rs,zs,_multi=_MULTI)   #galpy units

        else:
            sys.exit("Error in create_bestfit_reference_mockdata(): No reference mock data creation for _DATATYPE "+str(_DATATYPE))

        #create folder:
        if not os.path.exists(mockdata_path+datasetname_reference):
            os.makedirs(mockdata_path+datasetname_reference)                 

        #____Also save mock data_____
        save_pickles(mockdatafilename,
                        rs*_REFR0*ro,   #[kpc]
                        vRs*_REFV0*vo,  #[km/s]
                        phis,           #[deg]
                        vTs*_REFV0*vo,  #[km/s]
                        zs*_REFR0*ro,   #[kpc]
                        vzs*_REFV0*vo   #[kms]
                        )


        #_____write all analysis parameters to file_____
        write_RoadMapping_parameters(
                datasetname_reference,
                testname        = testname_original,
                datatype        = _DATATYPE,
                pottype         = ANALYSIS['pottype'],
                sftype          = ANALYSIS['sftype'],
                dftype          = 0,    #quasiisothermal df (Binney & McMillan 2011), no outlier model
                priortype       = 0,    #flat priors
                noStars         = _NSTARS,
                potParTrue_phys = potPar_phys,
                potParFitBool   = ANALYSIS['potParFitBool'],
                dfParTrue_phys  = dfPar_phys,
                dfParEst_phys   = dfPar_phys,
                dfParFitBool    = dfParFitBool,
                sfParTrue_phys  = ANALYSIS['sfParEst_phys'],
                mockdatapath    = mockdata_path,
                N_spatial       = 20,
                N_velocity      = 28,
                N_sigma         = 5.5,
                vT_galpy_max    = 1.5,
                MCMC_use_fidDF  = True,
                noMCMCsteps     = 400,
                noMCMCburnin    = 200
                )
    else:

            #_____load data from file_____
            savefile= open(mockdatafilename,'rb')
            rs    = pickle.load(savefile)/_REFR0/ro   #R   [kpc] or actually now galpy
            vRs   = pickle.load(savefile)/_REFV0/vo   #vR  [km/s] or actually now galpy
            phis  = pickle.load(savefile)             #phi [deg]
            vTs   = pickle.load(savefile)/_REFV0/vo   #vT  [km/s] or actually now galpy
            zs    = pickle.load(savefile)/_REFR0/ro   #z   [kpc] or actually now galpy
            vzs   = pickle.load(savefile)/_REFV0/vo   #vz  [km/s] or actually now galpy
            savefile.close()

    #-----------------------------------------------------------------------

    if with_actions:

        actiondatafilename = mockdata_path+datasetname_reference+"/"+datasetname_reference+"_mockdata_actions.sav"
        if (not os.path.exists(actiondatafilename)) or redo_everything:

            print "* Calculate actions *"
            actions = setup_data_actions(pot,aA,
                               ANALYSIS['dftype'],
                               rs,vRs,vTs,zs,vzs,   #data in galpy units
                               dfParFid_galpy,ro,
                               _MULTI)
            jr_data = actions[0,:] *_REFR0*ro*_REFV0*vo
            lz_data = actions[1,:] *_REFR0*ro*_REFV0*vo
            jz_data = actions[2,:] *_REFR0*ro*_REFV0*vo

            save_pickles(actiondatafilename,
                            jr_data,        #[kpc km/s]
                            lz_data,        #[kpc km/s]
                            jz_data,        #[kpc km/s]
                            actions[3:7,:]    #guiding star radius and frequencies in galpy units
                            )

        #--------------------------------

        if testname_original is None:
            originalactiondatafilename = mockdata_path+datasetname_original+"/"+datasetname_original+'_mockdata_actions.sav'
        else:
            originalactiondatafilename = mockdata_path+datasetname_original+"/"+datasetname_original+'_'+testname_original+'_mockdata_actions.sav'

        if originalmockdatafilename is None:
            originalmockdatafilename = mockdata_path+datasetname_original+"/"+datasetname_original+"_mockdata.sav"

        if (not os.path.exists(originalactiondatafilename)) or redo_everything:

            #_____load data from file_____
            savefile= open(originalmockdatafilename,'rb')
            rs    = pickle.load(savefile)/_REFR0/ro   #R   [kpc] or actually now galpy
            vRs   = pickle.load(savefile)/_REFV0/vo   #vR  [km/s] or actually now galpy
            phis  = pickle.load(savefile)             #phi [deg]
            vTs   = pickle.load(savefile)/_REFV0/vo   #vT  [km/s] or actually now galpy
            zs    = pickle.load(savefile)/_REFR0/ro   #z   [kpc] or actually now galpy
            vzs   = pickle.load(savefile)/_REFV0/vo   #vz  [km/s] or actually now galpy
            savefile.close()

            print "* Calculate actions for original data *"
            actions = setup_data_actions(pot,aA,
                               ANALYSIS['dftype'],
                               rs,vRs,vTs,zs,vzs,   #data in galpy units
                               dfParFid_galpy,ro,
                               _MULTI)
            jr_data = actions[0,:] *_REFR0*ro*_REFV0*vo
            lz_data = actions[1,:] *_REFR0*ro*_REFV0*vo
            jz_data = actions[2,:] *_REFR0*ro*_REFV0*vo

            save_pickles(originalactiondatafilename,
                            jr_data,        #[kpc km/s]
                            lz_data,        #[kpc km/s]
                            jz_data,        #[kpc km/s]
                            actions[3:7,:]    #guiding star radius and frequencies in galpy units
                            )

