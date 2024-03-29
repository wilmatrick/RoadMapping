#_____import packages_____
import sys
import math
import pickle
import os
import numpy
from read_RoadMapping_parameters import read_RoadMapping_parameters
from coord_trafo import radecDM_to_galcencyl, radecDMvlospmradec_to_galcencyl
from galpy.util import bovy_coords

def read_and_setup_data_set(datasetname,testname=None,mockdatapath=None):

    """
        NAME:
           read_and_setup_data_set
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started read_and_setup_data_set.py on the basis of BovyCode/py/read_and_setup_data_set.py - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    if mockdatapath is None: 
        sys.exit("Error in read_and_setup_data_set(): No mockdatapath is specified.")

    #_____read analysis parameters_____
    ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            mockdatapath=mockdatapath,
            print_to_screen=False
            )

    if ANALYSIS['datatype'] == 1:   
        #========== PERFECT MOCK DATA ==========

        # _____read mock data:_____
        datafilename = mockdatapath+datasetname+"/"+ datasetname+"_mockdata.sav"
        if   os.path.exists(datafilename): datafile = open(datafilename,'rb')
        else:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No mockdata file can be found, in "+\
                     datafilename)
        R_data   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vR_data  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        phi_data = pickle.load(datafile)          #[deg]
        vT_data  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        z_data   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vz_data  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        datafile.close()

        #_____test size of data set_____
        noStars = ANALYSIS['noStars']
        if len(R_data) != noStars:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "Number of stars in data set is ambiguous: \n"+\
                     datafilename+": "+str(len(R_data))+" vs. "+str(noStars))

    elif ANALYSIS['datatype'] == 2: 
        #========== MOCK DATA WITH MEASUREMENT ERRORS==========

        # _____read mock data:_____
        datafilename = mockdatapath+datasetname+"/"+ datasetname+"_mockdata_with_errors.sav"
        if   os.path.exists(datafilename): datafile = open(datafilename,'rb')
        else:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No mockdata file can be found, in "+\
                     datafilename)
        ra_rad          = pickle.load(datafile)
        dec_rad         = pickle.load(datafile)
        DM_mag          = pickle.load(datafile)
        vlos_kms        = pickle.load(datafile)
        pm_ra_masyr     = pickle.load(datafile)
        pm_dec_masyr    = pickle.load(datafile)
        datafile.close()

        #_____sample error space:_____

        noStars                = ANALYSIS['noStars']
        _N_ERROR_SAMPLES       = ANALYSIS['N_error_samples']
        random_seed_for_errors = ANALYSIS['random_seed_for_errors']

        # ... set random seed for MC convolution:
        numpy.random.seed(seed=random_seed_for_errors)
        
        # ... draw random numbers for random gaussian errors:
        eta = numpy.random.randn(6,_N_ERROR_SAMPLES,noStars)
        # Note: 
        # eta[0,:,:] is a (_N_ERROR_SAMPLES,noStars) shaped array. 
        # ra_rad[:,0], e.g., is a (noStars) shaped array and 
        # according to broadcasting rules, ra_rad[:,0] will be stretched to 
        # be a (_N_ERROR_SAMPLES,noStars) array as well, with the original 
        # ra_rad[:,0] in each row, when adding it to eta[0,:,:].

        # ... draw random points from Gaussian error around measurements:
        ra_rad_err       = eta[0,:,:] * ra_rad      [:,1] + ra_rad      [:,0]
        dec_rad_err      = eta[1,:,:] * dec_rad     [:,1] + dec_rad     [:,0]
        DM_mag_err       = eta[2,:,:] * DM_mag      [:,1] + DM_mag      [:,0]
        vlos_kms_err     = eta[3,:,:] * vlos_kms    [:,1] + vlos_kms    [:,0]
        pm_ra_masyr_err  = eta[4,:,:] * pm_ra_masyr [:,1] + pm_ra_masyr [:,0]
        pm_dec_masyr_err = eta[5,:,:] * pm_dec_masyr[:,1] + pm_dec_masyr[:,0]

        # ... flatten everything 
        # (to prepare for trafo & action & likelihood calculation):
        ra_rad_err       =       ra_rad_err.flatten()
        dec_rad_err      =      dec_rad_err.flatten()
        DM_mag_err       =       DM_mag_err.flatten()
        vlos_kms_err     =     vlos_kms_err.flatten()
        pm_ra_masyr_err  =  pm_ra_masyr_err.flatten()
        pm_dec_masyr_err = pm_dec_masyr_err.flatten()

        # ... sun's coordinates:
        sunCoords_phys = ANALYSIS['sunCoords_phys']
        Xsun_kpc, Ysun_kpc, Zsun_kpc = bovy_coords.cyl_to_rect(
                    sunCoords_phys[0],                  #R_sun [kpc]
                    sunCoords_phys[1] / 180. * math.pi, #phi [rad]
                    sunCoords_phys[2]                   #z [kpc]
                    )
        vXsun_kms, vYsun_kms, vZsun_kms = bovy_coords.cyl_to_rect_vec(
                    sunCoords_phys[3],                  #v_R [km/s]
                    sunCoords_phys[4],                  #v_T [km/s]
                    sunCoords_phys[5],                  #v_z [km/s]
                    sunCoords_phys[1] / 180. * math.pi, #phi [rad]
                    )

        # ... transform to galactocentric cylindrical coordinates
        # [R_kpc, phi_rad, z_kpc,vR_kms,vT_kms, vz_kms]
        out = radecDMvlospmradec_to_galcencyl(
                            ra_rad_err,
                            dec_rad_err,
                            DM_mag_err,
                            vlos_kms_err,
                            pm_ra_masyr_err,
                            pm_dec_masyr_err,
                            quiet=True,
                            Xsun_kpc=Xsun_kpc,
                            Ysun_kpc=Ysun_kpc,
                            Zsun_kpc=Zsun_kpc,
                            vXsun_kms=vXsun_kms,
                            vYsun_kms=vYsun_kms,
                            vZsun_kms=vZsun_kms
                            )

        # ... rescale units:
        #R_data   = out[0] /_REFR0   #[kpc  /_REFR0] #not used in Approx.
        #phi_data = out[1] * 180. / math.pi #[deg]   #not used in Approx.
        #z_data   = out[2] /_REFR0   #[kpc  /_REFR0] #not used in Approx.
        vR_data  = out[3] /_REFV0   #[km/s /_REFV0]
        vT_data  = out[4] /_REFV0   #[km/s /_REFV0]
        vz_data  = out[5] /_REFV0   #[km/s /_REFV0]

        # _____ERROR APPROXIMATION (Jo's idea)_____
        # Idea: Assume now that only (vR,vT,vz) are affected by the 
        # measurement errors and (R,phi,z) are measured/known exactly. 
        # The advantage of this approximation is, that we do not have to 
        # convolve the selection function with the distance error. There 
        # IS a distance error, but it only affects the velocities, so the 
        # likelihood normalisation stays the same as for perfect data.

        # ... perfect positions:
        ra_rad_obs       = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + ra_rad      [:,0]
        dec_rad_obs      = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + dec_rad     [:,0]
        DM_mag_obs       = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + DM_mag      [:,0]
        # flatten:
        ra_rad_obs       =       ra_rad_obs.flatten()
        dec_rad_obs      =      dec_rad_obs.flatten()
        DM_mag_obs       =       DM_mag_obs.flatten()

        # ... transform (ra,dec,DM) --> (R,phi,z):
        out = radecDM_to_galcencyl(
                            ra_rad_obs,
                            dec_rad_obs,
                            DM_mag_obs,
                            quiet=True,
                            Xsun_kpc=Xsun_kpc,
                            Ysun_kpc=Ysun_kpc,
                            Zsun_kpc=Zsun_kpc
                            )

        # ... rescale units:
        R_data   = out[0] /_REFR0          #[kpc  /_REFR0]
        phi_data = out[1] * 180. / math.pi #[deg]
        z_data   = out[2] /_REFR0          #[kpc  /_REFR0]

    elif ANALYSIS['datatype'] == 3:
        #========== PERFECT MOCK DATA, MARGINALIZATION OVER COORDINATE ==========

        ngl_marginal = ANALYSIS['ngl_marginal']

        # _____read mock data_____
        datafilename = mockdatapath+datasetname+"/"+ datasetname+"_mockdata.sav"
        if   os.path.exists(datafilename): datafile = open(datafilename,'rb')
        else:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No mockdata file can be found, in "+\
                     datafilename)
        R_data   = numpy.repeat(pickle.load(datafile)/_REFR0,ngl_marginal)   #[kpc  /_REFR0]
        vR_data  = numpy.repeat(pickle.load(datafile)/_REFV0,ngl_marginal)   #[km/s /_REFV0]
        phi_data = numpy.repeat(pickle.load(datafile)       ,ngl_marginal)   #[deg]
        vT_data  = numpy.repeat(pickle.load(datafile)/_REFV0,ngl_marginal)   #[km/s /_REFV0]
        z_data   = numpy.repeat(pickle.load(datafile)/_REFR0,ngl_marginal)   #[kpc  /_REFR0]
        vz_data  = numpy.repeat(pickle.load(datafile)/_REFV0,ngl_marginal)   #[km/s /_REFV0]
        datafile.close()

    elif ANALYSIS['datatype'] == 4: 
        #========== PERFECT MOCK DATA, MIX OF TWO POPULATIONS ==========

        #_____read main data set_____
        datafilename = mockdatapath+datasetname+"_MAIN/"+ datasetname+"_MAIN_mockdata.sav"
        if   os.path.exists(datafilename): datafile = open(datafilename,'rb')
        else:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No mockdata file can be found, in "+\
                     datafilename+"_MAIN")
        R_data_main   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vR_data_main  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        phi_data_main = pickle.load(datafile)          #[deg]
        vT_data_main  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        z_data_main   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vz_data_main  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        datafile.close()
        #_____read pollution data set_____
        datafilename = mockdatapath+datasetname+"_POLL/"+ datasetname+"_POLL_mockdata.sav"
        if   os.path.exists(datafilename): datafile = open(datafilename,'rb')
        else:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No mockdata file can be found, in "+\
                     datafilename+"_POLL")
        R_data_poll   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vR_data_poll  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        phi_data_poll = pickle.load(datafile)          #[deg]
        vT_data_poll  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        z_data_poll   = pickle.load(datafile)/_REFR0   #[kpc  /_REFR0]
        vz_data_poll  = pickle.load(datafile)/_REFV0   #[km/s /_REFV0]
        datafile.close()
        #_____add data sets_____
        R_data   = numpy.append(  R_data_main,  R_data_poll)
        z_data   = numpy.append(  z_data_main,  z_data_poll)
        phi_data = numpy.append(phi_data_main,phi_data_poll)
        vR_data  = numpy.append( vR_data_main, vR_data_poll)
        vT_data  = numpy.append( vT_data_main, vT_data_poll)
        vz_data  = numpy.append( vz_data_main, vz_data_poll)
        #_____test size of data set_____
        noStars = ANALYSIS['noStars']
        if len(R_data_main) != noStars[1] or len(R_data_poll) != noStars[2] or len(R_data) != noStars[0]:
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "Number of stars in data sets are ambiguous: \n"+\
                     datafilename+"_MAIN: "+str(len(R_data_main))+" vs. "+str(noStars[1])+"\n"+\
                     datafilename+"_POLL: "+str(len(R_data_poll))+" vs. "+str(noStars[2])+"\n"+\
                     "in total: "+str(noStars[0]))
    else:
        sys.exit("Error in read_and_setup_data_set(): "+\
                 "data type "+str(ANALYSIS['datatype'])+" is not defined.")

    return R_data, vR_data, phi_data, vT_data, z_data, vz_data 
