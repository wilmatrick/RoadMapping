#_____import packages_____
#from __past__ import division
import sys
import math
import pandas
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
           2016-12-13 - Added datatype 5, which uses TGAS/RAVE data and samples from a covariance error matrix. - Trick (MPIA)
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
        sys.exit("Error in read_and_setup_data_set(), datatype=2: The code uses ANALYSIS['sunCoords_phys'] given as galactocentric cylindrical coordinates of the Sun for datatype=2. datatype=5 however uses ANALYSIS['sunCoords_phys'] given as galactocentric (x,y,z) coordinates. If I want to use datatye=2 again, the code should be changed analogous to datatype=5. For historical reasons I leave the original version of the code here, for the time being.")
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
                            Xgc_sun_kpc=Xsun_kpc,
                            Ygc_sun_kpc=Ysun_kpc,
                            Zgc_sun_kpc=Zsun_kpc,
                            vXgc_sun_kms=vXsun_kms,
                            vYgc_sun_kms=vYsun_kms,
                            vZgc_sun_kms=vZsun_kms
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
                            Xgc_sun_kpc=Xsun_kpc,
                            Ygc_sun_kpc=Ysun_kpc,
                            Zgc_sun_kpc=Zsun_kpc
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

    elif ANALYSIS['datatype'] == 5: 
        #========== RED CLUMP TGAS DATA WITH MEASUREMENT ERRORS; ==========
        #========== CORRELATIONS in RA, DEC, PM_RA, PM_DEC;
        #========== DISTANCE ERROR DIRECTLY IN DISTANCE

        # _____read data:_____
        datafilename = mockdatapath+datasetname+"/"+ datasetname+"_data_with_corr_errors.csv.gz"
        if not  os.path.exists(datafilename):
            sys.exit("Error in read_and_setup_data_set(): "+\
                     "No data file can be found, in "+\
                     datafilename)
        data = pandas.read_csv(datafilename)
        noStars = ANALYSIS['noStars']
        mean = numpy.zeros((noStars,6))
        std = numpy.zeros((noStars,6))
        cov = numpy.zeros((noStars,6,6))

        #6D coordinates:
        mean[:,0] = data['ra'         ].values #TGAS right ascension [deg]
        mean[:,1] = data['dec'        ].values #TGAS declination [deg]
        mean[:,2] = data['RC_dist_kpc'].values #red clump distance according to Bovy et al. (2014) [kpc]
        mean[:,3] = data['pmra'       ].values #TGAS RA proper motion [mas/yr] 
        mean[:,4] = data['pmdec'      ].values #TGAS DEC proper motion [mas/yr]
        mean[:,5] = data['HRV_x'      ].values #heliocentric radial velocity from RAVE [km/s]

        #standard deviation:
        std[:,0] = data['ra_error'         ].values / 3.6e6 #[deg]
        std[:,1] = data['dec_error'        ].values / 3.6e6 #[deg]
        std[:,2] = data['RC_dist_error_kpc'].values         #[kpc]
        std[:,3] = data['pmra_error'       ].values         #[mas/yr]
        std[:,4] = data['pmdec_error'      ].values         #[mas/yr]
        std[:,5] = data['eHRV_x'           ].values         #[km/s] 
        #ATTENTION: (RA,DEC) coordinate is in deg, error in mas, in TGAS

        #variances & correlations:
        cov[:,0,0] = std[:,0]**2
        cov[:,1,1] = std[:,1]**2
        cov[:,2,2] = std[:,2]**2
        cov[:,3,3] = std[:,3]**2
        cov[:,4,4] = std[:,4]**2
        cov[:,5,5] = std[:,5]**2

        cov[:,0,1] = data['ra_dec_corr'    ].values * std[:,0] * std[:,1]
        cov[:,0,2] = 0. #no correlation between RA & d
        cov[:,0,3] = data['ra_pmra_corr'   ].values * std[:,0] * std[:,3]
        cov[:,0,4] = data['ra_pmdec_corr'  ].values * std[:,0] * std[:,4]
        cov[:,0,5] = 0. #no correlation between RA & vlos

        cov[:,1,2] = 0. #no correlation between DEC & d
        cov[:,1,3] = data['dec_pmra_corr'  ].values * std[:,1] * std[:,3]
        cov[:,1,4] = data['dec_pmdec_corr' ].values * std[:,1] * std[:,4]
        cov[:,1,5] = 0. #no correlation between DEC & vlos

        cov[:,2,3] = 0. #no correlation between d & pmra
        cov[:,2,4] = 0. #no correlation between d & pmdec
        cov[:,2,5] = 0. #no correlation between d & vlos

        cov[:,3,4] = data['pmra_pmdec_corr'].values * std[:,3] * std[:,4]
        cov[:,3,5] = 0. #no correlation between pmra & vlos

        cov[:,4,5] = 0. #no correlation between pmdec & vlos

        cov[:,1,0] = cov[:,0,1]
        cov[:,2,0] = cov[:,0,2]
        cov[:,3,0] = cov[:,0,3]
        cov[:,4,0] = cov[:,0,4]
        cov[:,5,0] = cov[:,0,5]
        cov[:,2,1] = cov[:,1,2]
        cov[:,3,1] = cov[:,1,3]
        cov[:,4,1] = cov[:,1,4]
        cov[:,5,1] = cov[:,1,5]
        cov[:,3,2] = cov[:,2,3]
        cov[:,4,2] = cov[:,2,4]
        cov[:,5,2] = cov[:,2,5]
        cov[:,4,3] = cov[:,3,4]
        cov[:,5,3] = cov[:,3,5]
        cov[:,5,4] = cov[:,4,5]

        #_____sample error space:_____
        _N_ERROR_SAMPLES       = ANALYSIS['N_error_samples']
        random_seed_for_errors = ANALYSIS['random_seed_for_errors']

        # ... set random seed for MC convolution:
        numpy.random.seed(seed=random_seed_for_errors)

        # ... draw random numbers from multivariate Gaussian error 
        #     for each star:
        err = numpy.zeros((6,_N_ERROR_SAMPLES,noStars))
        for ii in range(noStars):
            err[:,:,ii] = numpy.random.multivariate_normal(mean[ii,:],cov[ii,:,:],_N_ERROR_SAMPLES).T
        
        
        # ... assign coordinates and unit conversion:
        deg2rad          = math.pi/180.
        ra_rad_err       = err[0,:,:] * deg2rad #from RA [deg]
        dec_rad_err      = err[1,:,:] * deg2rad #from DEC [deg]
        DM_mag_err       = 5.*numpy.log10(err[2,:,:]*1000.)-5.  #from dist [kpc]
        pmra_masyr_err  = err[3,:,:]    #from pm_ra [mas/yr]
        pmdec_masyr_err = err[4,:,:]    #from pm_dec [mas/yr]
        vlos_kms_err    = err[5,:,:]    #from vlos [km/s]

        # ... flatten everything 
        # (to prepare for trafo & action & likelihood calculation):
        ra_rad_err      =       ra_rad_err.flatten()
        dec_rad_err     =      dec_rad_err.flatten()
        DM_mag_err      =       DM_mag_err.flatten()
        pmra_masyr_err  =  pmra_masyr_err.flatten()
        pmdec_masyr_err = pmdec_masyr_err.flatten()
        vlos_kms_err    =     vlos_kms_err.flatten()

        # ... sun's coordinates:
        sunCoords_phys = ANALYSIS['sunCoords_phys']
        Xsun_kpc, Ysun_kpc, Zsun_kpc = sunCoords_phys[0], sunCoords_phys[1], sunCoords_phys[2]
        vXsun_kms, vYsun_kms, vZsun_kms = sunCoords_phys[3],sunCoords_phys[4], sunCoords_phys[5]

        # ... transform to galactocentric cylindrical coordinates
        # [R_kpc, phi_rad, z_kpc,vR_kms,vT_kms, vz_kms]
        out = radecDMvlospmradec_to_galcencyl(
                            ra_rad_err,
                            dec_rad_err,
                            DM_mag_err,
                            vlos_kms_err,
                            pmra_masyr_err,
                            pmdec_masyr_err,
                            quiet=True,
                            Xgc_sun_kpc=Xsun_kpc,
                            Ygc_sun_kpc=Ysun_kpc,
                            Zgc_sun_kpc=Zsun_kpc,
                            vXgc_sun_kms=vXsun_kms,
                            vYgc_sun_kms=vYsun_kms,
                            vZgc_sun_kms=vZsun_kms
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

        # ... perfect positions & unit conversion:
        ra_rad_obs       = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + mean[:,0] * deg2rad    #from RA [deg]
        dec_rad_obs      = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + mean[:,1] * deg2rad    #from DEC [deg]
        DM_mag_obs       = numpy.zeros((_N_ERROR_SAMPLES,noStars)) + 5.*numpy.log10(mean[:,2]*1000.)-5. #from d [kpc]
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
                            Xgc_sun_kpc=Xsun_kpc,
                            Ygc_sun_kpc=Ysun_kpc,
                            Zgc_sun_kpc=Zsun_kpc
                            )

        # ... rescale units:
        R_data   = out[0] /_REFR0          #[kpc  /_REFR0]
        phi_data = out[1] * 180. / math.pi #[deg]
        z_data   = out[2] /_REFR0          #[kpc  /_REFR0]

    else:
        sys.exit("Error in read_and_setup_data_set(): "+\
                 "data type "+str(ANALYSIS['datatype'])+" is not defined.")

    return R_data, vR_data, phi_data, vT_data, z_data, vz_data 
