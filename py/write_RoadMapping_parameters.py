#_____import packages_____
import numpy
import sys
import os
from read_RoadMapping_parameters import read_RoadMapping_parameters
from outlier_model import scale_df_phys_to_fit, scale_df_fit_to_phys

def write_RoadMapping_parameters(datasetname,testname=None,
                            datatype=None,pottype=None,sftype=None,dftype=None,priortype=None,
                            noStars=None,
                            noMCMCsteps=None,noMCMCburnin=None,MCMC_use_fidDF=None,
                            N_spatial=None,N_velocity=None,N_sigma=None,vT_galpy_max=None,
                            potParTrue_phys=None,potParEst_phys=None,
                            potParMin_phys=None,potParMax_phys=None,
                            potParFitNo=None,potParFitBool=None,
                            dfParTrue_phys=None,dfParEst_phys=None,
                            dfParFid_phys=None,default_dfParFid=True,
                            dfParMin_fit=None,dfParMax_fit=None,
                            dfParFitNo=None,dfParFitBool=None,
                            sfParTrue_phys=None,sfParEst_phys=None,
                            marginalize_over=None,ngl_marginal=None,
                            N_error_samples=None,random_seed_for_errors=None,
                            errPar_obs=None,sunCoords_phys=None,
                            use_default_Delta=None,estimate_Delta=None,Delta_fixed=None,
                            aASG_accuracy=None,
                            mockdatapath='../data/',update=False,
                            datasetname_testname_to_be_copied=None
                            ):

    """
        NAME:
           write_RoadMapping_parameters
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-25 - Started write_RoadMapping_parameters on the basis of BovyCode/py/write_flexible_analysis_parameters.py - Trick (MPIA)
                      - Added selection function type 32, SPHERE + BOX + FREE CENTER - Trick (MPIA)
           2016-01-18 - Added pottype 5 and 51, Miyamoto-Nagai disk, Hernquist halo + Hernquist bulge for Elena D'Onghias Simulation
           2016-02-09 - Corrected bug. dfParFid_fit was not written in the file. Instead the dfParEst_fit was used. Now it's dfParFid_fit. - Trick (MPIA)
           2016-02-15 - Added pottype 6,7,61,71, Disk + Halo + Bulge potentials
           2016-04-11 - Upgrade to fileversion 2, which will be used for Dynamics Paper 2. This new version allows to control the actionAngleStaeckelGrid via the analysis input file.
           2016-09-22 - Added pottype 4,41 MWPotential2014 by Bovy (2015) - Trick (MPIA)
           2016-09-25 - Added pottype 42,421 MWPotential from galpy - Trick (MPIA)
           2016-12-13 - Added datatype 5, which uses TGAS/RAVE data and a covariance error matrix. - Trick (MPIA)
           2016-12-27 - Added priortype. - Trick (MPIA)
           2016-12-30 - Added pottype 8, 81 (for fitting to Gaia data). - Trick (MPIA)
           2017-01-03 - Added dftype. - Trick (MPIA)
           2017-01-09 - Added keyword datasetname_testname_to_be_copied to be able to copy an existing analysis file.
    """

    #analysis parameter file:
    if testname is None:
        filename = mockdatapath+datasetname+"/"+datasetname+"_analysis_parameters.txt"
    else:
        filename = mockdatapath+datasetname+"/"+datasetname+"_"+testname+"_analysis_parameters.txt"

    #if the file should only be updated, read existing file first:
    if update:
        if datasetname_testname_to_be_copied is not None:
            #if the file should be created as a copy & update from an existing but different analysis file:
            datasetname_previous = datasetname_testname_to_be_copied[0]
            testname_previous    = datasetname_testname_to_be_copied[1]
            out = read_RoadMapping_parameters(datasetname_previous,testname=testname_previous,mockdatapath=mockdatapath)
        elif os.path.exists(filename):
            #if this file should be updated
            out = read_RoadMapping_parameters(datasetname,testname=testname,mockdatapath=mockdatapath)
        else:
            sys.exit("Error in write_RoadMapping_parameters(): file "+\
                     filename+" does not exist and datasetname_testname_to_be_copied is not set.")


        

    #now open file to write:
    f = open(filename, 'w')

    #Reference scales:
    _REFR0 = 8.                 #[kpc], Reference radius 
    _REFV0 = 220.               #[km/s], Reference velocity

    #=================
    #=====GENERAL=====
    #=================

    if update and (datatype  is None): datatype  = out['datatype']
    if update and (pottype   is None): pottype   = out['pottype']
    if update and (sftype    is None): sftype    = out['sftype']
    if update and (dftype    is None): dftype    = out['dftype']
    if update and (priortype is None): priortype = out['priortype']

    f.write('# =========================================\n')
    f.write('# ===== MODEL PARAMETERS FOR ANALYSIS =====\n')
    f.write('# =========================================\n')
    f.write('# \n')
    f.write('# data set name: '+datasetname+'\n')
    if not testname is None:
        f.write('# test name    : '+testname+'\n')
    f.write('# \n')
    f.write('# ***** GENERAL SETUP *****\n')
    f.write('# * DATA & MODEL:\n')
    f.write('# \t\t data type / potential type / selection function type / df type / prior type / file version)\n')
    f.write('\t\t\t'+str(datatype)+'\t'+str(pottype)+'\t'+str(sftype)+'\t'+str(dftype)+'\t'+str(priortype)+'\t2\n')

    if   datatype == 1:  f.write('# data               type: 1 = perfect mock data\n')
    elif datatype == 2:  f.write('# data               type: 2 = mock data: observables with measurement errors\n')
    elif datatype == 3:  f.write('# data               type: 3 = perfect mock data + likelihood marginalization over one coordinate\n')
    elif datatype == 4:  f.write('# data               type: 4 = mix of two perfect data sets\n')
    elif datatype == 5:  f.write('# data               type: 5 = TGAS/RAVE red clump stars with covariance error matrix\n')
    else: sys.exit("Error in write_RoadMapping_parameters(): data type "+str(datatype)+" is not defined.")
    if   pottype  == 1:  f.write('# potential          type: 1 = isochrone\n')
    elif pottype  == 2:  f.write('# potential          type: 2 = 2-component KK-Staeckel (Batsleer & Dejonghe 1994) + Staeckel actions\n')
    elif pottype  == 21: f.write('# potential          type: 21 = 2-component KK-Staeckel (Batsleer & Dejonghe 1994) + StaeckelGrid actions\n')
    elif pottype  == 3:  f.write('# potential          type: 3 = MW-like (Bovy & Rix 2013) + Staeckel actions\n')
    elif pottype  == 31: f.write('# potential          type: 31 = MW-like (Bovy & Rix 2013) + StaeckelGrid actions\n')
    elif pottype  == 4:  f.write('# potential          type: 4 = MWPotential2014 by Bovy 2015 + Staeckel actions\n')    
    elif pottype  == 41: f.write('# potential          type: 41 = MWPotential2014 by Bovy 2015 + StaeckelGrid actions\n')   
    elif pottype  == 42: f.write('# potential          type: 42 = MWPotential from galpy + Staeckel actions\n')   
    elif pottype  == 421:f.write('# potential          type: 421 = MWPotential from galpy + StaeckelGrid actions\n')  
    elif pottype  == 5:  f.write("# potential          type: 5 = Miyamoto-Nagai disk, Hernquist halo & bulge (for D'Onghia simulation) + Staeckel actions\n")
    elif pottype  == 51: f.write("# potential          type: 51 = Miyamoto-Nagai disk, Hernquist halo & bulge (for D'Onghia simulation) + StaeckelGrid actions\n")
    elif pottype  == 6:  f.write("# potential          type: 6 = DoubleExponential disk, Hernquist halo & bulge (for D'Onghia simulation) + Staeckel actions\n")
    elif pottype  == 61: f.write("# potential          type: 61 = DoubleExponential disk, Hernquist halo & bulge (for D'Onghia simulation) + StaeckelGrid actions\n")
    elif pottype  == 7:  f.write("# potential          type: 7 = Miyamoto-Nagai disk, NFW halo & Hernquist bulge (galpy MWPotential like) + Staeckel actions\n")
    elif pottype  == 71: f.write("# potential          type: 71 = Miyamoto-Nagai disk, NFW halo & Hernquist bulge (galpy MWPotential like) + StaeckelGrid actions\n")
    elif pottype  == 8:  f.write("# potential          type: 8 = Miyamoto-Nagai disk, NFW halo & Hernquist bulge (to fit to Gaia data) + Staeckel actions\n")
    elif pottype  == 81: f.write("# potential          type: 81 = Miyamoto-Nagai disk, NFW halo & Hernquist bulge (to fit to Gaia data) + StaeckelGrid actions\n")
    else: sys.exit("Error in write_RoadMapping_parameters(): potential type "+str(pottype)+" is not defined.")
    if   sftype   == 1:  f.write('# selection function type: 1 = wedge (box completeness)\n')
    elif sftype   == 3:  f.write('# selection function type: 3 = sphere (box completeness)\n')
    elif sftype   == 31: f.write('# selection function type: 31 = sphere (incomplete in R & z)\n')
    elif sftype   == 32: f.write('# selection function type: 32 = sphere (box completeness + free center)\n')
    elif sftype   == 4:  f.write('# selection function type: 4 = incomplete shell\n')
    else: sys.exit("Error in write_RoadMapping_parameters(): selection function type "+str(sftype)+" is not defined.")
    if   dftype   == 0:  f.write('# distribution funct type: 0 = quasiisothermal df (Binney & McMillan 2011)\n')
    elif dftype   == 11: f.write('# distribution funct type: 11 = quasiisothermal df (Binney & McMillan 2011) + robust likelihood\n')
    elif dftype   == 12: f.write('# distribution funct type: 12 = quasiisothermal df (Binney & McMillan 2011) + halo outlier model\n')
    else: sys.exit("Error in write_RoadMapping_parameters(): distribution function type "+str(dftype)+" is not defined.")
    if   priortype == 0: f.write('# prior              type: 0 = flat priors on potential and log(DF) parameters.\n')
    elif priortype == 1: f.write('# prior              type: 1 = flat priors on parameters + Bovy & Rix (2013), eq. (41), prior on slope of rotation curve.\n')
    else: sys.exit("Error in write_RoadMapping_parameters(): prior type "+str(priortype)+" is not defined.")

    #MCMC accuracy parameters:
    if update and (noMCMCsteps    is None): noMCMCsteps  = out['noMCMCsteps']
    elif           noMCMCsteps    is None : noMCMCsteps  = 200
    if update and (noMCMCburnin   is None): noMCMCburnin = out['noMCMCburnin']
    elif           noMCMCburnin   is None : noMCMCburnin = 100
    if update and (MCMC_use_fidDF is None): MCMC_use_fidDF = int(out['MCMC_use_fidDF'])
    elif           MCMC_use_fidDF is None : MCMC_use_fidDF = 0
    else                                  : MCMC_use_fidDF = int(MCMC_use_fidDF)
    #Likelihood normalisation accuracy parameters:
    if update and (N_spatial      is None): N_spatial    = out['N_spatial']
    elif           N_spatial      is None : N_spatial    = 16
    if update and (N_velocity     is None): N_velocity   = out['N_velocity']
    elif           N_velocity     is None : N_velocity   = 24
    if update and (N_sigma        is None): N_sigma      = out['N_sigma']
    elif           N_sigma        is None : N_sigma      = 5.
    if update and (vT_galpy_max   is None): vT_galpy_max = out['vT_galpy_max']
    elif           vT_galpy_max   is None : vT_galpy_max = 1.5
    if             vT_galpy_max == 0.     : vT_galpy_max = 1.5
    #actionAngleStaeckel focal distance Delta:
    if update and (use_default_Delta is None): use_default_Delta = int(out['use_default_Delta'])
    elif           use_default_Delta is None : use_default_Delta = 1    #True
    else                                     : use_default_Delta = int(use_default_Delta)
    if update and (estimate_Delta    is None): estimate_Delta = int(out['estimate_Delta'])
    elif           estimate_Delta    is None : estimate_Delta = 0  #False
    else                                     : estimate_Delta = int(estimate_Delta)
    if update and (Delta_fixed is None)   : Delta_fixed = float(out['Delta_fixed'])
    elif           Delta_fixed is None    : Delta_fixed = 0.
    if pottype in numpy.array([1,2,21]) and (use_default_Delta != 1):
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "For pottype 1,2,21 it is required that use_default_Delta=True.")
    if (use_default_Delta == 1) and ((estimate_Delta != 0) or (Delta_fixed != 0.)):
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "If the default Staeckel Delta (0.45) is used, "+\
                 "then it is required that estimate_Delta=0 and Delta_fixed=0.")
    #actionAngleStaeckelGrid accuracy parameters:
    if pottype in numpy.array([21,31,41,421,51,61,71,81],dtype=int): use_aASG     = 1    #True for potentials that use the actionAngleStaeckelGrid
    else                                                 : use_aASG     = 0    #False
    if   update        and (aASG_accuracy is     None): aASG_accuracy = out['aASG_accuracy']
    elif use_aASG == 1 and (aASG_accuracy is     None): aASG_accuracy = numpy.array([5.,70.,40.,50.]) #Rmax=5.,nE=70,npsi=40,nLz=50
    elif use_aASG == 1 and (aASG_accuracy is not None): aASG_accuracy = numpy.array(aASG_accuracy)
    else                                              : aASG_accuracy = numpy.zeros(4)

    f.write('# \n')
    f.write('# * NUMERICAL PRECISION IN ANALYSIS:\n')
    #fileversion 0:
    #if vT_galpy_max == 1.5 or vT_galpy_max == 0.:   
    #    f.write('# \t\t N_spatial / N_velocity / N_sigma / does MCMC use fiducial DF? / # MC steps: total / # MC steps: burn-in:\n')
    #    f.write('\t\t\t'+str(N_spatial)+'\t'+str(N_velocity)+'\t'+str(N_sigma)+'\t'+str(MCMC_use_fidDF)+'\t'+str(noMCMCsteps)+'\t'+str(noMCMCburnin)+'\n')
    #else:
    #fileversion 1:
    #f.write('# \t\t N_spatial / N_velocity / N_sigma / vT_galpy_max / --- / ---\n')
    #f.write('\t\t\t'+str(N_spatial)+'\t'+str(N_velocity)+'\t'+str(N_sigma)+'\t'+str(vT_galpy_max)+'\t0\t0\n')
    #f.write('# \t\t Does MCMC use fiducial DF? / # MC steps: total / # MC steps: burn-in / --- / --- / ---\n')
    #f.write('\t\t\t'+str(MCMC_use_fidDF)+'\t'+str(noMCMCsteps)+'\t'+str(noMCMCburnin)+'\t0\t0\t0\n')
    #fileversion 2:
    f.write('# \t\t Normalisation: N_spatial / N_velocity / N_sigma / vT_galpy_max / --- / ---\n')
    f.write('\t\t\t'+str(N_spatial)+'\t'+str(N_velocity)+'\t'+str(N_sigma)+'\t'+str(vT_galpy_max)+'\t0\t0\n')
    f.write('# \t\t MCMC: Does MCMC use fiducial DF? / # MC steps: total / # MC steps: burn-in / --- / --- / ---\n')
    f.write('\t\t\t'+str(MCMC_use_fidDF)+'\t'+str(noMCMCsteps)+'\t'+str(noMCMCburnin)+'\t0\t0\t0\n')
    f.write('# \t\t ActionAngleStaeckel: Use fixed default Delta? / Estimate Delta for each potential? / new fixed Delta [galpy] / --- / --- / ---\n')
    f.write('\t\t\t'+str(use_default_Delta)+'\t'+str(estimate_Delta)+'\t'+str(Delta_fixed)+'\t0\t0\t0\n')
    f.write('# \t\t ActionAngleStaeckelGrid: Use StaeckelGrid? / Rmax [galpy] / nE / npsi / nLz / ---\n')
    f.write('\t\t\t'+str(use_aASG)+'\t'+str(aASG_accuracy[0])+'\t'+str(aASG_accuracy[1])+'\t'+str(aASG_accuracy[2])+'\t'+str(aASG_accuracy[3])+'\t0\n')

    #===================
    #=====DATA TYPE=====
    #===================

    if update and (noStars is None):
        noStars = out['noStars']
    elif noStars is None:
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "noStars must be set.")

    if datatype == 1:
        f.write('#\n')
        f.write('# ***** MOCK DATA: PERFECT *****\n')
        f.write('# * # stars / --- / --- / --- / --- / ---\n')
        f.write('\t\t\t'+str(noStars)+'\t0\t0\t0\t0\t0\n')

    elif datatype == 2:
        if update and (N_error_samples  is None): N_error_samples  = out['N_error_samples']
        elif           N_error_samples  is None : N_error_samples  = 200
        if update and (errPar_obs      is None): errPar_obs      = out['errPar_obs']
        elif           errPar_obs      is None : errPar_obs      = numpy.zeros(4)+numpy.nan   #default: measurement errors not known
        if update and (sunCoords_phys  is None): sunCoords_phys  = out['sunCoords_phys']
        elif           sunCoords_phys  is None :
            sys.exit("Error in write_RoadMapping_parameters(): "+\
             "position and velocity of sun (sunCoords_phys) must be set.")
        if update and (random_seed_for_errors is None): random_seed_for_errors = out['random_seed_for_errors']
        elif           random_seed_for_errors is None : random_seed_for_errors = numpy.random.random_integers(0,high=4294967295) 
        f.write('#\n')
        f.write('# ***** MOCK DATA: OBSERVABLES WITH MEASUREMENT ERRORS *****\n')
        f.write('# * # stars / # error MC samples / random seed for initialization / --- / --- / ---\n')
        f.write('\t\t\t'+str(noStars)+'\t'+str(N_error_samples)+'\t'+str(random_seed_for_errors)+'\t0\t0\t0\n')
        f.write('# * Global measurement errors:\n')
        f.write('#   d(distance modulus) [mag] / d(RA) & d(DEC) [rad] / d(v_los) [kms] / d(prop motion) [mas/yr] / --- / ---\n')
        f.write('\t\t\t'+str(errPar_obs[0])+'\t'+str(errPar_obs[1])+'\t'+str(errPar_obs[2])+'\t'+str(errPar_obs[3])+'\t0\t0\n')
        f.write('# * Position of the Sun:\n')
        f.write('#   R [kpc] / phi [deg] / z [kpc] / vR [km/s] / vT [km/s] / vz [km/s]\n')
        f.write('\t\t\t'+str(sunCoords_phys[0])+'\t'+str(sunCoords_phys[1])+'\t'+str(sunCoords_phys[2])+\
                    '\t'+str(sunCoords_phys[3])+'\t'+str(sunCoords_phys[4])+'\t'+str(sunCoords_phys[5])+'\n')

        sys.exit("Error in write_RoadMapping_parameters(), datatype=2: the code uses ANALYSIS['sunCoords_phys'] given as galactocentric cylindrical coordinates of the Sun for datatype=2. datatype=5 however uses ANALYSIS['sunCoords_phys'] given as galactocentric (x,y,z) coordinates. If I want to use datatye=2 again, the code should be changed analogous to datatype=5. For historical reasons I leave the original version of the code here, for the time being.")

    elif datatype == 3:
        marginal_coord = None
        if update and (marginalize_over is None): 
            marginal_coord = out['marginal_coord']
            if   marginal_coord == 1: marginalize_over = 'R' 
            elif marginal_coord == 2: marginalize_over = 'vR'
            elif marginal_coord == 4: marginalize_over = 'vT'
            elif marginal_coord == 5: marginalize_over = 'z' 
            elif marginal_coord == 6: marginalize_over = 'vz'
        f.write('#\n')
        f.write('# ***** MOCK DATA: PERFECT + MARGINALIZATION OVER '+marginalize_over+' ****\n')
        f.write('# * # stars / marginalize over: '+marginalize_over+' / N_GL / --- / --- / ---\n')
        if update and (ngl_marginal     is None): ngl_marginal   = out['ngl_marginal'] 
        else:                                     ngl_marginal   = N_velocity
        if   marginal_coord is not None: pass
        elif marginalize_over == 'R' : marginal_coord = 1
        elif marginalize_over == 'vR': marginal_coord = 2
        elif marginalize_over == 'vT': marginal_coord = 4
        elif marginalize_over == 'z' : marginal_coord = 5
        elif marginalize_over == 'vz': marginal_coord = 6
        else:
            sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "Coordinate "+marginalize_over+" to marginalize over is not known.")
        f.write('\t\t\t'+str(noStars)+'\t'+str(marginal_coord)+'\t'+str(ngl_marginal)+'\t0\t0\t0\n')

    elif datatype == 4:
        f.write('#\n')
        f.write('# ***** MOCK DATA: MIX OF TWO PERFECT DATA SETS *****\n')
        f.write('# * total # of stars / # in main data set / # in pollution data set / --- / --- / ---\n')
        f.write('\t\t\t'+str(noStars[0])+'\t'+str(noStars[1])+'\t'+str(noStars[2])+'\t0\t0\t0\n')

    elif datatype == 5:
        if update and (N_error_samples  is None): N_error_samples  = out['N_error_samples']
        elif           N_error_samples  is None : N_error_samples  = 200
        if update and (sunCoords_phys  is None): sunCoords_phys  = out['sunCoords_phys']
        elif           sunCoords_phys  is None :
            sys.exit("Error in write_RoadMapping_parameters(): "+\
             "position and velocity of sun (sunCoords_phys) in Galactocentric (x,y,z) frame must be set.")
        if update and (random_seed_for_errors is None): random_seed_for_errors = out['random_seed_for_errors']
        elif           random_seed_for_errors is None : random_seed_for_errors = numpy.random.random_integers(0,high=4294967295) 
        f.write('#\n')
        f.write('# ***** TGAS/RAVE DATA: RED CLUMP OBSERVABLES WITH COVARIANCE ERROR MATRIX *****\n')
        f.write('# * # stars / # error MC samples / random seed for initialization / --- / --- / ---\n')
        f.write('\t\t\t'+str(noStars)+'\t'+str(N_error_samples)+'\t'+str(random_seed_for_errors)+'\t0\t0\t0\n')
        f.write('# * Position of the Sun:\n')
        f.write('#   X_gc [kpc] / Y_gc [deg] / Z_gc [kpc] / vX_gc [km/s] / vY_gc [km/s] / vZ_gc [km/s]\n')
        f.write('\t\t\t'+str(sunCoords_phys[0])+'\t'+str(sunCoords_phys[1])+'\t'+str(sunCoords_phys[2])+\
                    '\t'+str(sunCoords_phys[3])+'\t'+str(sunCoords_phys[4])+'\t'+str(sunCoords_phys[5])+'\n')

    else:
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "data type "+str(datatype)+" is not defined.")
    #datatype 1: perfect
    #datatype 2: measurement errors
    #datatype 3: perfect + marginalize
    #datatype 4: perfect + mix of two sets
    #datatype 5: TGAS/RAVE data + covariance error matrix

    #===================
    #=====POTENTIAL=====
    #===================

    #____reload previous parameters (in case of update)_____
    if update:
        if potParTrue_phys is None: potParTrue_phys = out['potParTrue_phys']
        if potParEst_phys  is None: potParEst_phys  = out['potParEst_phys' ]
        if potParFitBool   is None: potParFitBool   = out['potParFitBool']
        else:                       potParFitBool   = numpy.array(potParFitBool,dtype=bool)
        #lower fit limit:
        if potParMin_phys is not None: potParMin_phys_new = numpy.array(potParMin_phys,copy=True)
        else:                          potParMin_phys_new = None
        potParMin_phys     = out['potParMin_phys' ]
        if potParMin_phys_new is not None:
            if numpy.sum(potParFitBool) == len(potParMin_phys_new):
                potParMin_phys[potParFitBool] = potParMin_phys_new
            elif len(potParMin_phys_new) == len(potParTrue_phys):
                potParMin_phys = potParMin_phys_new
        #upper fit limit:
        if potParMax_phys is not None: potParMax_phys_new = numpy.array(potParMax_phys,copy=True)
        else:                          potParMax_phys_new = None
        potParMax_phys     = out['potParMax_phys' ]
        if potParMax_phys_new is not None:
            if numpy.sum(potParFitBool) == len(potParMax_phys_new):
                potParMax_phys[potParFitBool] = potParMax_phys_new
            elif len(potParMax_phys_new) == len(potParTrue_phys):
                potParMax_phys == potParMax_phys_new
        #number of fit points:
        if potParFitNo is not None: potParFitNo_new = numpy.array(potParFitNo,copy=True)
        else:                       potParFitNo_new = None
        potParFitNo     = out['potParFitNo']
        if potParFitNo_new is not None:
            if numpy.sum(potParFitBool) == len(potParFitNo_new):
                potParFitNo[potParFitBool] = potParFitNo_new
            elif len(potParFitNo_new) == len(potParTrue_phys):
                potParFitNo = potParFitNo_new


    #_____default values_____
    #true values and estimate:
    if (potParTrue_phys is None) and (potParEst_phys is None):
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "either potParTrue_phys or potParEst_phys must be set.")
    if potParTrue_phys is None: potParTrue_phys = numpy.zeros(len(potParEst_phys))
    if potParEst_phys  is None: potParEst_phys  = numpy.array(potParTrue_phys,copy=True)

    #which parameters will be fitted:
    if (potParFitNo is None) and (potParFitBool is None):
        potParFitNo   = numpy.ones(len(potParEst_phys),dtype=int)
        potParFitBool = numpy.zeros(len(potParFitNo),dtype=bool)
    if potParFitNo   is None:
        potParFitBool = numpy.array(potParFitBool,dtype=bool)
        potParFitNo = numpy.ones(len(potParFitBool),dtype=int)
        potParFitNo[potParFitBool] = 3
    if potParFitBool is None: potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)
    else:                     potParFitBool   = numpy.array(potParFitBool,dtype=bool)
    
    #default fitting range:
    Min_default, Max_default = False, False
    if potParMin_phys is None: 
        potParMin_phys = numpy.array(potParEst_phys,copy=True)
        Min_default = True
    if potParMax_phys is None: 
        potParMax_phys = numpy.array(potParEst_phys,copy=True)
        Max_default = True
    for ii in range(len(potParFitNo)):
        if potParFitBool[ii]:
            if Min_default: potParMin_phys[ii] = 0.5*potParEst_phys[ii]   # 50% of best estimate
            if Max_default: potParMax_phys[ii] = 1.5*potParEst_phys[ii]


    if pottype == 1:
        #ISOCHRONE
        #potPar = [R0_kpc,vc_kms,b_kpc]
        f.write('#\n')
        f.write('# ***** POTENTIAL: ISOCHRONE *****\n')
        f.write('# \t\t true value / estimate / --- / fit min / fit max / # grid points\n')
        f.write('# R_0      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[0])+'\t'+str(potParEst_phys[0])+'\t0\t'+\
                    str(potParMin_phys[0])+'\t'+str(potParMax_phys [0])+'\t'+str(potParFitNo [0])+'\n')
        f.write('# v_c(R_0) [km/s] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[1])+'\t'+str(potParEst_phys[1])+'\t0\t'+\
                    str(potParMin_phys[1])+'\t'+str(potParMax_phys [1])+'\t'+str(potParFitNo [1])+'\n')
        f.write('# b        [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[2])+'\t'+str(potParEst_phys[2])+'\t0\t'+\
                    str(potParMin_phys[2])+'\t'+str(potParMax_phys [2])+'\t'+str(potParFitNo [2])+'\n')

    elif pottype == 2 or pottype == 21:
        #2-COMPONENT KUZMIN-KUTUZOV-STAECKEL-POTENTIAL (Batsleer & Dejonghe 1994)
        f.write('#\n')
        f.write('# ***** POTENTIAL: 2-COMPONENT KK STAECKEL (Batsleer & Dejonghe 1994)*****\n')
        f.write('# \t\t true value / estimate / --- / fit min / fit max / # grid points\n')
        f.write('# R_0      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[0])+'\t'+str(potParEst_phys[0])+'\t0\t'+\
                    str(potParMin_phys[0])+'\t'+str(potParMax_phys [0])+'\t'+str(potParFitNo [0])+'\n')
        f.write('# v_c(R_0) [km/s] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[1])+'\t'+str(potParEst_phys[1])+'\t0\t'+\
                    str(potParMin_phys[1])+'\t'+str(potParMax_phys [1])+'\t'+str(potParFitNo [1])+'\n')
        f.write('# Delta  [galpy?] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[2])+'\t'+str(potParEst_phys[2])+'\t0\t'+\
                    str(potParMin_phys[2])+'\t'+str(potParMax_phys [2])+'\t'+str(potParFitNo [2])+'\n')
        f.write('# (a/c)_Disk      =\n')
        f.write('\t\t\t'+str(potParTrue_phys[3])+'\t'+str(potParEst_phys[3])+'\t0\t'+\
                    str(potParMin_phys[3])+'\t'+str(potParMax_phys [3])+'\t'+str(potParFitNo [3])+'\n')
        f.write('# (a/c)_Halo      =\n')
        f.write('\t\t\t'+str(potParTrue_phys[4])+'\t'+str(potParEst_phys[4])+'\t0\t'+\
                    str(potParMin_phys[4])+'\t'+str(potParMax_phys [4])+'\t'+str(potParFitNo [4])+'\n')
        f.write('# k               =\n')
        f.write('\t\t\t'+str(potParTrue_phys[5])+'\t'+str(potParEst_phys[5])+'\t0\t'+\
                    str(potParMin_phys[5])+'\t'+str(potParMax_phys [5])+'\t'+str(potParFitNo [5])+'\n')

    elif pottype == 3 or pottype == 31:
        #MW-LIKE POTENTIAL (Bovy & Rix 2013)
        f.write('#\n')
        f.write('# ***** POTENTIAL: MW-LIKE (Bovy & Rix 2013)*****\n')
        f.write('# \t\t true value / estimate / --- / fit min / fit max / # grid points\n')
        f.write('# R_0      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[0])+'\t'+str(potParEst_phys[0])+'\t0\t'+\
                    str(potParMin_phys[0])+'\t'+str(potParMax_phys [0])+'\t'+str(potParFitNo [0])+'\n')
        f.write('# v_c(R_0) [km/s] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[1])+'\t'+str(potParEst_phys[1])+'\t0\t'+\
                    str(potParMin_phys[1])+'\t'+str(potParMax_phys [1])+'\t'+str(potParFitNo [1])+'\n')
        f.write('# R_d      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[2])+'\t'+str(potParEst_phys[2])+'\t0\t'+\
                    str(potParMin_phys[2])+'\t'+str(potParMax_phys [2])+'\t'+str(potParFitNo [2])+'\n')
        f.write('# z_h      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[3])+'\t'+str(potParEst_phys[3])+'\t0\t'+\
                    str(potParMin_phys[3])+'\t'+str(potParMax_phys [3])+'\t'+str(potParFitNo [3])+'\n')
        f.write('# f_h             =\n')
        f.write('\t\t\t'+str(potParTrue_phys[4])+'\t'+str(potParEst_phys[4])+'\t0\t'+\
                    str(potParMin_phys[4])+'\t'+str(potParMax_phys [4])+'\t'+str(potParFitNo [4])+'\n')
        f.write('# d(ln(v_c)) / d(ln(r)) =\n')
        f.write('\t\t\t'+str(potParTrue_phys[5])+'\t'+str(potParEst_phys[5])+'\t0\t'+\
                    str(potParMin_phys[5])+'\t'+str(potParMax_phys [5])+'\t'+str(potParFitNo [5])+'\n')

    elif pottype in numpy.array([4,41,42,421],dtype=int):
        #MWPotential2014 FROM GALPY (Bovy 2015) (4+41), or MWPotential FROM GALPY (42+421)
        f.write('#\n')
        if pottype in numpy.array([4,41],dtype=int):
            f.write('# ***** POTENTIAL: MWPotential2014 (Bovy 2015; galpy)*****\n')
        elif pottype in numpy.array([42,421],dtype=int):
            f.write('# ***** POTENTIAL: MWPotential (galpy)*****\n')
        f.write('# \t\t true value / estimate / --- / fit min / fit max / # grid points\n')
        f.write('# R_0      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[0])+'\t'+str(potParEst_phys[0])+'\t0\t'+\
                    str(potParMin_phys[0])+'\t'+str(potParMax_phys [0])+'\t'+str(potParFitNo [0])+'\n')
        f.write('# v_c(R_0) [km/s] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[1])+'\t'+str(potParEst_phys[1])+'\t0\t'+\
                    str(potParMin_phys[1])+'\t'+str(potParMax_phys [1])+'\t'+str(potParFitNo [1])+'\n')

    elif pottype in numpy.array([5,6,7,8,51,61,71,81],dtype=int):
        f.write('#\n')
        if pottype in numpy.array([5,51],dtype=int):
            #MIYAMOTO-NAGAI DISK + HERNQUIST HALO + HERNQUIST BULGE (for Elena D'Onghia simulation)
            f.write("# ***** POTENTIAL: MIYAMOTO-NAGAI DISK + HERNQUIST HALO + BULGE (for D'Onghia simulation)*****\n")
            scalelength,scaleheight='a_disk','b_disk'
        elif pottype in numpy.array([6,61],dtype=int):
            #DOUBLE EXPONENTIAL DISK + HERNQUIST HALO + HERNQUIST BULGE (for Elena D'Onghia simulation)
            f.write("# ***** POTENTIAL: DOUBLE EXPONENTIAL DISK + HERNQUIST HALO + BULGE (for D'Onghia simulation)*****\n")
            scalelength,scaleheight='hr_disk','hz_disk'
        elif pottype in numpy.array([7,71],dtype=int):
            #MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE (analytic MWPotential(2014)-like potential)
            f.write("# ***** POTENTIAL: MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE (galpy MWPotential-like)*****\n")
            scalelength,scaleheight='a_disk','b_disk'
        elif pottype in numpy.array([8,81],dtype=int):
            #MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE (for fitting to Gaia data)
            f.write("# ***** POTENTIAL: MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE (galpy MWPotential-like)*****\n")
            scalelength,scaleheight='a_disk','b_disk'
        f.write('# \t\t true value / estimate / --- / fit min / fit max / # grid points\n')
        f.write('# R_0      [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[0])+'\t'+str(potParEst_phys[0])+'\t0\t'+\
                    str(potParMin_phys[0])+'\t'+str(potParMax_phys [0])+'\t'+str(potParFitNo [0])+'\n')
        f.write('# v_c(R_0) [km/s] =\n')
        f.write('\t\t\t'+str(potParTrue_phys[1])+'\t'+str(potParEst_phys[1])+'\t0\t'+\
                    str(potParMin_phys[1])+'\t'+str(potParMax_phys [1])+'\t'+str(potParFitNo [1])+'\n')
        f.write('# '+scalelength+'   [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[2])+'\t'+str(potParEst_phys[2])+'\t0\t'+\
                    str(potParMin_phys[2])+'\t'+str(potParMax_phys [2])+'\t'+str(potParFitNo [2])+'\n')
        f.write('# '+scaleheight+'   [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[3])+'\t'+str(potParEst_phys[3])+'\t0\t'+\
                    str(potParMin_phys[3])+'\t'+str(potParMax_phys [3])+'\t'+str(potParFitNo [3])+'\n')
        f.write('# f_halo          =\n')
        f.write('\t\t\t'+str(potParTrue_phys[4])+'\t'+str(potParEst_phys[4])+'\t0\t'+\
                    str(potParMin_phys[4])+'\t'+str(potParMax_phys [4])+'\t'+str(potParFitNo [4])+'\n')
        f.write('# a_halo   [kpc]  =\n')
        f.write('\t\t\t'+str(potParTrue_phys[5])+'\t'+str(potParEst_phys[5])+'\t0\t'+\
                    str(potParMin_phys[5])+'\t'+str(potParMax_phys [5])+'\t'+str(potParFitNo [5])+'\n')
        if pottype in numpy.array([8,81],dtype=int):
            f.write('# M_bulge [10^10Msun] =\n')
            f.write('\t\t\t'+str(potParTrue_phys[6])+'\t'+str(potParEst_phys[6])+'\t0\t'+\
                        str(potParMin_phys[6])+'\t'+str(potParMax_phys [6])+'\t'+str(potParFitNo [6])+'\n')
            f.write('# a_bulge  [kpc]  =\n')
            f.write('\t\t\t'+str(potParTrue_phys[7])+'\t'+str(potParEst_phys[7])+'\t0\t'+\
                        str(potParMin_phys[7])+'\t'+str(potParMax_phys [7])+'\t'+str(potParFitNo [7])+'\n')

    else:
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "potential type "+str(pottype)+" is not defined.")


    #=============
    #=====QDF=====
    #=============

    #dfPar_phys = [hr_kpc,sr_kms,sz_kms,hsr_kpc,hsz_kpc]

    #____reload previous parameters (in case of update)_____
    if update:
        if dfParTrue_phys is None: dfParTrue_phys = out['dfParTrue_phys']
        if dfParEst_phys  is None: dfParEst_phys  = out['dfParEst_phys' ]
        if dfParFitBool   is None: dfParFitBool   = out['dfParFitBool']
        else:                      dfParFitBool   = numpy.array(dfParFitBool,dtype=bool)
        #lower fit limit:
        if dfParMin_fit is not None: dfParMin_fit_new = numpy.array(dfParMin_fit,copy=True)
        else:                        dfParMin_fit_new = None
        dfParMin_fit     = out['dfParMin_fit' ]
        if dfParMin_fit_new is not None:
            if numpy.sum(dfParFitBool) == len(dfParMin_fit_new):
                dfParMin_fit[dfParFitBool] = dfParMin_fit_new
            elif len(dfParMin_fit_new) == len(dfParTrue_phys):
                dfParMin_fit == dfParMin_fit_new
        #upper fit limit:
        if dfParMax_fit is not None: dfParMax_fit_new = numpy.array(dfParMax_fit,copy=True)
        else:                        dfParMax_fit_new = None
        dfParMax_fit     = out['dfParMax_fit' ]
        if dfParMax_fit_new is not None:
            if numpy.sum(dfParFitBool) == len(dfParMax_fit_new):
                dfParMax_fit[dfParFitBool] = dfParMax_fit_new
            elif len(dfParMax_fit_new) == len(dfParTrue_phys):
                dfParMax_fit == dfParMax_fit_new
        #number of fit points:
        if dfParFitNo is not None: dfParFitNo_new = numpy.array(dfParFitNo,copy=True)
        else:                      dfParFitNo_new = None
        dfParFitNo     = out['dfParFitNo']
        if dfParFitNo_new is not None:
            if numpy.sum(dfParFitBool) == len(dfParFitNo_new):
                dfParFitNo[dfParFitBool] = dfParFitNo_new
            elif len(dfParFitNo_new) == len(dfParTrue_phys):
                dfParFitNo == dfParFitNo_new
        #fiducial qdf:
        if default_dfParFid:
            dfParFid_phys = None    #reset fiducial qdf
        else:
            if dfParFid_phys is None: dfParFid_phys  = out['dfParFid_phys ']      

    #_____default values_____
    #true values and estimate:
    if (dfParTrue_phys is None) and (dfParEst_phys is None): 
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "either dfParTrue_phys or dfParEst_phys must be set.")
    if dfParTrue_phys is None: dfParTrue_phys = numpy.zeros(len(dfParEst_phys))
    else:                      dfParTrue_phys = numpy.array(dfParTrue_phys,dtype=float)
    if dfParEst_phys  is None: dfParEst_phys  = numpy.array(dfParTrue_phys,copy=True)
    else:                      dfParEst_phys  = numpy.array(dfParEst_phys ,dtype=float)
    dfParEst_fit   = scale_df_phys_to_fit(dftype,dfParEst_phys)
    dfParTrue_fit  = scale_df_phys_to_fit(dftype,dfParTrue_phys)

    #which parameters will be fitted:
    if (dfParFitNo is None) and (dfParFitBool is None):
        dfParFitNo   = numpy.ones(len(dfParEst_phys),dtype=int)
        dfParFitBool = numpy.zeros(len(dfParFitNo),dtype=bool)
    if dfParFitNo   is None:
        dfParFitBool = numpy.array(dfParFitBool,dtype=bool)
        dfParFitNo = numpy.ones(len(dfParFitBool),dtype=int)
        dfParFitNo[dfParFitBool] = 3
    if dfParFitBool is None: dfParFitBool   = numpy.array((dfParFitNo > 1),dtype=bool)
    else:                    dfParFitBool   = numpy.array(dfParFitBool,dtype=bool)

    #default fitting ranges:
    Min_default, Max_default = False, False
    if dfParMin_fit is None: 
        dfParMin_phys = numpy.array(dfParEst_phys,copy=True)
        Min_default = True
    else:
        dfParMin_phys = scale_df_fit_to_phys(dftype,dfParMin_fit)
    if dfParMax_fit is None: 
        dfParMax_phys = numpy.array(dfParEst_phys,copy=True)
        Max_default = True
    else: 
        dfParMax_phys = scale_df_fit_to_phys(dftype,dfParMax_fit)
    for ii in range(len(dfParFitNo)):
        if dfParFitBool[ii]:
            if Min_default: dfParMin_phys[ii] = 0.5*dfParEst_phys[ii]   # 50% of best estimate, all parameters > 0
            if Max_default: dfParMax_phys[ii] = 1.5*dfParEst_phys[ii]          
                
    if Min_default: dfParMin_fit  = scale_df_phys_to_fit(dftype,dfParMin_phys)
    if Max_default: dfParMax_fit  = scale_df_phys_to_fit(dftype,dfParMax_phys)

    #fiducial qdf parameters:
    if dfParFid_phys is None: 
        dfParFid_fit = 0.5 * (dfParMin_fit + dfParMax_fit)  #default set by fitting range
        dfParFid_phys = scale_df_fit_to_phys(dftype,dfParFid_fit)
    else:
        dfParFid_phys = numpy.array(dfParFid_phys,dtype=float)
        dfParFid_fit = scale_df_phys_to_fit(dftype,dfParFid_phys)

    if dftype in [0,11,12]:
        f.write('#\n')
        if dftype == 0:
            #QUASI-ISOTHERMAL DF:
            #dfPar = [ln_hr,ln_sr,ln_sz,ln_hsr,ln_hsz]
            f.write('# ***** QUASI-ISOTHERMAL DISTRIBUTION FUNCTION *****\n')
        elif dftype == 11:
            #QUASI-ISOTHERMAL DF + ROBUST LIKELIHOOD:
            #dfPar = [ln_hr,ln_sr,ln_sz,ln_hsr,ln_hsz]
            f.write('# ***** QUASI-ISOTHERMAL DISTRIBUTION FUNCTION + ROBUST LIKELIHOOD *****\n')
        elif dftype == 12:
            #QUASI-ISOTHERMAL DF + HALO OUTLIER MODEL:
            #dfPar = [ln_hr,ln_sr,ln_sz,ln_hsr,ln_hsz,p_out,ln_sv_out,ln_hsv_out]
            f.write('# ***** QUASI-ISOTHERMAL DISTRIBUTION FUNCTION + HALO OUTLIER MODEL *****\n')

        f.write('# * physical coordinates:\n')
        f.write('# \t\t true value / estimate / fiducial / fit min / fit max / # grid points\n')
        f.write('# \t\t                        (not used)(not used)(not used)   (not used)\n')
        f.write('#   h_R       [kpc]  =\n')
        f.write('\t\t\t'+str(dfParTrue_phys[0])+'\t'+str(dfParEst_phys[0])+'\t'+str(dfParFid_phys[0])+'\t'+\
                         str(dfParMin_phys [0])+'\t'+str(dfParMax_phys[0])+'\t'+str(dfParFitNo   [0])+'\n')
        f.write('#   sigma_R   [km/s] =\n')
        f.write('\t\t\t'+str(dfParTrue_phys[1])+'\t'+str(dfParEst_phys[1])+'\t'+str(dfParFid_phys[1])+'\t'+\
                         str(dfParMin_phys [1])+'\t'+str(dfParMax_phys[1])+'\t'+str(dfParFitNo   [1])+'\n')
        f.write('#   sigma_z   [km/s] =\n')
        f.write('\t\t\t'+str(dfParTrue_phys[2])+'\t'+str(dfParEst_phys[2])+'\t'+str(dfParFid_phys[2])+'\t'+\
                         str(dfParMin_phys [2])+'\t'+str(dfParMax_phys[2])+'\t'+str(dfParFitNo   [2])+'\n')
        f.write('#   h_sigma_R [kpc]  =\n')
        f.write('\t\t\t'+str(dfParTrue_phys[3])+'\t'+str(dfParEst_phys[3])+'\t'+str(dfParFid_phys[3])+'\t'+\
                         str(dfParMin_phys [3])+'\t'+str(dfParMax_phys[3])+'\t'+str(dfParFitNo   [3])+'\n')
        f.write('#   h_sigma_z [kpc]  =\n')
        f.write('\t\t\t'+str(dfParTrue_phys[4])+'\t'+str(dfParEst_phys[4])+'\t'+str(dfParFid_phys[4])+'\t'+\
                         str(dfParMin_phys [4])+'\t'+str(dfParMax_phys[4])+'\t'+str(dfParFitNo   [4])+'\n')

        if dftype == 12:
            f.write('#   p_out =\n')
            f.write('\t\t\t'+str(dfParTrue_phys[5])+'\t'+str(dfParEst_phys[5])+'\t'+str(dfParFid_phys[5])+'\t'+\
                             str(dfParMin_phys [5])+'\t'+str(dfParMax_phys[5])+'\t'+str(dfParFitNo   [5])+'\n')
            f.write('#   sigma_vel_out [km/s] =\n')
            f.write('\t\t\t'+str(dfParTrue_phys[6])+'\t'+str(dfParEst_phys[6])+'\t'+str(dfParFid_phys[6])+'\t'+\
                             str(dfParMin_phys [6])+'\t'+str(dfParMax_phys[6])+'\t'+str(dfParFitNo   [6])+'\n')
            f.write('#   h_sigma_vel_out [kpc] =\n')
            f.write('\t\t\t'+str(dfParTrue_phys[7])+'\t'+str(dfParEst_phys[7])+'\t'+str(dfParFid_phys[7])+'\t'+\
                             str(dfParMin_phys [7])+'\t'+str(dfParMax_phys[7])+'\t'+str(dfParFitNo   [7])+'\n')
        


        f.write('# * galpy coordinates:\n')
        f.write('# \t\t true value / estimate / fiducial / fit min / fit max / # grid points\n')
        f.write('# \t\t (not used)  (not used)   \n')
        f.write('#   ln( h_R     [_REFR0])   =\n')
        f.write('\t\t\t'+str(dfParTrue_fit[0])+'\t'+str(dfParEst_fit[0])+'\t'+str(dfParFid_fit[0])+'\t'+\
                         str(dfParMin_fit [0])+'\t'+str(dfParMax_fit[0])+'\t'+str(dfParFitNo  [0])+'\n')
        f.write('#   ln( sigma_R [_REFV0])   =\n')
        f.write('\t\t\t'+str(dfParTrue_fit[1])+'\t'+str(dfParEst_fit[1])+'\t'+str(dfParFid_fit[1])+'\t'+\
                         str(dfParMin_fit [1])+'\t'+str(dfParMax_fit[1])+'\t'+str(dfParFitNo  [1])+'\n')
        f.write('#   ln( sigma_z [_REFV0])   =\n')
        f.write('\t\t\t'+str(dfParTrue_fit[2])+'\t'+str(dfParEst_fit[2])+'\t'+str(dfParFid_fit[2])+'\t'+\
                         str(dfParMin_fit [2])+'\t'+str(dfParMax_fit[2])+'\t'+str(dfParFitNo  [2])+'\n')
        f.write('#   ln( h_sigma_R [_REFR0]) =\n')
        f.write('\t\t\t'+str(dfParTrue_fit[3])+'\t'+str(dfParEst_fit[3])+'\t'+str(dfParFid_fit[3])+'\t'+\
                         str(dfParMin_fit [3])+'\t'+str(dfParMax_fit[3])+'\t'+str(dfParFitNo  [3])+'\n')
        f.write('#   ln( h_sigma_z [_REFR0]) =\n')
        f.write('\t\t\t'+str(dfParTrue_fit[4])+'\t'+str(dfParEst_fit[4])+'\t'+str(dfParFid_fit[4])+'\t'+\
                         str(dfParMin_fit [4])+'\t'+str(dfParMax_fit[4])+'\t'+str(dfParFitNo  [4])+'\n')

        if dftype == 12:
            f.write('#   p_out =\n')
            f.write('\t\t\t'+str(dfParTrue_fit[5])+'\t'+str(dfParEst_fit[5])+'\t'+str(dfParFid_fit[5])+'\t'+\
                             str(dfParMin_fit [5])+'\t'+str(dfParMax_fit[5])+'\t'+str(dfParFitNo  [5])+'\n')
            f.write('#   ln( sigma_vel_out [_REFV0]) =\n')
            f.write('\t\t\t'+str(dfParTrue_fit[6])+'\t'+str(dfParEst_fit[6])+'\t'+str(dfParFid_fit[6])+'\t'+\
                             str(dfParMin_fit [6])+'\t'+str(dfParMax_fit[6])+'\t'+str(dfParFitNo  [6])+'\n')
            f.write('#   ln( h_sigma_vel_out [_REFR0]) =\n')
            f.write('\t\t\t'+str(dfParTrue_fit[7])+'\t'+str(dfParEst_fit[7])+'\t'+str(dfParFid_fit[7])+'\t'+\
                             str(dfParMin_fit [7])+'\t'+str(dfParMax_fit[7])+'\t'+str(dfParFitNo  [7])+'\n')

    else:
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                "distribution function type "+str(dftype)+" is not defined.")

    #============================
    #=====SELECTION FUNCTION=====
    #============================

    if update:
        if sfParTrue_phys is None: sfParTrue_phys = out['sfParTrue_phys']
        if sfParEst_phys  is None: sfParEst_phys  = out['sfParEst_phys' ]

    if sfParTrue_phys is None: 
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "sfParTrue_phys must be set.")
    else: sfParTrue_phys = numpy.array(sfParTrue_phys,dtype=float)
    if sfParEst_phys is None: sfParEst_phys = numpy.array(sfParTrue_phys,copy=True)

    if sftype == 1:
        #WEDGE + BOX
        #sfPar = [Rmin_kpc,Rmax_kpc,zmin_kpc,zmax_kpc,phimin_deg,phimax_deg]
        f.write('#\n')  
        f.write('# ***** OBSERVED VOLUME: WEDGE / COMPLETENESS: BOX *****\n')
        f.write('# \t\t true value / estimate / --- / --- / --- / ---\n')
        f.write('# R_min [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[0])+'\t'+str(sfParEst_phys[0])+'\t0\t0\t0\t0\n')
        f.write('# R_max [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[1])+'\t'+str(sfParEst_phys[1])+'\t0\t0\t0\t0\n')
        f.write('# z_min [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[2])+'\t'+str(sfParEst_phys[2])+'\t0\t0\t0\t0\n')
        f.write('# z_max [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[3])+'\t'+str(sfParEst_phys[3])+'\t0\t0\t0\t0\n')
        f.write('# phi_min [deg] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[4])+'\t'+str(sfParEst_phys[4])+'\t0\t0\t0\t0\n')
        f.write('# phi_max [deg] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[5])+'\t'+str(sfParEst_phys[5])+'\t0\t0\t0\t0\n')
    elif sftype == 3:
        #SPHERE + BOX
        #sfPar = [r_sun_kpc,d_sun_kpc]
        #max. radius of sphere / distance of sun (=sphere center) from Galactic center
        f.write('#\n')  
        f.write('# ***** OBSERVED VOLUME: SPHERE / COMPLETENESS: BOX *****\n')
        f.write('# \t\t true value / estimate / --- / --- / --- / ---\n')
        f.write('# r_obs [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[0])+'\t'+str(sfParEst_phys[0])+'\t0\t0\t0\t0\n')
        f.write('# d_obs [kpc] (=R_obs) =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[1])+'\t'+str(sfParEst_phys[1])+'\t0\t0\t0\t0\n')
    elif sftype == 31:
        #SPHERE + INCOMPLETENESS FUNCTION
        #sfPar = [r_sun_kpc,d_sun_kpc,eps_r,eps_z]
        #max. radius of sphere / distance of sun (=sphere center) from Galactic center / ...
        f.write('#\n')
        f.write('# ***** OBSERVED VOLUME: SPHERE / COMPLETENESS: BOX + LINEAR FUNCTION IN r FROM SUN AND z ABOVE PLANE *****\n')
        f.write('# \t\t true value / estimate / --- / --- / --- / ---\n')
        f.write('# r_obs [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[0])+'\t'+str(sfParEst_phys[0])+'\t0\t0\t0\t0\n')
        f.write('# d_obs [kpc] (=R_obs) =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[1])+'\t'+str(sfParEst_phys[1])+'\t0\t0\t0\t0\n')
        f.write('# eps_r in comp(r)=1-eps_r* r /r_obs =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[2])+'\t'+str(sfParEst_phys[2])+'\t0\t0\t0\t0\n')
        f.write('# eps_z in comp(z)=1-eps_z*|z|/r_obs =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[3])+'\t'+str(sfParEst_phys[3])+'\t0\t0\t0\t0\n')
    elif sftype == 32:
        #SPHERE + BOX + FREE CENTER
        #sfPar = [d_max_kpc,R_cen_kpc,phi_cen_deg,z_cen_kpc]
        #max. radius of sphere / (R,phi,z) coordinates of sphere center
        f.write('#\n')  
        f.write('# ***** OBSERVED VOLUME: SPHERE / COMPLETENESS: BOX / CENTER: FREE *****\n')
        f.write('# \t\t true value / estimate / --- / --- / --- / ---\n')
        f.write('# d_max   [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[0])+'\t'+str(sfParEst_phys[0])+'\t0\t0\t0\t0\n')
        f.write('# R_cen   [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[1])+'\t'+str(sfParEst_phys[1])+'\t0\t0\t0\t0\n')
        f.write('# phi_cen [deg] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[2])+'\t'+str(sfParEst_phys[2])+'\t0\t0\t0\t0\n')
        f.write('# z_cen   [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[3])+'\t'+str(sfParEst_phys[3])+'\t0\t0\t0\t0\n')
    elif sftype == 4:
        #INCOMPLETE SHELL
        #sfPar = [dmin_kpc,dmax_kpc,Rgc_sun_kpc,phigc_sun_deg,zgc_sun_kpc,file_no]
        #max. and min. radius of shell, galactocentric position of Sun/center of shell,
        #file number containing incompleteness information
        f.write('#\n')  
        f.write('# ***** OBSERVED VOLUME: SHELL / COMPLETENESS: FROM FILE / CENTER: FREE *****\n')
        f.write('# \t\t true value / estimate / --- / --- / --- / ---\n')
        f.write('# d_min      [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[0])+'\t'+str(sfParEst_phys[0])+'\t0\t0\t0\t0\n')
        f.write('# d_max      [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[1])+'\t'+str(sfParEst_phys[1])+'\t0\t0\t0\t0\n')
        f.write('# R_gc_Sun   [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[2])+'\t'+str(sfParEst_phys[2])+'\t0\t0\t0\t0\n')
        f.write('# phi_gc_Sun [deg] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[3])+'\t'+str(sfParEst_phys[3])+'\t0\t0\t0\t0\n')
        f.write('# z_gc_Sun   [kpc] =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[4])+'\t'+str(sfParEst_phys[4])+'\t0\t0\t0\t0\n')
        f.write('# file_no          =\n')
        f.write('\t\t\t'+str(sfParTrue_phys[5])+'\t'+str(sfParEst_phys[5])+'\t0\t0\t0\t0\n')
    else:
        sys.exit("Error in write_RoadMapping_parameters(): "+\
                 "selection function type "+str(sftype)+" is not defined.")

    if datatype in [2,5] and sftype == 4:
        print "Warning in write_RoadMapping_parameters(): "+\
              "If datatype = [2,5] and sftype = 4: Check that they use the same sun coordinates!"

    #_____close file_____
    f.close()





#f.write('\t\t\t'+str()+'\t'+str()+'\t'+str()+'\t'+str()+'\t'+str()+'\n')

    
