#_____import packages_____
import numpy
import sys

def read_RoadMapping_parameters(datasetname,testname=None,mockdatapath='../data/',fulldatapath=None,print_to_screen=False):

    """
        NAME:
           read_RoadMapping_parameters
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-27 - Started read_RoadMapping_parameters.py on the basis of BovyCode/py/read_flexible_analysis_parameters.py - Trick (MPIA)
                      - Added selection function type 32, SPHERE + BOX + FREE CENTER - Trick (MPIA)
           2016-01-18 - Added pottype 5 and 51, Miyamoto-Nagai disk, Hernquist halo + Hernquist bulge for Elena D'Onghias Simulation - Trick (MPIA)
           2016-02-15 - Added pottype 6,7,61,71, special Bulge + Disk + Halo potentials - Trick (MPIA)
           2016-04-11 - Upgrade to fileversion 2, which will be used for Dynamics Paper 2. This new version allows to control the actionAngleStaeckelGrid via the analysis input file.
           2016-09-22 - Added pottype 4 and 41, MWPotential2014 by Bovy (2015) - Trick (MPIA)
           2016-09-25 - Added pottype 42 and 421, MWPotential from galpy - Trick (MPIA)
           2016-12-13 - Added datatype 5, which uses TGAS/RAVE data and a covariance error matrix. - Trick (MPIA)
           2016-12-27 - Added priortype. - Trick (MPIA)
           2016-12-30 - Added pottype 8, 81 (for fitting to Gaia data). - Trick (MPIA)
    """

    #analysis parameter file:
    if testname is None:
        outname = datasetname+"_analysis_parameters.txt"
    else:
        outname = datasetname+"_"+testname+"_analysis_parameters.txt"
    if fulldatapath is None:
        filename = mockdatapath+datasetname+"/"+outname
    else:
        filename = fulldatapath+outname

    #load parameters:
    out = numpy.loadtxt(filename)

    #Reference scales:
    _REFR0 = 8.                 #[kpc], Reference radius 
    _REFV0 = 220.               #[km/s], Reference velocity

    #*************************
    #***** GENERAL SETUP *****
    #*************************
    #data & model:
    datatype    = int(round(out[0,0]))
    pottype     = int(round(out[0,1]))
    sftype      = int(round(out[0,2]))
    priortype   = int(round(out[0,3]))
    fileversion = int(round(out[0,5]))

    #numerical precision of analysis:
    if fileversion == 2:
        #likelihood normalisation:
        N_spatial      = int(round(out[1,0]))
        N_velocity     = int(round(out[1,1]))
        N_sigma        =           out[1,2]
        vT_galpy_max   =           out[1,3]
        #MCMC:
        MCMC_use_fidDF =      bool(out[2,0])
        noMCMCsteps    = int(round(out[2,1]))
        noMCMCburnin   = int(round(out[2,2]))
        #actionAngleStaeckel Delta:
        use_default_Delta =     bool(out[3,0])
        estimate_Delta    =     bool(out[3,1])
        Delta_fixed       =    float(out[3,2])
        #actionAngleStaeckelGrid:
        use_aASG       =        bool(out[4,0])
        aASG_accuracy  = numpy.array(out[4,1:5],dtype=float)
        nl = 5 #so far four lines: general setup + 4 lines numerical precision
    elif fileversion == 1:
        #likelihood normalisation:
        N_spatial      = int(round(out[1,0]))
        N_velocity     = int(round(out[1,1]))
        N_sigma        =           out[1,2]
        vT_galpy_max   =           out[1,3]
        #MCMC:
        MCMC_use_fidDF =      bool(out[2,0])
        noMCMCsteps    = int(round(out[2,1]))
        noMCMCburnin   = int(round(out[2,2]))
        #actionAngleStaeckel Delta:
        use_default_Delta = True
        estimate_Delta    = False
        Delta_fixed       = 0
        #actionAngleStaeckelGrid:
        use_aASG       = (pottype in numpy.array([21,31,41,421,51,61,71,81],dtype=int))
        aASG_accuracy  = numpy.zeros(4)+numpy.nan
        nl = 3 #so far three lines: general setup + 2 lines numerical precision
    elif fileversion == 0:
        #likelihood normalisation:
        N_spatial      = int(round(out[1,0]))
        N_velocity     = int(round(out[1,1]))
        N_sigma        =           out[1,2]
        vT_galpy_max   = 1.5
        #MCMC:
        MCMC_use_fidDF =      bool(out[1,3])
        noMCMCsteps    = int(round(out[1,4]))
        noMCMCburnin   = int(round(out[1,5]))
        #actionAngleStaeckel Delta:
        use_default_Delta = True
        estimate_Delta    = False
        Delta_fixed       = 0
        #actionAngleStaeckelGrid:
        use_aASG       = (pottype in numpy.array([21,31,41,421,51,61,71,81],dtype=int))
        aASG_accuracy  = numpy.zeros(4)+numpy.nan
        nl = 2 #so far two lines: general setup + numerical precision


    #===================
    #=====DATA TYPE=====
    #===================

    marginal_coord   = 0
    ngl_marginal     = 0
    N_error_samples  = 0
    random_seed_for_errors = 0
    errPar_obs       = numpy.zeros(4) + numpy.nan
    sunCoords_phys   = numpy.zeros(6) + numpy.nan

    if datatype == 1: 

        noStars      = int(round(out[nl,0]))

        #next line:
        nl += 1    #1 more line: number of data points

        if print_to_screen:
            print "DATA TYPE: * mock data"
            print "           * no measurement errors"
            print "           * ",noStars," stars"

    elif datatype == 2:

        noStars          = int(round(out[nl,0]))
        N_error_samples  = int(round(out[nl,1]))
        random_seed_for_errors = int(round(out[nl,2]))
        errPar_obs       = numpy.array(out[nl+1,0:4],dtype=float)
        sunCoords_phys   = numpy.array(out[nl+2, : ],dtype=float)
        sys.exit("Error in read_RoadMapping_parameters(), datatype=2: the code uses ANALYSIS['sunCoords_phys'] given as galactocentric cylindrical coordinates of the Sun for datatype=2. datatype=5 however uses ANALYSIS['sunCoords_phys'] given as galactocentric (x,y,z) coordinates. If I want to use datatye=2 again, the code should be changed analogous to datatype=5. For historical reasons I leave the original version of the code here, for the time being.")

        #next line:
        nl += 3    #3 more lines: number of data points + globar errors + position of sun

        if print_to_screen:
            print "DATA TYPE: * mock data"
            print "           * ",noStars," stars"
            print "           * observables with measurement errors"
            print "           * 'general' measurement errors (just for info):"
            print "                d(distance modulus) [mag], d(RA) & d(DEC) [rad]"
            print "               ",errPar_obs[0:2]
            print "                d(v_los) [kms], d(prop motion) [mas/yr]"
            print "               ",errPar_obs[2:4]
            print "           * APPROXIMATION: ignoring position errors in the analysis"
            print "                  (except effect of distance errors on velocities)"
            print "           * Coordinates of the sun (heliocentric observables --> Galactocentric coordinates):"
            print "                R [kpc], phi [deg], z [kpc], vR [km/s], vT [km/s], vz [km/s]"
            print "               ",sunCoords_phys               

    elif datatype == 3:

        noStars        = int(round(out[nl,0]))
        marginal_coord = int(round(out[nl,1]))
        ngl_marginal   = int(round(out[nl,2]))

        #next line:
        nl += 1    #1 more line: number of data points

        if   marginal_coord == 1: marginalize_over = 'R'
        elif marginal_coord == 2: marginalize_over = 'vR'
        elif marginal_coord == 4: marginalize_over = 'vT'
        elif marginal_coord == 5: marginalize_over = 'z'
        elif marginal_coord == 6: marginalize_over = 'vz'
        else:
            sys.exit("Error in read_RoadMapping_analysis_parameters(): "+\
                 "Coordinate no. "+marginal_coord+" to marginalize over is not known.")
        
        if print_to_screen:
            print "DATA TYPE: * mock data"
            print "           * no measurement errors"
            print "           * ",noStars," stars"
            print "           * don't use data coordinate ",marginalize_over
            print "           * marginalization over ",marginalize_over," with N_gl = ",ngl_marginal

    elif datatype == 4:

        noStars     = [int(round(out[nl,0])),int(round(out[nl,1])),int(round(out[nl,2]))]

        #next line
        nl += 1    #1 more line: number of data points

        if print_to_screen:
            print "DATA TYPE: * mock data"
            print "           * mix of two data sets: "
            print "              "+datasetname+"_MAIN with",noStars[1]," stars"
            print "              "+datasetname+"_POLL with",noStars[2]," stars"
            print "           * in total ",noStars[0]," stars"
            print "           * no measurement errors"

    elif datatype == 5:

        noStars          = int(round(out[nl,0]))
        N_error_samples  = int(round(out[nl,1]))
        random_seed_for_errors = int(round(out[nl,2]))
        sunCoords_phys   = numpy.array(out[nl+1, : ],dtype=float)

        #next line:
        nl += 2    #2 more lines: number of data points + galactocentric (x,y,z) position & velocity of sun

        if print_to_screen:
            print "DATA TYPE: * TGAS/RAVE red clump stars"
            print "           * ",noStars," stars"
            print "           * observables with covariance error matrix"
            print "           * APPROXIMATION: ignoring position errors in the analysis"
            print "                  (except effect of position errors on velocities)"
            print "           * Coordinates of the sun (heliocentric observables --> Galactocentric coordinates):"
            print "                X_gc [kpc], Y_gc [kpc], Z_gc [kpc], vX_gc [km/s], vY_gc [km/s], vZ_gc [km/s]"
            print "               ",sunCoords_phys 

    else:
        sys.exit("Error in read_RoadMapping_analysis_parameters(): "+\
                 "data type "+str(datatype)+" is not defined.")

    #===================
    #=====POTENTIAL=====
    #===================

    if pottype == 1:
        #ISOCHRONE
        #potPar = [R0_kpc,vc_kms,b_kpc]
        potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$b$ [kpc]']
        potNamesScreen = ['R_sun [kpc]','v_c [km/s]','b [kpc]']

        potParTrue_phys = numpy.array(out[nl:nl+3,0],dtype=float)
        potParEst_phys  = numpy.array(out[nl:nl+3,1],dtype=float)
        potParMin_phys  = numpy.array(out[nl:nl+3,3],dtype=float)
        potParMax_phys  = numpy.array(out[nl:nl+3,4],dtype=float)
        potParFitNo     = numpy.array(out[nl:nl+3,5],dtype=int)
        potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)

        #physical boundaries:
        potParLowerBound_phys = numpy.array([0.,0.,0.])
        potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf,numpy.inf])

        #next line:
        nl += 3

        if print_to_screen:
            print "POTENTIAL: * isochrone potential"
            print "           * true parameters:"
            print "                R_sun [kpc], v_circ(R_sun) [km/s], b [kpc]"
            print "               ",potParTrue_phys

    elif pottype == 2 or pottype == 21:
        #2-COMPONENT KK STAECKEL
        #potPar = [R0_kpc,vc_kms,Delta,ac_D,ac_H,k]
        potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$\Delta$','$(a/c)_{Disk}$','$(a/c)_{Halo}$','$k$']
        potNamesScreen = ['R_sun [kpc]','v_c [km/s]','Delta','ac_D','ac_H','k']

        potParTrue_phys = numpy.array(out[nl:nl+6,0],dtype=float)
        potParEst_phys  = numpy.array(out[nl:nl+6,1],dtype=float)
        potParMin_phys  = numpy.array(out[nl:nl+6,3],dtype=float)
        potParMax_phys  = numpy.array(out[nl:nl+6,4],dtype=float)
        potParFitNo     = numpy.array(out[nl:nl+6,5],dtype=int)
        potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)

        #physical boundaries:
        potParLowerBound_phys = numpy.array([0.,0.,0.,2.,1.,0.])
        potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf,numpy.inf,numpy.inf,2.,1.])

        #next line:
        nl += 6

        if print_to_screen:
            print "POTENTIAL: * 2-component KK Staeckel potential"
            print "             (Batsleer & Dejonghe 1994)"
            if   pottype == 2:  print "           * action calculation: actionAngleStaeckel"
            elif pottype == 21: print "           * action calculation: actionAngleStaeckelGrid"
            print "           * true parameters:"
            print "                R_sun [kpc], v_circ(R_sun) [km/s], Delta, ac_D, ac_H, k"
            print "               ",potParTrue_phys

    elif pottype == 3 or pottype == 31:
        #MW-LIKE (Bovy & Rix 2013)
        #potPar = [R0_kpc,vc_kms,Rd_kpc,zh_kpc,fh,dlnvcdlnr]
        potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$R_d$ [kpc]','$z_h$ [kpc]','$f_h$','$d(\ln v_c) / d(\ln r)$']
        potNamesScreen = ['R_sun [kpc]','v_c [km/s]','R_d [kpc]','z_h [kpc]','f_h','d(ln v_c)/d(ln r)']

        potParTrue_phys = numpy.array(out[nl:nl+6,0],dtype=float)
        potParEst_phys  = numpy.array(out[nl:nl+6,1],dtype=float)
        potParMin_phys  = numpy.array(out[nl:nl+6,3],dtype=float)
        potParMax_phys  = numpy.array(out[nl:nl+6,4],dtype=float)
        potParFitNo     = numpy.array(out[nl:nl+6,5],dtype=int)
        potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)

        #physical boundaries:
        potParLowerBound_phys = numpy.array([0.,0.,0.,0.,0.,-numpy.inf])
        potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf,numpy.inf,numpy.inf,1.,numpy.inf])

        #next line:
        nl += 6

        if print_to_screen:
            print "POTENTIAL: * MW-like potential"
            print "             (Bovy & Rix 2013)"
            if   pottype == 3:  print "           * action calculation: actionAngleStaeckel"
            elif pottype == 31: print "           * action calculation: actionAngleStaeckelGrid"
            print "           * true parameters:"
            print "                R_sun [kpc], v_circ(R_sun) [km/s], R_d [kpc], z_h [kpc], f_h, d(ln v_c)/d(ln r)"
            print "               ",potParTrue_phys
    
    elif pottype in numpy.array([4,41,42,421],dtype=int):
        #MWPotential2014 (Bovy2015) (4) or MWPotential from galpy (42)
        #potPar = [R0_kpc,vc_kms]
        potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]']
        potNamesScreen = ['R_sun [kpc]','v_c [km/s]']

        potParTrue_phys = numpy.array(out[nl:nl+2,0],dtype=float)
        potParEst_phys  = numpy.array(out[nl:nl+2,1],dtype=float)
        potParMin_phys  = numpy.array(out[nl:nl+2,3],dtype=float)
        potParMax_phys  = numpy.array(out[nl:nl+2,4],dtype=float)
        potParFitNo     = numpy.array(out[nl:nl+2,5],dtype=int)
        potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)

        #physical boundaries:
        potParLowerBound_phys = numpy.array([0.,0.])
        potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf])

        #next line:
        nl += 2

        if print_to_screen:
            
            if   pottype in numpy.array([4 ,41 ],dtype=int): print "POTENTIAL: * MWPotential2014 (Bovy 2015)"
            elif pottype in numpy.array([42,421],dtype=int): print "POTENTIAL: * MWPotential from galpy"
            if   pottype in numpy.array([4 ,42 ],dtype=int): print "           * action calculation: actionAngleStaeckel"
            elif pottype in numpy.array([41,421],dtype=int): print "           * action calculation: actionAngleStaeckelGrid"
            print "           * true parameters:"
            print "                R_sun [kpc], v_circ(R_sun) [km/s]"
            print "               ",potParTrue_phys
    
    elif pottype in numpy.array([5,6,7,8,51,61,71,81],dtype=int):
        if pottype == 5 or pottype == 51:
            #POTENTIAL 1 FOR ELENA D'ONGHIA SIMULATION
            #potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
            potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$a_{disk}$ [kpc]','$b_{disk}$ [kpc]','$f_{halo}$','$a_{halo}$ [kpc]']
            potNamesScreen = ['R_sun [kpc]','v_c [km s$^-1$]','a_d [kpc]','b_d [kpc]','f_h','a_h [kpc]']
        elif pottype == 6 or pottype == 61:
            #POTENTIAL 2 FOR ELENA D'ONGHIA SIMULATION
            #potPar = [R0_kpc,vc_kms,hr_disk_kpc,hz_disk_kpc,f_halo,a_halo_kpc]
            potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$h_{r,disk}$ [kpc]','$h_{z,disk}$ [kpc]','$f_{halo}$','$a_{halo}$ [kpc]']
            potNamesScreen = ['R_sun [kpc]','v_c [km s$^-1$]','hr_d [kpc]','hz_d [kpc]','f_h','a_h [kpc]']
        elif pottype == 7 or pottype == 71:
            #GALPY MWPotential LIKE MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE
            #potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
            potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$a_{disk}$ [kpc]','$b_{disk}$ [kpc]','$f_{halo}$','$a_{halo}$ [kpc]']
            potNamesScreen = ['R_sun [kpc]','v_c [km s$^-1$]','a_d [kpc]','b_d [kpc]','f_h','a_h [kpc]']
        elif pottype == 8 or pottype == 81:
            #POTENTIAL FOR FIT TO GAIA DATA WITH MIYAMOTO-NAGAI DISK + NFW HALO + HERNQUIST BULGE
            #potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc,M_bulge_1010Msun,a_bulge_kpc]
            potNamesLatex  = ['$R_\odot$ [kpc]','$v_{circ}(R_\odot)$ [km s$^-1$]','$a_{disk}$ [kpc]','$b_{disk}$ [kpc]','$f_{halo}$','$a_{halo}$ [kpc]','$M_{bulge}$ [$10^{10}$ M$_\odot$]','$a_{bulge}$ [kpc]']
            potNamesScreen = ['R_sun [kpc]','v_c [km s$^-1$]','a_d [kpc]','b_d [kpc]','f_h','a_h [kpc]','M_b [10^10 M_sun]','a_b [kpc]']

        if pottype in [5,6,7,51,61,71]:
            npotpar = 6
        elif pottype in [8,81]:
            npotpar = 8

        potParTrue_phys = numpy.array(out[nl:nl+npotpar,0],dtype=float)
        potParEst_phys  = numpy.array(out[nl:nl+npotpar,1],dtype=float)
        potParMin_phys  = numpy.array(out[nl:nl+npotpar,3],dtype=float)
        potParMax_phys  = numpy.array(out[nl:nl+npotpar,4],dtype=float)
        potParFitNo     = numpy.array(out[nl:nl+npotpar,5],dtype=int)
        potParFitBool   = numpy.array((potParFitNo > 1),dtype=bool)

        #physical boundaries:
        potParLowerBound_phys = numpy.zeros(npotpar)
        if   pottype in [5,6,7,51,61,71]: potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf,numpy.inf,numpy.inf,1.,numpy.inf])
        elif pottype in [8,81]:           potParUpperBound_phys = numpy.array([numpy.inf,numpy.inf,numpy.inf,numpy.inf,1.,numpy.inf,numpy.inf,numpy.inf])

        #next line:
        nl += npotpar

        if print_to_screen:
            if   pottype in numpy.array([5,51]     ,dtype=int): print "POTENTIAL: * potential with Miyamoto-Nagai disk, Hernquist halo + bulge "
            elif pottype in numpy.array([6,61]     ,dtype=int): print "POTENTIAL: * potential with Double Exponential disk, Hernquist halo + bulge "
            elif pottype in numpy.array([7,71]     ,dtype=int): print "POTENTIAL: * potential with Miyamoto-Nagai disk, NFW halo + Hernquist bulge "
            elif pottype in numpy.array([8,81]     ,dtype=int): print "POTENTIAL: * potential with Miyamoto-Nagai disk, NFW halo + Hernquist bulge "
            if   pottype in numpy.array([5,51,6,61],dtype=int): print "             (for simulation by Elena D'Onghia)"
            elif pottype in numpy.array([7,71]     ,dtype=int): print "             (galpy MWPotential like)"
            elif pottype in numpy.array([8,81]     ,dtype=int): print "             (for fitting to Gaia data)"
            if   pottype in numpy.array([5,6,7,8]    ,dtype=int): print "           * action calculation: actionAngleStaeckel"
            elif pottype in numpy.array([51,61,71,81] ,dtype=int): print "           * action calculation: actionAngleStaeckelGrid"
            print "           * true parameters:"
            if   pottype in numpy.array([5,51,7,71],dtype=int):print "                R_sun [kpc], v_circ(R_sun) [km/s], a_d [kpc], b_d [kpc], f_h, a_h [kpc]"
            elif pottype in numpy.array([6,61]     ,dtype=int):print "                R_sun [kpc], v_circ(R_sun) [km/s], hr_d [kpc], hz_d [kpc], f_h, a_h [kpc]"
            elif pottype in numpy.array([8,81]     ,dtype=int):print "                R_sun [kpc], v_circ(R_sun) [km/s], a_d [kpc], b_d [kpc], f_h, a_h [kpc], M_b [10^10 Msun], a_b [kpc]"
            print "               ",potParTrue_phys
    else:
        sys.exit("Error in read_RoadMapping_analysis_parameters(): "+\
                 "potential type "+str(pottype)+" is not defined.")

    #scaling:

    if potParFitNo[0] == 1:
        ro = potParEst_phys[0] / _REFR0
        ro_known = True
    else:
        ro = None
        ro_known = False
    if potParFitNo[1] == 1:
        vo = potParEst_phys[1] / _REFV0
        vo_known = True    
    else:
        vo = None
        vo_known = False

    if print_to_screen:
        print "           * parameters kept fixed?"
        print "               ",numpy.invert(potParFitBool)
        print "             at value"
        print "               ",potParEst_phys



    #=============
    #=====QDF=====
    #=============
    #dfPar_phys = [hr_kpc,sr_kms,sz_kms,hsr_kpc,hsz_kpc]

    dfNamesLatex = ['ln($h_R$/8kpc)','ln($\sigma_{R,0}$/220km s$^-1$)','ln($\sigma_{z,0}$/220km s$^-1$)',\
                    'ln($h_{\sigma,R}$/8kpc)','ln($h_{\sigma,z}$/8kpc)']
    dfNamesScreen =  ['ln(h_R)','ln(s_R0)','ln(s_z0)',\
                      'ln(h_s_R)','ln(h_s_z)']

    dfParTrue_phys = numpy.array(out[nl:nl+5,0],dtype=float)
    dfParEst_phys  = numpy.array(out[nl:nl+5,1],dtype=float)
    dfParFid_phys  = numpy.array(out[nl:nl+5,2],dtype=float)
    dfParMin_phys  = numpy.array(out[nl:nl+5,3],dtype=float)
    dfParMax_phys  = numpy.array(out[nl:nl+5,4],dtype=float)
    
    #next line:
    nl += 5

    dfParTrue_fit = numpy.array(out[nl:nl+5,0],dtype=float)
    dfParEst_fit  = numpy.array(out[nl:nl+5,1],dtype=float)
    dfParFid_fit  = numpy.array(out[nl:nl+5,2],dtype=float)
    dfParMin_fit  = numpy.array(out[nl:nl+5,3],dtype=float)
    dfParMax_fit  = numpy.array(out[nl:nl+5,4],dtype=float)
    dfParFitNo    = numpy.array(out[nl:nl+5,5],dtype=int)
    dfParFitBool  = numpy.array((dfParFitNo > 1),dtype=bool)

    #next line:
    nl += 5

    if print_to_screen:
        print "QUASI-ISOTHERMAL DISTRIBUTION FUNCTION:"
        print "           * true parameters:"
        print "                hr [kpc], sr [km/s], sz [km/s], hsr [kpc], hsz [kpc]"
        print "               ",dfParTrue_phys
        print "           * parameters kept fixed?"
        print "               ",numpy.invert(dfParFitBool)
        print "             at value"
        print "               ",dfParEst_phys
        print "           * fiducial qdf parameters:"
        print "               ",dfParFid_phys

    #physical boundaries:
    dfParLowerBound_fit = numpy.array([-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf])
    dfParUpperBound_fit = numpy.array([numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf])

    #============================
    #=====SELECTION FUNCTION=====
    #============================

    if sftype == 1:
        #WEDGE + BOX
        #sfPar = [Rmin_kpc,Rmax_kpc,zmin_kpc,zmax_kpc,phimin_deg,phimax_deg]
        sfParTrue = numpy.array(out[nl:nl+6,0],dtype=float)
        sfParEst  = numpy.array(out[nl:nl+6,1],dtype=float)

        if print_to_screen:
            print "SELECTION FUNCTION:"
            print "           * wedge"
            print "           * true parameters:"
            print "                R_min [kpc], R_max [kpc], z_min [kpc], z_max [kpc], phi_min [deg], phi_max [deg]"
            print "               ",sfParTrue
            if numpy.sum(sfParTrue != sfParEst) > 0:
                print "           * parameters used in analysis:"
                print "               ",sfParEst
    elif sftype == 3:
        #SPHERE + BOX
        #sfPar = [r_obs_kpc,d_obs_kpc]
        #max. radius of sphere / distance of sun (=sphere center) from Galactic center
        sfParTrue = numpy.array(out[nl:nl+2,0],dtype=float)
        sfParEst  = numpy.array(out[nl:nl+2,1],dtype=float)
        
        if print_to_screen:
            print "SELECTION FUNCTION:"
            print "           * sphere around sun"
            print "           * true parameters:"
            print "                r_obs [kpc], d_obs [kpc]"
            print "               ",sfParTrue
            if numpy.sum(sfParTrue != sfParEst) > 0:
                print "           * parameters used in analysis:"
                print "               ",sfParEst
    elif sftype == 31:
        #SPHERE + INCOMPLETENESS FUNCTION
        #sfPar = [r_sun_kpc,d_sun_kpc,eps_r,eps_z]
        #max. radius of sphere / distance of sun (=sphere center) from Galactic center / ...
        sfParTrue = numpy.array(out[nl:nl+4,0],dtype=float)
        sfParEst  = numpy.array(out[nl:nl+4,1],dtype=float)
        
        if print_to_screen:
            print "SELECTION FUNCTION:"
            print "           * sphere around sun"
            print "           * incompleteness function:"
            print "             comp(r,z) = (1 - eps_r *  r /r_obs)"
            print "                       * (1 - eps_z * |z|/r_obs)"
            print "           * true parameters:"
            print "                r_obs [kpc], d_obs [kpc], eps_r, eps_z"
            print "               ",sfParTrue
            if numpy.sum(sfParTrue != sfParEst) > 0:
                print "           * parameters used in analysis:"
                print "               ",sfParEst
    elif sftype == 32:
        #SPHERE + BOX + FREE TO CHOOSE CENTER
        #sfPar = [d_max_kpc,R_cen_kpc,phi_cen_deg,z_cen_kpc]
        #max. radius of sphere / (R,phi,z) coordinates of sphere center
        sfParTrue = numpy.array(out[nl:nl+4,0],dtype=float)
        sfParEst  = numpy.array(out[nl:nl+4,1],dtype=float)
        
        if print_to_screen:
            print "SELECTION FUNCTION:"
            print "           * sphere around sun"
            print "           * true parameters:"
            print "                d_max [kpc], R_cen [kpc], phi_cen [deg], z_cen [kpc]"
            print "               ",sfParTrue
            if numpy.sum(sfParTrue != sfParEst) > 0:
                print "           * parameters used in analysis:"
                print "               ",sfParEst
    elif sftype == 4:
        #INCOMPLETE SHELL
        #sfPar = [dmin_kpc,dmax_kpc,Rgc_sun_kpc,phigc_obs_deg,zgc_sun_kpc,file_no]
        #max. and min. radius of shell, galactocentric position of Sun/center of shell,
        #file number containing incompleteness information
        sfParTrue = numpy.array(out[nl:nl+6,0],dtype=float)
        sfParEst  = numpy.array(out[nl:nl+6,1],dtype=float)
        
        if print_to_screen:
            print "SELECTION FUNCTION:"
            print "           * shell around sun"
            print "           * incompleteness from file no. ",int(sfParTrue[5])
            print "           * true parameters:"
            print "                d_min [kpc], d_max [kpc], R_Sun [kpc], phi_Sun [deg], z_Sun [kpc]"
            print "               ",sfParTrue[0:5]
            if numpy.sum(sfParTrue != sfParEst) > 0:
                print "           * parameters used in analysis:"
                print "               ",sfParEst  
    else:
        sys.exit("Error in read_RoadMapping_analysis_parameters(): "+\
                 "selection function type "+str(sftype)+" is not defined.")

    if print_to_screen:

        print "PRIORS:"
        if priortype in [0,1]:
            print "           * Flat priors on potential parameters (inside physical regime)."
            print "           * Logarithmically flat priors on DF parameters."
        if priortype == 1:
            print "           * Bovy & Rix (2013), eq. (41), prior on slope of rotation curve."

        print "NUMERICAL PRECISION:"
        print "           * N_spatial = ",N_spatial,", N_velocity = ",N_velocity
        print "           * N_sigma   = ",N_sigma,", vT_max [galpy] = ",vT_galpy_max
        print "           * # of MCMC steps: ",noMCMCsteps,", # of burn-in steps: ",noMCMCburnin
        if MCMC_use_fidDF: 
            print "           * Analysis uses FIDUCIAL qDF for fitting range also in the MCMC."
        else:
            print "           * Analysis uses CURRENT qDF for fitting range in the MCMC."
        if datatype in [2,5]:
            print "           * N_error_samples = ",N_error_samples,", random seed = ",random_seed_for_errors
        if estimate_Delta:
            print "           * Analysis uses the Staeckel fudge and ESTIMATES the focal length DELTA for each potential."
        if not estimate_Delta and not use_default_Delta:
            print "           * Analysis uses the Staeckel fudge with FIXED focal length DELTA=",Delta_fixed,"*ro for each potential."
        if not estimate_Delta and use_default_Delta:
            print "           * Analysis uses the Staeckel fudge with DEFAULT focal length DELTA=0.45*ro for each potential."
        if use_aASG:
            print "           * Analysis uses the actionAngleStaeckelGrid with:"
            print "             Rmax = ",aASG_accuracy[0],", nE = ",int(aASG_accuracy[1]),","
            print "             npsi = ",int(aASG_accuracy[2]),", nLz = ",int(aASG_accuracy[3])

    return {
            'datatype'  :datatype,
            'pottype'   :pottype,
            'sftype'    :sftype,
            'priortype' :priortype,
            'noStars'   :noStars,
            'potParTrue_phys':potParTrue_phys,'dfParTrue_phys':dfParTrue_phys,'dfParTrue_fit':dfParTrue_fit,
            'potParEst_phys' :potParEst_phys , 'dfParEst_phys': dfParEst_phys, 'dfParEst_fit': dfParEst_fit,
                                               'dfParFid_phys': dfParFid_phys, 'dfParFid_fit': dfParFid_fit,
            'potParMin_phys' :potParMin_phys , 'dfParMin_phys': dfParMin_phys, 'dfParMin_fit': dfParMin_fit,
            'potParMax_phys' :potParMax_phys , 'dfParMax_phys': dfParMax_phys, 'dfParMax_fit': dfParMax_fit,
            'potParFitNo'  :potParFitNo  ,'dfParFitNo'  :dfParFitNo,
            'potParFitBool':potParFitBool,'dfParFitBool':dfParFitBool,
            'potParLowerBound_phys':potParLowerBound_phys,
            'potParUpperBound_phys':potParUpperBound_phys,
            'dfParLowerBound_fit':dfParLowerBound_fit,
            'dfParUpperBound_fit':dfParUpperBound_fit,
            'sfParTrue_phys' :sfParTrue ,'sfParEst_phys':sfParEst,
            'ro':ro,'vo':vo,'ro_known':ro_known,'vo_known':vo_known,
            'potNamesScreen':potNamesScreen,'dfNamesScreen':dfNamesScreen,
            'potNamesLatex':potNamesLatex,'dfNamesLatex':dfNamesLatex,
            'N_spatial':N_spatial,'N_velocity':N_velocity,'N_sigma':N_sigma,'vT_galpy_max':vT_galpy_max,
            'noMCMCsteps':noMCMCsteps,'noMCMCburnin':noMCMCburnin,'MCMC_use_fidDF':MCMC_use_fidDF,
            'marginal_coord':marginal_coord,'ngl_marginal':ngl_marginal,
            'N_error_samples':N_error_samples,'random_seed_for_errors':random_seed_for_errors,
            'errPar_obs':errPar_obs,'sunCoords_phys':sunCoords_phys,
            'use_default_Delta':use_default_Delta,'estimate_Delta':estimate_Delta,'Delta_fixed':Delta_fixed,
            'aASG_accuracy':aASG_accuracy
            }

    
