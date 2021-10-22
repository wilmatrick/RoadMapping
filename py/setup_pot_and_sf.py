#_____import packages_____
#from __past__ import division
#from __future__ import print_function
from galpy.potential import IsochronePotential
from galpy.potential import KuzminKutuzovStaeckelPotential
from galpy import potential
from galpy.actionAngle import actionAngleIsochrone,actionAngleStaeckel,actionAngleStaeckelGrid
from galpy.util import bovy_conversion
from SF_Sphere import SF_Sphere
from SF_Wedge import SF_Wedge
from SF_Cylinder import SF_Cylinder
from SF_IncompleteShell import SF_IncompleteShell
import numpy
import sys
import os
import pickle

def setup_Potential_and_ActionAngle_object(pottype,potPar_phys,**kwargs):

    """
        NAME:
           setup_Potential_and_ActionAngle_object
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started setup_Potential_and_ActionAngle_object.py on the basis of BovyCode/py/setup_Potential_and_ActionAngle_object.py - Trick (MPIA)
           2016-01-08 - Corrected bug: For the Staeckel Fudge one should use Delta = 0.45*ro (according to BR13) and NOT Delta = 0.55
           2016-01-18 - Added pottype 5 and 51, Miyamoto-Nagai disk, Hernquist halo + Hernquist bulge for Elena D'Onghias Simulation
           2016-02-15 - Added pottype 6,7,61,71 to the list and rewrote 5-7 to use the new function setup_DiskHaloBulge_potential().
           2016-04-03 - Changed default accuracy of actionAngleStaeckelGrid. From Rmax = 10, nE = 50, npsi = 50, nLz = 60 to Rmax =  5, nE = 70, npsi = 40, nLz = 50
           2016-04-15 - Added an option to return only the potential, and one option to set Delta by hand.
           2016-09-24 - Added pottype 41, 42, 421 . - Trick (MPIA)
           2016-12-30 - Added pottype 8, 81 (for fitting to Gaia data; with MN disk). - Trick (MPIA)
           2017-01-17 - Added pottype 82, 821 (for fitting to Gaia data; with 3xMN disk). - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    #_____scale position of sun and circular velocity to galpy units_____
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0

    if pottype == 1:
        #==========ISOCHRONE==========
        #potParArr = [ro,vo,b_kpc]
        #transformation from physical to galpy units:
        b  = potPar_phys[2] / _REFR0 / ro
        #check, if parameters are physical:
        if b <= 0. or vo <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential
        pot = IsochronePotential(b=b,normalize=True)
        #initialize ActionAngleIsochrone object
        aA = actionAngleIsochrone(ip=pot)

    elif pottype == 2 or pottype == 21:
        #==========2-COMPONENT KUZMIN-KUTUZOV-STAECKEL-POTENTIAL==========
        #==========(Batsleer & Dejonghe 1994) with StaeckelActions=======
        #potParArr = [ro,vo,Delta,ac_D,ac_H,k]
        #transformation from physical to galpy units: not needed
        Delta = potPar_phys[2]  #focal length of (lambda,nu) coordinate system
        ac_D  = potPar_phys[3]  #axis ratio of coordinate surfaces of disk component
        ac_H  = potPar_phys[4]  #axis ratio of coordinate surfaces of halo component
        k     = potPar_phys[5]  #relative contribution of disk to total mass
        #check, if parameters are physical:
        if Delta <= 0. or ac_D <  2. or \
            ac_H <= 1. or ac_H >= 2. or \
            k    <= 0. or k    >= 1. or \
            vo   <= 0.:
            raise RuntimeError("unphysical potential parameters")    
        #coordinate transformation (R_sun=1,z=0) <--> (lambda,nu):
        g_D = Delta**2 / (1.-ac_D**2)
        g_H = Delta**2 / (1.-ac_H**2)
        a_D = g_D - Delta**2
        a_H = g_H - Delta**2
        l   = 1. - a_D          #lambda = R**2 - a, nu = -g at z=0
        q   = a_H - a_D         #eq. (6) in Batsleer & Dejonghe (1994)
        #v_circ contributions due to halo and disk (in eq. (10)):
        termD = numpy.sqrt(l  )*(numpy.sqrt(l  )+numpy.sqrt(-g_D  ))**2
        termH = numpy.sqrt(l-q)*(numpy.sqrt(l-q)+numpy.sqrt(-g_D-q))**2
        #amplitude of normalized potential, to get galpy units v_circ(R_sun=1)=1:
        #GM = (v_circ/R_sun)**2 / (k / termD + (1.-k) / termH) --> (eq. (10))
        GM  = 1.                / (k / termD + (1.-k) / termH)
        #setup two components, disk and halo, potentials (already normalized):
        V_D = KuzminKutuzovStaeckelPotential(amp=GM*k     ,ac=ac_D,Delta=Delta,normalize=False)
        V_H = KuzminKutuzovStaeckelPotential(amp=GM*(1.-k),ac=ac_H,Delta=Delta,normalize=False)
        pot = [V_D,V_H]
    elif pottype == 3 or pottype == 31:
        #==========MW-LIKE=========================================
        #==========(Bovy & Rix 2013) with StaeckelActions==========
        #potParArr = [ro,vo,Rd_kpc,zh_kpc,fh,dlnvcdlnr]
        #transformation from physical to galpy units:
        Rd        = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        zh        = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        fh        = potPar_phys[4]               # halo contribution to the disk's v_c^2
        dlnvcdlnr = potPar_phys[5]               # slope of overall circular velocity rotation curve, d(ln(v_c)) / d(ln(r))
        #check, if parameters are physical:
        if Rd <= 0. or zh <= 0. or vo <= 0. or fh <= 0. or fh >= 1.:
            raise RuntimeError("unphysical potential parameters")
        #setup interpolated potential
        pot = setup_MWlike_potential(
                    numpy.array([Rd,fh,vo,zh,dlnvcdlnr]),
                    ro,
                    returnrawpot=False,
                    ngrid=101
                    )
    elif pottype == 4 or pottype == 41:
        #========== GALPY MW-POTENTIAL 2014 ===========
        #========== with StaeckelActions ==============
        #potParArr = [ro,vo]
        #check, if parameters are physical:
        if vo <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        from galpy.potential import MWPotential2014
        pot = MWPotential2014
    elif pottype == 42 or pottype == 421:
        #========== GALPY MW-POTENTIAL ===========
        #========== with StaeckelActions ==============
        #potParArr = [ro,vo]
        #check, if parameters are physical:
        if vo <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        from galpy.potential import MWPotential
        pot = MWPotential
    elif pottype == 5 or pottype == 51:
        #========== POTENTIAL 1 FOR ELENA D'ONGHIA SIMULATION ==========
        #========== Miyamoto-Nagai disk, Hernquist halo + bulge ========
        #========== with Staeckel actions ==============================
        #potParArr = [ro,vo,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
        #transformation from physical to galpy units:
        a_d = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        b_d = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        f_h = potPar_phys[4]               # halo contribution to the disk's v_c^2
        a_h = potPar_phys[5] / _REFR0 / ro # dark matter halo scale length
        #check, if parameters are physical:
        if a_d <= 0. or b_d <= 0. or vo <= 0. or f_h <= 0. or f_h >= 1. or a_h <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        pot = setup_DiskHaloBulge_potential(
                    numpy.array([ro,vo,a_d,b_d,f_h,a_h]),
                    Disk             = 'MiyamotoNagai',
                    Halo             = 'Hernquist',
                    a_bulge_kpc      = 0.25099812,
                    M_bulge_1010Msun = 0.952400755715,
                    s_bulge          = None
                    )
    elif pottype == 6 or pottype == 61:
        #========== POTENTIAL 2 FOR ELENA D'ONGHIA SIMULATION ==========
        #========== Double exponential disk, Hernquist halo + bulge ========
        #========== with Staeckel actions ==============================
        #potParArr = [ro,vo,hr_disk_kpc,hz_disk_kpc,f_halo,a_halo_kpc]
        #transformation from physical to galpy units:
        h_r = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        h_z = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        f_h = potPar_phys[4]               # halo contribution to the disk's v_c^2
        a_h = potPar_phys[5] / _REFR0 / ro # dark matter halo scale length
        #check, if parameters are physical:
        if h_r <= 0. or h_z <= 0. or vo <= 0. or f_h <= 0. or f_h >= 1. or a_h <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        pot = setup_DiskHaloBulge_potential(
                    numpy.array([ro,vo,h_r,h_z,f_h,a_h]),
                    Disk             = 'DoubleExponential',
                    Halo             = 'Hernquist',
                    a_bulge_kpc      = 0.25099812,
                    M_bulge_1010Msun = 0.952400755715,
                    s_bulge          = None
                    )
    elif pottype == 7 or pottype == 71:
        #========== MWPotential(2014)-LIKE POTENTIAL FOR FITTING ==========
        #========== Miyamoto-Nagai disk, NFW halo, Hernquist bulge ========
        #========== with Staeckel actions ==============================
        #potParArr = [ro,vo,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
        #transformation from physical to galpy units:
        a_d = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        b_d = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        f_h = potPar_phys[4]               # halo contribution to the disk's v_c^2
        a_h = potPar_phys[5] / _REFR0 / ro # dark matter halo scale length
        #check, if parameters are physical:
        if a_d <= 0. or b_d <= 0. or vo <= 0. or f_h <= 0. or f_h >= 1. or a_h <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        pot = setup_DiskHaloBulge_potential(
                    numpy.array([ro,vo,a_d,b_d,f_h,a_h]),
                    Disk             = 'MiyamotoNagai',
                    Halo             = 'NFW',
                    a_bulge_kpc      = 0.6, #analogous to galpy MWPotential
                    M_bulge_1010Msun = None,
                    s_bulge          = 0.05 #analogous to galpy MWPotential
                    )
    elif pottype == 8 or pottype == 81:
        #========== MNdHbNFWh-POTENTIAL FOR FITTING TO GAIA DATA ==========
        #========== Miyamoto-Nagai disk, NFW halo, Hernquist bulge ========
        #========== with Staeckel actions ==============================
        #potParArr = [ro,vo,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc,M_bulge_1010Msun,a_bulge_kpc]
        #transformation from physical to galpy units:
        a_d = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        b_d = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        f_h = potPar_phys[4]               # halo contribution to the disk's v_c^2
        a_h = potPar_phys[5] / _REFR0 / ro # dark matter halo scale length
        #bulge parameters in physical units:
        M_b_1010Msun = potPar_phys[6] # mass of hernquist bulge
        a_b_kpc      = potPar_phys[7] # bulge scale length
        #check, if parameters are physical:
        if a_d <= 0. or b_d <= 0. or vo <= 0. or f_h <= 0. or \
           f_h >= 1. or a_h <= 0. or M_b_1010Msun <= 0. or a_b_kpc <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        pot = setup_DiskHaloBulge_potential(
                    numpy.array([ro,vo,a_d,b_d,f_h,a_h]),
                    Disk             = 'MiyamotoNagai',
                    Halo             = 'NFW',
                    a_bulge_kpc      = a_b_kpc,
                    M_bulge_1010Msun = M_b_1010Msun,
                    s_bulge          = None,
                    )
    elif pottype == 82 or pottype == 821:
        #========== 3MNdHbNFWh-POTENTIAL FOR FITTING TO GAIA DATA ==========
        #========== 3xMiyamoto-Nagai disk approximation (Smith et al. 2015),
        #========== NFW halo, Hernquist bulge ========
        #========== with Staeckel actions ==============================
        #potParArr = [ro,vo,hr_disk_kpc,hz_disk_kpc,f_halo,a_halo_kpc,M_bulge_1010Msun,a_bulge_kpc]
        #transformation from physical to galpy units:
        hr_d = potPar_phys[2] / _REFR0 / ro # stellar disk scale length
        hz_d = potPar_phys[3] / _REFR0 / ro # stellar disk scale height
        f_h  = potPar_phys[4]               # halo contribution to the disk's v_c^2
        a_h  = potPar_phys[5] / _REFR0 / ro # dark matter halo scale length
        #bulge parameters in physical units:
        M_b_1010Msun = potPar_phys[6] # mass of hernquist bulge
        a_b_kpc      = potPar_phys[7] # bulge scale length
        #check, if parameters are physical:
        if hr_d <= 0. or hz_d <= 0. or vo <= 0. or f_h <= 0. or \
           f_h >= 1. or a_h <= 0. or M_b_1010Msun <= 0. or a_b_kpc <= 0.:
            raise RuntimeError("unphysical potential parameters")
        #setup potential:
        pot = setup_DiskHaloBulge_potential(
                    numpy.array([ro,vo,hr_d,hz_d,f_h,a_h]),
                    Disk             = 'MN3_ExponentialSechDisk',
                    Halo             = 'NFW',
                    a_bulge_kpc      = a_b_kpc,
                    M_bulge_1010Msun = M_b_1010Msun,
                    s_bulge          = None,
                    )
    else:
        sys.exit("Error in setup_potential_and_action_object(): "+\
                 "potential type "+str(pottype)+" is not defined.")

    #_____return only potential_____
    if ('return_only_potential' in kwargs) and (kwargs['return_only_potential']):
        return pot

    #_____prepare ActionAngle object initialization and setup Delta_____
    if (pottype in numpy.array([3,31,4,41,42,421,5,51,6,61,7,71,8,81,82,821],dtype=int)):
        if 'aAS_Delta' in kwargs: 
            Delta = kwargs['aAS_Delta']
        else: 
            Delta = 0.45*ro 
            #delta=0.45 * R0 is a good estimate for the Milky Way's 
            #Staeckel approximation (cf. Bovy&Rix 2013)
        #print("Delta = ",Delta)
    elif (pottype in numpy.array([1,2],dtype=int)):
        pass #no Delta needed
    else:
        sys.exit("Error in setup_potential_and_action_object(): "+\
                 "potential type "+str(pottype)+" is not defined yet for "+\
                 "the Delta (yes/no) setup.")



    if pottype in numpy.array([2,3,4,42,5,6,7,8,82],dtype=int):
        #==========StaeckelActions=======
        #initialize ActionAngle object:
        aA = actionAngleStaeckel(pot=pot,delta=Delta,c=True)
        #       c=True (default): use C implementation to speed up actionAngleStaeckel calculations, 
        #                         plus this is needed to calculate frequencies
    elif pottype in numpy.array([21,31,41,421,51,61,71,81,821],dtype=int):
        #==========StaeckelActions on a Grid=======
        if '_MULTI' in kwargs: numcores = kwargs['_MULTI']
        else:                  numcores = 1

        #initialize ActionAngle object:
        if 'aASG_accuracy' in kwargs: 
            aASG_acc = kwargs['aASG_accuracy']
            Rmax = float(    aASG_acc[0])
            nE   = int(round(aASG_acc[1]))
            npsi = int(round(aASG_acc[2]))
            nLz  = int(round(aASG_acc[3]))
            aA = actionAngleStaeckelGrid(pot=pot,delta=Delta,Rmax=Rmax,
                     nE=nE,npsi=npsi,nLz=nLz,numcores=numcores,c=True)
        else:
            #This is the galpy default:
            #aA = actionAngleStaeckelGrid(pot=pot,delta=Delta,Rmax=5.,
            #         nE=25,npsi=25,nLz=30,numcores=numcores,c=True)
            #This is the accuracy I did most of paper 1 with:
            #aA = actionAngleStaeckelGrid(pot=pot,delta=Delta,Rmax=10.,
            #         nE=50,npsi=50,nLz=60,numcores=numcores,c=True)
            #This accuracy should be slightly better (for Paper 1 - potential test, and for Paper 2):
            aA = actionAngleStaeckelGrid(pot=pot,delta=Delta,Rmax=5.,
                     nE=70,npsi=40,nLz=50,numcores=numcores,c=True)
    else:
        sys.exit("Error in setup_potential_and_action_object(): "+\
                 "potential type "+str(pottype)+" is not defined yet for "+\
                 "the action object creation.")

    return pot,aA

#--------------------------------------------------------------------

def setup_SelectionFunction_object(sftype,sfPar_phys,ro,df=None,**kwargs):
    """
        NAME:
           setup_SelectionFunction_object
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started setup_SelectionFunction_object.py on the basis of BovyCode/py/setup_SelectionFunction_object.py - Trick (MPIA)
           2016-09-20 - Incompleteness_function is now function of (R_galpy,phi_deg,z_galpy) in Galactocentric coordinates, and not anymore of r,z in solarcentric coordinates - Trick (MPIA)
           2016-09-16 - Added **kwargs to allow pre-calculated incompleteness in sftype=4. - Trick (MPIA)
           2016-12-12 - Added in sftype=4 (Incomplete Shell) the file_no=2 (red clump selection function for TGAS/RAVE with excl. disk). - Trick (MPIA)
           2017-04-08 - Added in sftype=4 (Incomplete Shell) the file_no=3 (red clump selection function for TGAS/RAVE with excl. disk and wrong abs. magnitudes to test SF.) - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    if sftype == 1: #Wedge-shaped selection function
        # sfPar_phys = [Rmin_kpc,Rmax_kpc,zmin_kpc,zmax_kpc,phimin_deg,phimax_deg]
        # R and z limits in galpy units
        sfRzs = numpy.array(sfPar_phys[0:4]) /_REFR0/ro   
        #    SF_Wedge(Rmin,Rmax,zmin,zmax,pmin,pmax,df=None)
        sf = SF_Wedge(
                sfRzs[0],       #Rmin
                sfRzs[1],       #Rmax
                sfRzs[2],       #zmin
                sfRzs[3],       #zmax
                sfPar_phys[4],   #pmin
                sfPar_phys[5],   #pmax
                df=df
                )
    elif sftype == 2:   #cylindrical selection function around sun
        # sfPar_phys = [robs_kpc,dobs_kpc,zmin_kpc,zmax_kpc]
        # size and position of sphere in galpy units
        sfTemp = numpy.array(sfPar_phys)  /_REFR0/ro  
        #    SF_Cylinder(rsun,dsun,zmin,zmax,df=None)
        sf = SF_Cylinder(
                sfTemp[0],      #r_obs
                sfTemp[1],      #d_obs
                sfTemp[2],      #zmin
                sfTemp[3],      #zmax
                df=df
                )
    elif sftype == 3:   #spherical selection function around sun
        # sfPar_phys = [robs_kpc,dobs_kpc]
        #size and position of sphere in galpy units
        sfTemp = numpy.array(sfPar_phys)  /_REFR0/ro  
        #    SF_Sphere(rsun,dsun,df=None)
        sf = SF_Sphere(
                sfTemp[0],      #r_obs
                sfTemp[1],      #d_obs
                df=df
                )
    elif sftype == 31:   #spherical selection function and incompleteness function
        # sfPar_phys = [robs_kpc,dobs_kpc,eps_r,eps_z]
        #size and position of sphere in galpy units
        sfTemp = numpy.array(sfPar_phys[0:2]) /_REFR0/ro
        r_obs  = sfTemp[0]
        d_obs  = sfTemp[1]
        degtorad = math.pi/180.
        #    SF_Sphere(rsun,dsun,df=None)
        sf = SF_Sphere(
                r_obs,      #r_obs
                d_obs,      #d_obs
                df=df
                )
        #incompleteness function:
        setup_incompleteness = False
        eps_r = sfPar_phys[2]
        eps_z = sfPar_phys[3]
        #     r: distance from sun
        #     z: height above/below plane
        if eps_r == 0. and eps_z == 0.:
            #no imcompleteness
            setup_incompleteness = False
        elif eps_z == 0. and eps_r != 0.:
            sys.exit('Error in setup_SelectionFunction_object(), sf_type=31: the incompleteness function with (R,z,phi) input has not been tested yet.')
            #incompleteness_function = lambda r, z: (1. - eps_r * r             / r_obs)    #obsolete as of 16-09-20
            incompleteness_function = lambda R,p_deg,z: (1. - eps_r / r_obs * numpy.sqrt(
                                                                                 (d_obs-R*numpy.cos(p_deg*degtorad))**2 +
                                                                                 (      R*numpy.sin(p_deg*degtorad))**2 +
                                                                                 z**2
                                                                                 ))
            setup_incompleteness = True
        elif eps_r == 0. and eps_z != 0.:
            #incompleteness_function = lambda r, z: (1. - eps_z * numpy.fabs(z) / r_obs)    #obsolete as of 16-09-20
            incompleteness_function = lambda R,p_deg,z: (1. - eps_z * numpy.fabs(z) / r_obs)
            setup_incompleteness = True
        else:
            sys.exit("Error in setup_SelectionFunction_object(): "+\
                     "this incompleteness function is not implemented yet.")
        if setup_incompleteness:
            # incompleteness_maximum = peak of completeness, used for 
            #         rejection sampling, as function does not 
            #         have to be normalized
            #incompleteness_maximum  = incompleteness_function(0.,0.)   #obsolete as of 16-09-20
            incompleteness_maximum  = incompleteness_function(d_obs,0.,0.)                              
            sf.set_incompleteness_function(
                    incompleteness_function,
                    incompleteness_maximum
                    )
    elif sftype == 32: #spherical selection function, box completeness, variable center
                # sfPar_phys = [d_max_kpc,R_cen_kpc,phi_cen_deg,z_cen_kpc]

        #    SF_Sphere(rsun,dsun,df=None)
        sf = SF_Sphere(
                sfPar_phys[0]/_REFR0/ro,      #d_max
                sfPar_phys[1]/_REFR0/ro,      #R_cen
                zcen      =sfPar_phys[3]/_REFR0/ro,   #z_cen
                phicen_deg=sfPar_phys[2],   #phi_cen
                df         =df
                )
        #with size and position of sphere in galpy units...

    elif sftype == 4: #incomplete shell selection function
        # sfPar_phys = [dmin_kpc,dmax_kpc,
        #               Rgc_sun_kpc,phigc_obs_deg,zgc_sun_kpc,
        #               file_no]

        #    SF_IncompleteShell(dmin,dmax,Rgc_Sun,
        #                       zgc_Sun=0.,phigc_Sun_deg=0.,
        #                       df=None,
        #                       SF_of_hpID_dkpc=None,NSIDE=None,dbin_kpc=None,galpy_to_kpc=None,  
        #                       SF_of_R_z=None,Rbin_kpc=None,zbin_kpc=None)

        #check, if incompleteness data was precalculated:
        SF_of_R_z = None
        Rbin_kpc  = None
        zbin_kpc  = None
        if 'SF_of_R_z_precalc' in kwargs: SF_of_R_z = kwargs['SF_of_R_z_precalc']
        if 'Rbin_kpc_precalc'  in kwargs: Rbin_kpc  = kwargs['Rbin_kpc_precalc']
        if 'zbin_kpc_precalc'  in kwargs: zbin_kpc  = kwargs['zbin_kpc_precalc']

        #use precalculated SF incompleteness data from keyword:
        if (SF_of_R_z is not None) and (Rbin_kpc is not None) and (zbin_kpc is not None):
            sf = SF_IncompleteShell(
                    sfPar_phys[0]/_REFR0/ro,      #dmin [galpy]
                    sfPar_phys[1]/_REFR0/ro,      #dmax [galpy]
                    sfPar_phys[2]/_REFR0/ro,      #Rgc_Sun [galpy]
                    zgc_Sun        =sfPar_phys[4]/_REFR0/ro, #zgc_Sun [galpy]
                    phigc_Sun_deg  =sfPar_phys[3], #phigc_Sun [deg]
                    df             =df,
                    SF_of_hpID_dkpc=None,
                    NSIDE          =None,
                    dbin_kpc       =None,
                    galpy_to_kpc   =_REFR0*ro,  
                    SF_of_R_z      =SF_of_R_z,
                    Rbin_kpc       =Rbin_kpc,
                    zbin_kpc       =zbin_kpc
                    )
        else: #use precalculated SF incompleteness data from file:
            file_no = int(sfPar_phys[5])
            if file_no in [1,2,3,4]:    #file_no=1: TGAS_RAVE_footprint, file_no=2: TGAS_RAVE_red_clump_SF, file_no=3: TGAS_RAVE_red_clump_testSF
                if   file_no == 1: selection_function_name = 'TGAS_RAVE_footprint'
                elif file_no == 2: selection_function_name = 'TGAS_RAVE_red_clump'
                elif file_no == 3: selection_function_name = 'TGAS_RAVE_red_clump_testSF'
                elif file_no == 4: selection_function_name = 'TGAS_RAVE_red_clump_lessQC'
                #1. Reading the SF as function of healpix_ID and distance from sun:
                filename_SF_hpID_dkpc = '../data/'+selection_function_name+'_SF_of_hpID_dkpc.sav'
                if os.path.exists(filename_SF_hpID_dkpc):
                    savefile        = open(filename_SF_hpID_dkpc,'rb')
                    NSIDE           = pickle.load(savefile)
                    dbin_kpc        = pickle.load(savefile)
                    SF_of_hpID_dkpc = pickle.load(savefile)
                    savefile.close()
                else:
                    sys.exit("Error in setup_SelectionFunction_object: "+filename_SF_hpID_dkpc+" does not exist.")
                dmin_kpc = str(sfPar_phys[0])
                dmax_kpc = str(sfPar_phys[1])
                Rsun_kpc = str(sfPar_phys[2])
                zsun_kpc = str(sfPar_phys[4])
                #2. Reading the SF as function of R and z:
                filename_SF_R_z = '../data/'+selection_function_name+'_SF_of_R_z_rmax='+dmax_kpc+'_rmin='+dmin_kpc+'_Rsun='+Rsun_kpc+'_zsun='+zsun_kpc+'.sav'
                if os.path.exists(filename_SF_R_z):
                    savefile  = open(filename_SF_R_z,'rb')
                    Rbin_kpc  = pickle.load(savefile)
                    zbin_kpc  = pickle.load(savefile)
                    SF_of_R_z = pickle.load(savefile)
                    savefile.close()
                else:
                    #If the file does not exist, create it:
                    print("..........File '"+filename_SF_R_z+"' does not exist yet.")
                    print("..........Now integrating over shell.")
                    sf = SF_IncompleteShell(
                            sfPar_phys[0]/_REFR0/ro,      #dmin [galpy]
                            sfPar_phys[1]/_REFR0/ro,      #dmax [galpy]
                            sfPar_phys[2]/_REFR0/ro,      #Rgc_Sun [galpy]
                            zgc_Sun        =sfPar_phys[4]/_REFR0/ro, #zgc_Sun [galpy]
                            phigc_Sun_deg  =sfPar_phys[3], #phigc_Sun [deg]
                            df             =df,
                            SF_of_hpID_dkpc=SF_of_hpID_dkpc,
                            NSIDE          =NSIDE,
                            dbin_kpc       =dbin_kpc,
                            galpy_to_kpc   =_REFR0*ro,  
                            SF_of_R_z      =None,
                            Rbin_kpc       =None,
                            zbin_kpc       =None
                            )
                    SF_of_R_z, Rbin_kpc, zbin_kpc = sf._prepare_and_set_SF_of_R_z(
                        nbin_R=400,nbin_z=400,border=0.1/_REFR0/ro,
                        plotfilename='../out/'+selection_function_name+'_SF_preparation_rmax='+dmax_kpc+'_rmin='+dmin_kpc+'_Rsun='+Rsun_kpc+'_zsun='+zsun_kpc+'.png',
                        savefilename=filename_SF_R_z
                        )
                    print("..........DONE!!!")
            else:
                sys.exit("Error in setup_SelectionFunction_object: file_no "+file_no+" is not known yet.")
            #    SF_IncompleteShell(dmin,dmax,Rgc_Sun,
            #                       zgc_Sun=0.,phigc_Sun_deg=0.,
            #                       df=None,
            #                       SF_of_hpID_dkpc=None,NSIDE=None,dbin_kpc=None,galpy_to_kpc=None,  
            #                       SF_of_R_z=None,Rbin_kpc=None,zbin_kpc=None)
            sf = SF_IncompleteShell(
                    sfPar_phys[0]/_REFR0/ro,      #dmin [galpy]
                    sfPar_phys[1]/_REFR0/ro,      #dmax [galpy]
                    sfPar_phys[2]/_REFR0/ro,      #Rgc_Sun [galpy]
                    zgc_Sun        =sfPar_phys[4]/_REFR0/ro, #zgc_Sun [galpy]
                    phigc_Sun_deg  =sfPar_phys[3], #phigc_Sun [deg]
                    df             =df,
                    SF_of_hpID_dkpc=SF_of_hpID_dkpc,
                    NSIDE          =NSIDE,
                    dbin_kpc       =dbin_kpc,
                    galpy_to_kpc   =_REFR0*ro,  
                    SF_of_R_z      =SF_of_R_z,
                    Rbin_kpc       =Rbin_kpc,
                    zbin_kpc       =zbin_kpc
                    )
            
    else:
        sys.exit("Error in setup_SelectionFunction_object(): this selection "+\
                 "function type is not defined.")

    return sf

#---------------------------------------------------------------------

def setup_MWlike_potential(potparams,ro,
                    interpDens=False,interpdvcircdr=False,
                    returnrawpot=False,ngrid=101):
    """
    NAME:
        setup potential
    PURPOSE:
        ???
    INPUT:
        potparams - ([float,float,float,float,float]) - Galaxy potential model parameters [R_d, f_h, v_c, z_h, d ln(vc) / d ln(r)], 
                                               i.e. [disk scale length, circular velocity at R_0, halo fraction, disk scale height, flattness of rotation curve at R0]
                                               units: galpy units!
        ro - (float) - = 8kpc/R_0, usually = 1, otherwise it corresponds to a different R_0 in the analysis
        interpDens - (bool, optional, default=False) - ???
        interpdvcircdr - (bool, optional, default=False) - ???
        returnrawpot - (bool, optional, default=False) - if set to False: Use an interpolated version of the potential for speed
        ngrid - (int) - if the potential is interpolated: number of grid points in each R and z direction on which potential is interpolated
    OUTPUT:
        ???
    HISTORY:
        201?-??-?? - Written - Bovy (???)
        2013-12-09 - Comments added - Trick (MPIA)
    """
    #_____potential parameters_____
    Rd        = potparams[0]     # stellar disk scale length in [galpy units]
    fh        = potparams[1]     # halo contribution to the disk's v_c^2
    vo        = potparams[2]     # circular velocity at R_0 = 1 in [_REFV0].
    zh        = potparams[3]     # stellar disk scale height
    dlnvcdlnr = potparams[4]     # slope of overall circular velocity rotation curve, d(ln(v_c)) / d(ln(r))
    
     #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    #_____bulge potential: Hernquist bulge______
    #Hernquist bulge parameters:
    _GMBULGE= 17208.0           # = G * M_bulge [kpc (km/s)^2] corresponds to 4 x 10^9 Msolar, mass of the Galaxy's Hernquist bulge times G - cf. p.7, 3.2, in Bovy & Rix (2013)
    _ABULGE= 0.6                # [kpc] - scale radius of the Galaxy's Hernquist bulge - cf. p.7, 3.2, in Bovy & Rix (2013)
    #     The default normalization is such, that 
    #         v_c^2 = R * |F_R| = R * (d Phi)/(d R) = 1 at R0.
    #     To normalize the potential properly, calculate the circular velocity of the desired Hernquist bulge (given formula) in physical units:
    #         v_c^2 = (GM/a) * (R/a) / (1+(R/a))^2 
    #     in [km/s], with R=R0=_REFR0*ro and a in [kpc]:
    bulge_vc2 = _GMBULGE/_ABULGE * (_REFR0*ro/_ABULGE) / (1.+(_REFR0*ro/_ABULGE))**2.   #[(km/s)^2]
    #     The amplitude of the total potential, i.e. the total v_c^2, in physical units is given by (vo*_REFV0)^2:
    total_vc2 = (_REFV0 * vo)**2.    #[(km/s)^2]
    #     And as the normalization of this potential component is given as the fraction of this potential's v_c^2 to the total potential,
    #     and , divide by (vo*_REFV0)^2.
    amplitude_bulge = bulge_vc2 / total_vc2
    #     Now initialize the potential with these scales:
    bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=amplitude_bulge)

    #______gas disk potential_____
    # Add an exponential gas disk with a local surface density of 13 Msol/pc^2 
    # with a scale height of 130 pc, and a scale length of twice the stellar disk scale length.
    # (cf. Bovy & Rix 2013, p. 8, left column)
    #
    # First initialize an unnormalized gas disk with correct scales:
    hz = 130./1000./ro/_REFR0                                               #gas disk scale height of 130 pc [galpy units]
    gp = potential.DoubleExponentialDiskPotential(hr=2.*Rd,                 #gas disk scale length = 2 * stellar disk scale length
                                                  hz=hz,                    #gas disk scale height
                                                  normalize=1.)             #no normalization (yet)
    # Secondly, calculate the surface density at R = 1, z = 0,
    gassurfdens = 2. * gp.dens(1.,0.) * gp._hz                                #[galpy units]
    gassurfdens *= bovy_conversion.surfdens_in_msolpc2(vo*_REFV0,ro*_REFR0)   #[Msun / pc^2]
    #     the same calculation by Jo: gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
    # Thirdly, use this to scale the disk potential to 13 Msol/pc^2:
    amplitude_gasdisk = 13. / gassurfdens   #Wie haengt die surface density mit der circular velocity zusammen???
    gp= potential.DoubleExponentialDiskPotential(hr=2.*Rd,
                                                 hz=hz,
                                                 normalize=amplitude_gasdisk)

    #_____contribution of disk and halo_____
    # si = |F_R,i| / |F_R,tot| = v_c,i^2 / v_c,tot^2
    sdsh = 1.-amplitude_gasdisk-amplitude_bulge    #from 1 = sd + sh + sg + sb, with sb = amplitude_bulge and sg = amplitude_gasdisk
    sd   = (1.-fh)*sdsh                            # = (1-fh)*(sd + sh), where fh = F_R,h / (F_R,h + F_R,d)
    sh   = sdsh-sd

    #_____stellar disk potential_____
    dp = potential.DoubleExponentialDiskPotential(hr=Rd,
                                                  hz=zh,
                                                  normalize=sd)    #normalize stellar disk such that it fits together with the free parameter fh and the bulge contribution
    
    #_____halo potential_____
    # first, calculate slope alpha of halo potential by considering given potential distributions of disk and bulge and the given slope of the rotation curve.
    plhalo= plhalo_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],sh)  #slope alpha of halo potential / spherical power-law potential
    if plhalo < 0. or plhalo > 3:
        raise RuntimeError("plhalo=%f" % plhalo)
    hp= potential.PowerSphericalPotential(alpha=plhalo,
                                          normalize=sh)
                                          
    #_____return potential_____                              
    if returnrawpot:
        return [dp,hp,bp,gp]
    else:
        #_____Use an interpolated version for speed_____
        #setup spatial grid on which the potential is interpolated: R = [0.01,20] and z = [0,1]
        return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],
                                           rgrid=(numpy.log(0.01),numpy.log(20.),ngrid),
                                           zgrid=(0.,1.,ngrid),
                                           logR=True,                   #the input of rgrid is in log
                                           interpepifreq=True,interpverticalfreq=True,interpvcirc=True,interpPot=True,  #these functions will be interpolated
                                           interpDens=interpDens,interpdvcircdr=interpdvcircdr,
                                           use_c=True,enable_c=True)    #use c for speeding up of interpolation
         #TO DO: How does interpRZPotential work????


#---------------------------------------------------------------------------------

def plhalo_from_dlnvcdlnr(dlnvcdlnr,diskpot,bulgepot,fh):
    """
    NAME:
        plhalo_from_dlnvcdlnr
    PURPOSE:
        Calculate the halo's shape corresponding to this rotation curve derivative.
        The halo has the shape Phi = - A R^(2-alpha) / (alpha-2). This function returns alpha.
    INPUT:
        dlnvcdlnr - d ln(vc) / d ln(r), flattness of rotation curve at R0
        diskpot - potential of the disk normalized such that the total potential [bulge,gas disk,stellar disk,halo] 
                  causes a circular velocity v_c=1 at R0
        bulgepot - potential of the bulge (and gas disk) normalized analogous
        fh - this is NOT the halo fraction fh = F_R,h / (F_R,h + F_R,d) used as free potential parameter, but rather fh --> sh = F_R,h / F_R,tot
    OUTPUT:
        ???
    HISTORY:
        201?-??-?? - Written - Bovy (???)
        2013-12-09 - some comments added - Trick (MPIA)
        2019-12-12 - changed input parameter order of evaluateRforces() - Trick (MPA)
    """
    #Method:
    #1. The second derivative of the halo potential, which we will use in the following, is:
    #   d^2 Phi_h / d R^2 = (1-alpha) * A * R^(-alpha) = (alpha-1) * F_R,h / R < 0,
    #   as F_R,h < 0.
    #2. The derivative of v_c^2 is:
    #   d v_c^2 / d R =   2 * (v_c^2 / R) * dlnvcdlnr
    #                 = - 2 * F_R,tot     * dlnvcdlnr
    #   where dlnvcdlnr = (d(ln v_c) / d(ln R)) is the given shape of the overall rotation curve. And F_R,tot < 0.
    #3. The derivative of v_c^2 is also:
    #   v_c^2 = R * d Phi / d R
    #   d v_c^2 / d R = sum_i [d v_c,i^2 / d R]                 with i = [b,g,d,h] = [bulge,gas disk,stellar disk,halo]
    #                 = sum_i [-F_R,i + R * d^2 Phi_i / d R^2]
    #                 = sum_j [d v_c,j^2 / d R]                 - F_R,h + R * d^2 Phi_h / d R^2         with j = [b,g,d]
    #                 = sum_j [d v_c,j^2 / d R]                 - F_R,h + R * (alpha-1) * F_R,h / R    (using 1.)
    #                 = sum_j [d v_c,j^2 / d R]                 - (2-alpha) F_R,h
    #4. by equating 2. and 3. and solving for alpha, we get:
    #   alpha = 2 - [2 * F_R,tot * dlnvcdlnr + sum_j (d v_c,j^2 / d R)            ] / F_R,h
    #         = 2 - [2 *           dlnvcdlnr + sum_j (d v_c,j^2 / d R) / F_R,tot  ] * F_R,tot / F_R,h
    #         = 2 - [2 *           dlnvcdlnr - sum_j (d v_c,j^2 / d R) /|F_R,tot| ] *|F_R,tot|/|F_R,h|
    #         = 2 - [2 *           dlnvcdlnr - sum_j (d v_c,j^2 / d R) /|F_R,tot| ] / sh                  with s_h = |F_R,h| / |F_R,tot| > 0
    #5. |F_R,tot| = 1 in this normalization at R=1.

    #First calculate the derivatives dvc^2/dR of disk and bulge     
    dvcdr_disk= -potential.evaluateRforces(diskpot,1.,0.)+potential.evaluateR2derivs(diskpot,1.,0.)
             #= -F_R(R=1,z=0,disk)                       +d^2 Phi / d R^2 (R=1,z=0,disk)      #this is d v_circ^2 / dR = - F_R + R * d^2 Phi / d R^2
    dvcdr_bulge= -potential.evaluateRforces(bulgepot,1.,0.)+potential.evaluateR2derivs(bulgepot,1.,0.)
             #= -F_R(R=1,z=0,bulge)                        +d^2 Phi / d R^2 (R=1,z=0,bulge)  
    return 2.-(2.*dlnvcdlnr-dvcdr_disk-dvcdr_bulge)/fh

#--------------------------------------------------------------------------------------------------------------------

def setup_DiskHaloBulge_potential(potparams,Disk='MiyamotoNagai',Halo='Hernquist',a_bulge_kpc=None,M_bulge_1010Msun=None,s_bulge=None):
    """
    NAME:
        setup_DiskHaloBulge_potential
    PURPOSE:
        This function can be used to setup a simple Hernquist Bulge + Disk + Halo potential.
        Pottype 5 and 51 uses this function and sets up a Miyamoto-Nagai disk, 
        Hernquist halo + Hernquist bulge for the Elena D'Onghias Simulation.
        I also use it to fit a MNdHbNFWh-Potential to Gaia data.
    INPUT:
        a_bulge_kpc - (float) - scale length of the Hernquist bulge in kpc. 
                                For Pottype 5 and 51 use: a_bulge_kpc=0.25099812
        M_bulge_1010Msun - (float) - total mass of the Hernquist bulge in 10^10 M_sun.
                                For Pottype 5 and 51 use: M_bulge_1010Msun=0.952400755715
        s_bulge - (float) - normalization constant of the bulge: FR_bulge/FR_total. 
                            If M_bulge_1010Msun is set, this is not needed.
    OUTPUT:
    HISTORY:
        2016-01-18 - written - Trick (MPIA)
        2016-02-15 - rewrote and renamed this function to make it more flexible and to use different combinations of disks, halos and bulges. - Trick (MPIA)
        2017-01-17 - Added MN3_ExponentialSechDisk. - Trick (MPIA)
    """

    #_____potential parameters_____
    ro  = potparams[0]     # solar position in [_REFR0]
    vo  = potparams[1]     # circular velocity at R_0 = 1 in [_REFV0].
    a_d = potparams[2]     # stellar disk scale length in [galpy units] 
    b_d = potparams[3]     # stellar disk scale height in [galpy units]
    f_h = potparams[4]     # halo contribution to the disk's v_c^2
    a_h = potparams[5]     # dark matter halo scale length in [galpy units]

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    #_____bulge potential: Hernquist bulge______
    if a_bulge_kpc is None:
        sys.exit("Error in setup_DiskHaloBulge_potential(): "+\
                 "To set-up the Hernquist bulge a scale length a in "+\
                 "kpc (a_bulge_kpc) must be specified.")
    if M_bulge_1010Msun is not None and s_bulge is None:
        #grav. constant:
        G = bovy_conversion._G/1000.*1e10 #(km/s)^2 * kpc / 10^10 M_odot
        #setup potential in physical units:
        amp_bulge = 2. * M_bulge_1010Msun * G
        bulgepot = potential.HernquistPotential(
                        amp      =amp_bulge,
                        a        =a_bulge_kpc,
                        normalize=False
                        )
        #normalize:
        FR_bulge = bulgepot.Rforce(ro*_REFR0,0.)    #[(km/s)^2 / kpc]
        FR_total = -(vo*_REFV0)**2/(ro*_REFR0)     #[(km/s)^2 / kpc]
        s_bulge  = FR_bulge/FR_total
    elif (s_bulge is not None and M_bulge_1010Msun is not None) or \
         (s_bulge is     None and M_bulge_1010Msun is     None):
        sys.exit("Error in setup_DiskHaloBulge_potential(): "+\
                 "To set-up the Hernquist bulge one and only one of "+\
                 "the two parameters M_bulge_1010Msun and s_bulge is "+\
                 "needed to normalize the bulge.")
    #setup potential in galpy units:
    bulgepot = potential.HernquistPotential(
                    a        =a_bulge_kpc/_REFR0/ro,
                    normalize=s_bulge
                    )

    #_____setup relative contributions_____
    #(1) 1 = s_bulge + s_disk + s_halo
    #(2) f_h = s_halo / (s_disk + s_halo)
    s_halo = f_h * (1.-s_bulge)
    s_disk = 1. - s_bulge - s_halo

    if Disk == 'MiyamotoNagai':
        #_____disk potential: Miyamoto-Nagai disk_____
        diskpot = potential.MiyamotoNagaiPotential(
                        a        =a_d, 
                        b        =b_d, 
                        normalize=s_disk)
    elif Disk == 'DoubleExponential':
        #_____disk potential: DoubleExponentialDisk_____
        diskpot = potential.DoubleExponentialDiskPotential(
                        hr        =a_d, 
                        hz        =b_d, 
                        normalize=s_disk)
    elif Disk == 'MN3_ExponentialSechDisk':
        #_____disk potential: 3xMiyamoto-Nagai approximation to ExponentialSechDisk_____
        #Smith et al. 2015 <http://adsabs.harvard.edu/abs/2015arXiv150200627S>
        diskpot = potential.MN3ExponentialDiskPotential(
                        hr        =a_d, 
                        hz        =b_d, 
                        normalize =s_disk,
                        sech      =True,
                        posdens   =True
                        )
    else:
        sys.exit("Error in setup_DiskHaloBulge_potential(): "+\
                 "Disk type "+Disk+" is not known. So far "+\
                 "implemented are: MiyamotoNagai and DoubleExponential.")

    #_____halo potential: Hernquist halo_____
    if Halo == 'Hernquist':
        halopot = potential.HernquistPotential(
                        a        =a_h,
                        normalize=s_halo
                        )
    elif Halo == 'NFW':
        halopot = potential.NFWPotential(
                        a        =a_h,
                        normalize=s_halo
                        )
    else:
        sys.exit("Error in setup_DiskHaloBulge_potential(): "+\
                 "Halo type "+Halo+" is not known. So far "+\
                 "implemented are: Hernquist and NFW.")

    return [diskpot,halopot,bulgepot]
