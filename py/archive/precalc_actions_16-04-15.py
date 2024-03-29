#_____import packages_____
import math
import multiprocessing
import numpy
import sys
import time
from galpy.df import quasiisothermaldf
from galpy.util import multi
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object,setup_SelectionFunction_object

#-------------------------------------------------------------------

def setup_data_actions(pot,aA,
                       R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,   #data in galpy units
                       dfParFid_galpy,ro,
                       _MULTI):

    """
        NAME:
           setup_data_actions
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started setup_data_actions.py on the basis of BovyCode/py/setup_data_actions.py - Trick (MPIA)
    """


    #.....setup fiducial qdf.....
    # (the fiducial qdf is only used to calculate the actions 
    # and frequencies. The qdf parameters are not used in any way 
    # and are therefore arbitrarily set.)
    qdf_fid = quasiisothermaldf(
                        dfParFid_galpy[0], #hr
                        dfParFid_galpy[1], #sr
                        dfParFid_galpy[2], #sz
                        dfParFid_galpy[3], #hsr
                        dfParFid_galpy[4], #hsz   
                        pot=pot,aA=aA,
                        cutcounter=True, 
                        ro=ro
                        )

    #.....calculate actions and frequencies using the fiducial qdf.....
    # by evaluating the action angle object within the qdf for each 
    # mock data star coordinates
    # out = (lnqdf,jr,lz,jz,rg,kappa,nu,Omega)
    ndata      = len(R_galpy)
    jr_data    = numpy.zeros(ndata)
    lz_data    = numpy.zeros(ndata)
    jz_data    = numpy.zeros(ndata)
    rg_data    = numpy.zeros(ndata)
    kappa_data = numpy.zeros(ndata)
    nu_data    = numpy.zeros(ndata)
    Omega_data = numpy.zeros(ndata)
    if _MULTI is None or _MULTI == 1:
        #...evaluation on one core:
        out = qdf_fid(
                    R_galpy,
                    vR_galpy,
                    vT_galpy,
                    z_galpy,
                    vz_galpy,
                    log=True,
                    _return_actions=True,
                    _return_freqs=True
                    )
        jr_data    = out[1]
        lz_data    = out[2]
        jz_data    = out[3]
        rg_data    = out[4]
        kappa_data = out[5]
        nu_data    = out[6]
        Omega_data = out[7]    
    elif _MULTI > 1:
        #...evaluation on multiple cores:

        # number of cores to use: N
        N = numpy.amin([
                ndata,
                multiprocessing.cpu_count(),
                _MULTI
                ])

        # data points to evaluate on one core:
        M = int(math.floor(ndata / N))

        # first evaluate arrays on each core to make use of 
        # the fast evaluation of input arrays:
        multiOut =  multi.parallel_map(
                                (lambda x: qdf_fid(
                                    R_galpy [x*M:(x+1)*M],
                                    vR_galpy[x*M:(x+1)*M],
                                    vT_galpy[x*M:(x+1)*M],
                                    z_galpy [x*M:(x+1)*M],
                                    vz_galpy[x*M:(x+1)*M],
                                    log=True,
                                    _return_actions=True,
                                    _return_freqs=True
                                    )),
                                range(N),
                                numcores=N
                                )
        for x in range(N):
            jr_data   [x*M:(x+1)*M] = multiOut[x][1]
            lz_data   [x*M:(x+1)*M] = multiOut[x][2]
            jz_data   [x*M:(x+1)*M] = multiOut[x][3]
            rg_data   [x*M:(x+1)*M] = multiOut[x][4]
            kappa_data[x*M:(x+1)*M] = multiOut[x][5]
            nu_data   [x*M:(x+1)*M] = multiOut[x][6]
            Omega_data[x*M:(x+1)*M] = multiOut[x][7]

        # number of data points not yet evaluated:
        K = ndata % N

        # now calculate the rest of the data:
        if K > 0:
            
            multiOut =  multi.parallel_map(
                                    (lambda x: qdf_fid(
                                        R_galpy [N*M+x],
                                        vR_galpy[N*M+x],
                                        vT_galpy[N*M+x],
                                        z_galpy [N*M+x],
                                        vz_galpy[N*M+x],
                                        log=True,
                                        _return_actions=True,
                                        _return_freqs=True
                                        )),
                                    range(K),
                                    numcores=numpy.amin([
                                        K,
                                        multiprocessing.cpu_count(),
                                        _MULTI
                                        ])
                                    )
            for x in range(K):
                jr_data   [N*M+x] = multiOut[x][1]
                lz_data   [N*M+x] = multiOut[x][2]
                jz_data   [N*M+x] = multiOut[x][3]
                rg_data   [N*M+x] = multiOut[x][4]
                kappa_data[N*M+x] = multiOut[x][5]
                nu_data   [N*M+x] = multiOut[x][6]
                Omega_data[N*M+x] = multiOut[x][7]
    
    actions = numpy.zeros((7,ndata))
    actions[0,:] = jr_data
    actions[1,:] = lz_data
    actions[2,:] = jz_data
    actions[3,:] = rg_data
    actions[4,:] = kappa_data
    actions[5,:] = nu_data
    actions[6,:] = Omega_data

    return actions
            
            

#------------------------------------------------------------------------------------------


def precalc_pot_actions_sf(pottype,sftype,
                        potPar_phys,dfParFid_fit,sfParEst_phys,
                        R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                        ro_known,
                        _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_MULTI):

    """
        NAME:
           precalc_pot_actions_sf
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started precalc_pot_actions_sf.py on the basis of BovyCode/py/precalc_pot_actions_sf.py - Trick (MPIA)
           2016-02-18 - Added _MULTI keyword to setup_Potential_and_ActionAngle_object().
    """

    #_____Reference scales_____
    _REFR0 = 8.                 #[kpc]
    _REFV0 = 220.               #[km/s]

    #_____setup potential / Action object_____
    #print "Start potential setup"
    #start_1 = time.time()
    try:
        pot, aA = setup_Potential_and_ActionAngle_object(
                        pottype,
                        potPar_phys,
                        _MULTI=_MULTI
                        )
        pot_physical = True
    except RuntimeError as e:
        pot = None
        aA  = None
        sf  = None
        actions = None
        pot_physical = False
        return pot,aA,sf,actions,pot_physical
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0
    #zeit_t = time.time() - start_1
    #print "POTENTIAL: ",round(zeit_t,2)," sec"

    #_____rescale fiducial qdf parameters to galpy units_____
    #these parameters are used to set the integration grid 
    #of the qdf over velocities to get the density.
    traf = numpy.array([ro,vo,vo,ro,ro])
    dfParFid_galpy = numpy.exp(dfParFid_fit) / traf

    #_____calculate actions and frequencies of the data_____
    #before calculating the actions, the data is properly 
    #scaled to galpy units with the current potentials vo and ro
    #start_2 = time.time()
    if pottype == 1: #isochrone
        actions = setup_data_actions(pot,aA,
               R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
               dfParFid_galpy,ro,
               1)  #somehow evaluation on only one core seems to be fastest with the isochrone
    else:
        actions = setup_data_actions(pot,aA,
               R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
               dfParFid_galpy,ro,
               _MULTI)
    #zeit_t = time.time() - start_2
    #print "DATA ACTIONS: ",round(zeit_t,2)," sec"


    #_____initialize selection function_____
    if ro_known:
        sf = setup_SelectionFunction_object(
                        sftype,
                        sfParEst_phys,
                        ro
                        )
    else:
        sys.exit("Error in precalc_pot_actions_sf(): "+\
                 "How to deal with the selection function if ro is not known? To do: Implement.")

    #_____setup fiducial qdf and calculate actions on velocity grid_____
    # velocity grid that corresponds to the sigmas of the fiducial qdf.
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
    #start_3 = time.time()
    sf.set_fiducial_df_actions_Bovy(
               qdf_fid,
               nrs=_N_SPATIAL_R,nzs=_N_SPATIAL_Z,
               ngl_vel=_NGL_VELOCITY,
               n_sigma=_N_SIGMA,
               vT_galpy_max=_VT_GALPY_MAX,
               _multi=_MULTI,
               )
    """#setup grid in (R,z,vR,vT,vz) for integration over velocity at (R,z):
    sf.setup_velocity_integration_grid(
                qdf_fid,
                nrs=_N_SPATIAL_R,nzs=_N_SPATIAL_Z,
                ngl_vel=_NGL_VELOCITY,
                n_sigma=_N_SIGMA,
                vT_galpy_max=_VT_GALPY_MAX,
                )
    #calculate actions at each grid point on the grid above:
    sf.calculate_actions_on_vig_using_fid_pot(qdf_fid,_multi=_MULTI)"""
    #zeit_t = time.time() - start_3
    #print "FIDUCIAL ACTIONS: ",round(zeit_t,2)," sec"

    return pot,aA,sf,actions,pot_physical

#-------------------------------------------------------------------
