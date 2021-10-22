#_____import packages_____
#from __past__ import division
from precalc_actions import precalc_pot_actions_sf
import numpy
import sys
import multiprocessing
import multiprocessing.sharedctypes
import ctypes
from setup_pot_and_sf import setup_SelectionFunction_object

#=================================================================================

def shared_data_DFfit_only_MCMC(pottype,sftype,datatype,dftype,
                            potParEst_phys,dfParFid_fit,sfParEst_phys,
                            R_data,vR_data,vT_data,z_data,vz_data,
                            ro_known,
                            _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,
                            _MULTI,
                            aASG_accuracy,use_default_Delta,estimate_Delta,Delta_fixed,
                            current_path):

    """
    NAME:
        logprob_MCMC_fitDF_only
    PURPOSE:
    INPUT:
        p - (float numpy array)
          - current walker position in potential and qdf parameter space [potPar,dfPar]
    OUTPUT:
    HISTORY:
        2016-02-?? - Written. - Trick (MPIA)
        2016-04-15 - Added the parameters governing the actionAngle Delta and accuracy to precalc_pot_actions_sf().
        2017-01-03 - Added dftype. - Trick (MPIA)
    """

    #_____Reference scales_____
    _REFR0 = 8.                 #[kpc]
    _REFV0 = 220.               #[km/s]

    #_____scale data to galpy units_____
    ro = potParEst_phys[0] / _REFR0
    vo = potParEst_phys[1] / _REFV0
    R_galpy   = R_data /ro
    vR_galpy  = vR_data/vo
    vT_galpy  = vT_data/vo
    z_galpy   = z_data /ro
    vz_galpy  = vz_data/vo
    if datatype == 3: #perfect mock data, marginalization over one coordinate
        sys.exit("Error in shared_data_DFfit_only_MCMC(): "+\
                 "The special case of marginalizing over a coordinate "+\
                 "is not yet implemented. For a normal fit of pot and DF "+\
                 "this is done in logprob_MCMC(), but logprob_MCMC_fitDF_only() "+\
                 "uses only actions, no actual phase-space data.")

    #_____precalculate all actions_____
    pot,aA,sf,data_actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,dftype,
                            potParEst_phys,dfParFid_fit,sfParEst_phys,
                            R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                            ro_known,
                            _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,
                            _MULTI,
                            aASG_accuracy,use_default_Delta,estimate_Delta,Delta_fixed
                            )

    #_____setup shared memory for data actions_____
    #define the shared memory array
    ndata = len(R_galpy)
    data_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,7*ndata)
    #copy values into the shared array        
    data_shared_base[:] = data_actions.flatten()
    #this array is used to access the shared array
    data_shared = numpy.frombuffer(data_shared_base)
    #reshape the array to match the shape of the original array
    data_shared = data_shared.reshape((7,ndata))

    #_____setup shared memory for fiducial actions_____
    #define the shared memory array
    nfid = _N_SPATIAL_R * _N_SPATIAL_Z * _NGL_VELOCITY**3
    fid_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,7*nfid)
    #copy values into the shared array        
    fid_shared_base[0     :  nfid] = sf._jr_fid.flatten()    #jr_fiducial
    fid_shared_base[  nfid:2*nfid] = sf._lz_fid.flatten()    #lz_fiducial
    fid_shared_base[2*nfid:3*nfid] = sf._jz_fid.flatten()    #jz_fiducial
    fid_shared_base[3*nfid:4*nfid] = sf._rg_fid.flatten()    #rg_fiducial
    fid_shared_base[4*nfid:5*nfid] = sf._kappa_fid.flatten() #kappa_fiducial
    fid_shared_base[5*nfid:6*nfid] = sf._nu_fid.flatten()    #nu_fiducial
    fid_shared_base[6*nfid:7*nfid] = sf._Omega_fid.flatten() #Omega_fiducial
    #this array is used to access the shared array
    fid_shared = numpy.frombuffer(fid_shared_base)
    #reshape the array to match the shape of the original array
    fid_shared = fid_shared.reshape((7,_N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY**3))

    #_____save shared data_____
    #write current path into file:
    shared_data_filename     = current_path+'_shared_data_actions.npy'
    shared_fiducial_filename = current_path+'_shared_fiducial_actions.npy'
    #write data actions into binary file:
    numpy.save(shared_data_filename,data_shared)
    #write fiducial actions into binary file:
    numpy.save(shared_fiducial_filename,fid_shared)

#=================================================================================

def shared_data_MCMC(R_data,vR_data,vT_data,z_data,vz_data,
                     current_path):

    #_____setup shared memory for data_____
    #define the shared memory array
    ndata = len(R_data)
    data_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,5*ndata)
    #copy values into the shared array        
    data_shared_base[0      :  ndata] =  R_data
    data_shared_base[  ndata:2*ndata] = vR_data
    data_shared_base[2*ndata:3*ndata] = vT_data
    data_shared_base[3*ndata:4*ndata] =  z_data
    data_shared_base[4*ndata:5*ndata] = vz_data
    #this array is used to access the shared array
    data_shared = numpy.frombuffer(data_shared_base)
    #reshape the array to match the shape of the original array
    data_shared = data_shared.reshape((5,ndata))

    #_____save shared data_____
    #write current path into file:
    shared_data_filename = current_path+'_shared_data.npy'
    #write data into binary file:
    numpy.save(shared_data_filename,data_shared)

#=================================================================================

def shared_data_incompleteShell(sftype,sfPar_phys,ro,
                                current_path):

    #_____load pre-calculated incompleteness data_____
    sf = setup_SelectionFunction_object(
                sftype,
                sfPar_phys,
                ro,
                df=None
                )
    SF_of_R_z = sf._incomp_SF_of_R_z
    Rbin_kpc  = sf._incomp_Rbin_kpc
    zbin_kpc  = sf._incomp_zbin_kpc

    #_____setup shared memory for data_____
    #define the shared memory array
    nR = len(Rbin_kpc)
    nz = len(zbin_kpc)
    ndata = nR+nz+numpy.shape(SF_of_R_z)[0]*numpy.shape(SF_of_R_z)[1]+2
    data_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,ndata)
    #copy values into the shared array
    data_shared_base[0]            = nR
    data_shared_base[1]            = nz
    data_shared_base[2:nR+2]       = Rbin_kpc
    data_shared_base[nR+2:nR+nz+2] = zbin_kpc
    data_shared_base[nR+nz+2::]   = SF_of_R_z.flatten()
    #this array is used to access the shared array
    data_shared = numpy.frombuffer(data_shared_base)

    #_____save shared data_____
    #write current path into file:
    shared_data_filename = current_path+'_shared_SF_incompleteShell.npy'
    #write data into binary file:
    numpy.save(shared_data_filename,data_shared)


