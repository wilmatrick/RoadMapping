
def shared_data_DFfit_only_MCMC(pottype,sftype,datatype,
                            potPar_phys,dfParFid_fit,sfParEst_phys,
                            R_data,vR_data,vT_data,z_data,vz_data,
                            ro_known,
                            _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_MULTI,
                            current_path):

    #_____Reference scales_____
    _REFR0 = 8.                 #[kpc]
    _REFV0 = 220.               #[km/s]

    #_____scale data to galpy units_____
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0
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
    pot,aA,sf,data_actions,pot_physical = precalc_pot_actions_sf(pottype,sftype,
                            potPar_phys,dfParFid_fit,sfParEst_phys,
                            R_galpy,vR_galpy,vT_galpy,z_galpy,vz_galpy,
                            ro_known,
                            _N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY,_N_SIGMA,_VT_GALPY_MAX,_MULTI)

    #_____setup shared memory for data actions_____
    #define the shared memory array
    ndata = len(R_galpy)
    data_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,7*ndata)
    #copy values into the shared array        
    data_shared_base[:] = actions.flatten()
    print "Could be that this flattening does not work..."
    #this array is used to access the shared array
    data_shared = numpy.frombuffer(data_shared_base)
    #reshape the array to match the shape of the original array
    data_shared = data_shared.reshape((7,ndata))


    #_____setup shared memory for fiducial actions_____
    #define the shared memory array
    nfid = _N_SPATIAL_R * _N_SPATIAL_Z * _NGL_VELOCITY**3
    fid_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,7*nfid)
    #copy values into the shared array        
    fid_shared_base[0     :  nfid] = sf._jr_fid    #jr_fiducial
    fid_shared_base[  nfid:2*nfid] = sf._lz_fid    #lz_fiducial
    fid_shared_base[2*nfid:3*nfid] = sf._jz_fid    #jz_fiducial
    fid_shared_base[3*nfid:4*nfid] = sf._rg_fid    #rg_fiducial
    fid_shared_base[4*nfid:5*nfid] = sf._kappa_fid #kappa_fiducial
    fid_shared_base[5*nfid:6*nfid] = sf._nu_fid    #nu_fiducial
    fid_shared_base[6*nfid:7*nfid] = sf._Omega_fid #Omega_fiducial
    #this array is used to access the shared array
    fid_shared = numpy.frombuffer(fid_shared_base)
    #reshape the array to match the shape of the original array
    fid_shared = fid_shared.reshape((7,_N_SPATIAL_R,_N_SPATIAL_Z,_NGL_VELOCITY**3))

    #Setting up the StaeckelFudge ActionGrid is very slow. 
    #As we precalculate all actions anyway, we use the standard StaeckelFudge
    #to set up the potential in the MCMC chain.
    if pottype in numpy.array([21,31,51,61,71]): 
        pottype_slim = (pottype-1)/10
        info_MCMC['pottype'] = pottype_slim
    print "TO DO: test, if this works. Or, rather do this in the main code. It is less confusing to change global parameters there."

    #_____save shared data_____
    #write current path into file:
    shared_data_filename = current_path+'_shared_data_actions.npy'
    shared_fiducial_filename = current_path+'_shared_fiducial_actions.npy'
    #write data actions into binary file:
    numpy.save(shared_data_filename,data_shared)
    #write fiducial actions into binary file:
    numpy.save(shared_fiducial_filename,fiducial_shared)

#=================================================================================

def shared_data_MCMC(R_data,vR_data,vT_data,z_data,vz_data,
                     noStars,_N_ERROR_SAMPLES,
                     current_path):

    #_____setup shared memory for data_____
    #define the shared memory array
    ndata = noStars*_N_ERROR_SAMPLES
    data_shared_base = multiprocessing.sharedctypes.RawArray(ctypes.c_double,5*ndata)#multiprocessing.Array(ctypes.c_double,5*ndata)#numpy.zeros((5,noStars*_N_ERROR_SAMPLES))
    #copy values into the shared array        
    data_shared_base[0      :  ndata] =  R_data
    data_shared_base[  ndata:2*ndata] = vR_data
    data_shared_base[2*ndata:3*ndata] = vT_data
    data_shared_base[3*ndata:4*ndata] =  z_data
    data_shared_base[4*ndata:5*ndata] = vz_data
    #this array is used to access the shared array
    data_shared = numpy.frombuffer(data_shared_base)#numpy.ctypeslib.as_array(data_shared_base.get_obj())
    #reshape the array to match the shape of the original array
    data_shared = data_shared.reshape((5,ndata))

    #_____save shared data_____
    #write current path into file:
    shared_data_filename = current_path+'_shared_data.npy'
    #write data into binary file:
    numpy.save(shared_data_filename,data_shared)

