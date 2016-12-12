#_____import packages_____
import math
import scipy.optimize
import scipy.interpolate
import numpy
import sys
import pickle

#_____exponential dispersion function to fit_____
def mloglike_vel_disp(x,*args):
    
    #fit parameters (scalar input only!):
    sig_0 = x[0]
    h_sig = x[1]
    
    #data:
    Rs    = args[0]
    vs    = args[1]

    #fixed parameters:
    Rsun  = args[2]

    #velocity dispersion at R:
    disp = sig_0 * numpy.exp(-(Rs-Rsun) / h_sig)    
    
    #velocity distribution / likelihood:
    #p(v,R) = N(0,disp(R)):
    log_p = -(vs - 0.)**2 / (2. * disp**2) - numpy.log(numpy.sqrt(2.* math.pi * disp**2))

    #minus likelihood (for minimizing the function):
    return -numpy.sum(log_p)

#_____exponential density function to fit_____
def mloglike_dens(x,*args):
    
    #fit parameters (scalar input only!):
    h_R = x
    
    #data:
    Rs  = args[0]

    #fixed parameters:
    Rmin  = args[1]
    Rmax  = args[2]
    
    # normalized log of tracer distribution / likelihood:
    # p(R) = R * exp(-R/h_R) / Mtot
    # Mtot = h_R * (h_R + Rmin) * exp(-Rmin/h_R) - h_R * (h_R + Rmax) * exp(-Rmax/h_R)
    Mtot   = h_R * (h_R + Rmin) * numpy.exp(-Rmin/h_R) - h_R * (h_R + Rmax) * numpy.exp(-Rmax/h_R)
    log_p = -Rs / h_R + numpy.log(Rs) - numpy.log(Mtot)

    #minus likelihood (for minimizing the function):
    return -numpy.sum(log_p)

#_____the main function to call_____
def estimate_fiducial_qdf(R_kpc=None,vR_kms=None,
                          phi_deg=None,vT_kms=None,
                          z_kpc=None,vz_kms=None,
                          Rsun_kpc=None,
                          Rmin_kpc=None,Rmax_kpc=None,zcen_kpc=0.,phicen_deg =0.,
                          sftype=None):

    """
        NAME:
           estimate_fiducial_qdf
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-27 - Started estimate_fiducial_qdf.py on the basis of BovyCode/py/estimate_fiducial_qdf.py - Trick (MPIA)
                      - Deleted ro and vo keywords. Not needed.
                      - Added zcen_kpc=0. and phicen_deg=0. keywords.
    """

    #_____global constants_____  
    _REFR0 = 8.        #spatial scaling
    _REFV0 = 220.      #velocity scaling

    #_____fitting velocity disersion profiles_____

    #...radial velocity dispersion...

    #initial guess:
    x0 = [0.5*(30.+60.),8.] #[km/s,kpc]
    #boundary conditions:
    bounds = ((10.,70.),(4.,16.))
    #   this comes from fig. 6 in Bovy & Rix 2013, second panel
    #   and p.10, second column, hsr = 8 kpc.
    
    #minimize:
    res = scipy.optimize.minimize(mloglike_vel_disp, x0, args=(R_kpc,vR_kms,Rsun_kpc), bounds=bounds)
    sig_R   = res.x[0]
    h_sig_R = res.x[1]
    if not res.success:
        print "Problem in radial velocity dispersion: ",res.message

    #...vertical velocity dispersion...

    #initial guess:
    x0 = [0.5*(16.+80.),5.] #[km/s,kpc]
    #boundary conditions:
    bounds = ((1.,90.),(4.,16.))
    #   this comes from fig. 6 in Bovy & Rix 2013, third panel
    #   and p.10, second column for the fitting limits of hsr
    
    #minimize:
    res = scipy.optimize.minimize(mloglike_vel_disp, x0, args=(R_kpc,vz_kms,Rsun_kpc), bounds=bounds)
    sig_z   = res.x[0]
    h_sig_z = res.x[1]
    if not res.success:
        print "Problem in vertical velocity dispersion: ",res.message

    #____fitting the tracer scale length_____

    #...selecting tracers for h_R...
    if sftype == 3 or sftype == 32: #spherical selection function
        z_min    = zcen_kpc
        phi_mean = phicen_deg
        d_z      = 1.5 * numpy.std(z_kpc)
        d_phi    = numpy.std(phi_deg)
    else:
        sys.exit("Error in estimate_fiducial_qdf(): The selection function type "+sftype+" has not been implemented so far.")
        #z_min    = min(numpy.fabs(z_kpc))
        #phi_mean = numpy.mean(phi_deg)
        #d_z      = 1.5 * numpy.std(z_kpc)
        #d_phi    = numpy.std(phi_deg)
    index = (phi_deg > (phi_mean - d_phi)) * \
            (phi_deg < (phi_mean + d_phi)) * \
            (numpy.fabs(z_kpc) > z_min)    * \
            (numpy.fabs(z_kpc) < (z_min + d_z))
    R_kpc_tracers = R_kpc[index]
    

    #...fitting the tracer density...

    #boundary conditions:
    bounds = [(1.2,20.)]
    #initial guess:
    x0 = [0.5*(1.6+4.85)] #[kpc]
    #   this comes from fig. 6 in Bovy & Rix 2013, first panel
    #   and p.10, second column for the fitting limits

    #minimize:
    res = scipy.optimize.minimize(mloglike_dens, x0, args=(R_kpc_tracers,Rmin_kpc,Rmax_kpc), bounds=bounds)
    h_R   = res.x
    if not res.success:
        print "Problem in tracer scale length: ",res.message


    """#_____transforming the measured tracer scale length to qdf scale length_____
    #(following Bovy: approxFitResult() in pixelFitDF.py)

    #read data from fig. 5 in Bovy & Rix 2013, first panel:
    savefile= open('hrhrhr.sav','rb')
    hrhrhr= pickle.load(savefile)   # = hr_data/hr_qdf
    qdfhrs= pickle.load(savefile)   # = hr_qdf [kpc]
    qdfsrs= pickle.load(savefile)   # = sr_qdf [km/s]
    savefile.close()

    #pick the right curve in first panel of fig. 5:
    srs_qdf          = qdfsrs[0,:]               #pick one row/column, they look the same anyway
    indx             = numpy.argmin((sig_R-srs_qdf)**2.) #we select the closest sig_R curve, i.e. color
    hrs_qdf          = qdfhrs[:,indx]            #this is the x-axis [kpc]
    hrs_data_hrs_qdf = hrhrhr[:,indx]            #this is the y-axis
    hrs_data         = hrs_data_hrs_qdf * hrs_qdf   #this is now the y_axis in terms of hr_data [kpc]
    
    #interpolate along this curve:
    hrSpline= scipy.interpolate.UnivariateSpline(
                        numpy.log(hrs_data),
                        numpy.log(hrs_qdf),
                        k=3)
    lnhrin    = hrSpline(numpy.log(h_R))
    h_R_qdf = numpy.exp(lnhrin)[0]"""
    h_R_qdf = h_R

    # return the set of best fit qdf parameters for the fiducial qdf, 
    # that is used to set the integration ranges of the density over the velocity:
    return (h_R_qdf,sig_R,sig_z,h_sig_R,h_sig_z)

    
    
    

    
