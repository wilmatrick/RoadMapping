#_____import packages_____
import math
import scipy.optimize
import scipy.interpolate
import numpy
import sys
import pickle
import matplotlib.pylab as plt
from galpy.potential import MiyamotoNagaiPotential

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
    
    sys.exit("Error in mloglike_dens(): I'm not sure anymore, if the "+\
             "Jacobian is needed in the likelihood or not. Check code!")
    # normalized log of tracer distribution / likelihood:
    # p(R) = R * exp(-R/h_R) / Mtot
    # Mtot = h_R * (h_R + Rmin) * exp(-Rmin/h_R) - h_R * (h_R + Rmax) * exp(-Rmax/h_R)
    Mtot   = h_R * (h_R + Rmin) * numpy.exp(-Rmin/h_R) - h_R * (h_R + Rmax) * numpy.exp(-Rmax/h_R)
    log_p = -Rs / h_R + numpy.log(Rs) - numpy.log(Mtot)

    #minus likelihood (for minimizing the function):
    return -numpy.sum(log_p)

#_____exponential density function to fit_____
#(properly normalized in selection function)
#(double exponential disk)
def mloglike_dens_expdisk(x,*args):
    
    #fit parameters (scalar input only!):
    h_R = x[0]
    h_z = x[1]
    
    #data:
    Rs  = args[0]
    zs  = args[1]

    #fixed parameters:
    sf     = args[2]
    sftype = args[3]
    xgl    = args[4]
    wgl    = args[5]
    
    # normalized log of tracer distribution / likelihood:
    # p(R,z) = exp(-R/h_R - |z|/h_z) / Mtot

    #define function func(R,z) to integrate:
    rho = lambda rr,zz: numpy.exp(-rr/h_R - numpy.abs(zz)/h_z)

    if sftype == 4: #Incomplete shell
        
        #total mass in selection function:
        Mtot = sf._fastGLint_IncompleteShell(rho,xgl,wgl)

    else:
        sys.exit("Error in mloglike_dens_expdisk(): sftype = "+str(sftype)+" is not defined yet.")

    log_p = -Rs/h_R - numpy.abs(zs)/h_z - numpy.log(Mtot)

    #minus likelihood (for minimizing the function):
    return -numpy.sum(log_p)

#_____exponential density function to fit_____
#(properly normalized in selection function)
#(Miyamoto-Nagai disk)
def mloglike_dens_MNdisk(x,*args):

    #fit parameters (scalar input only!):
    a = x[0]
    b = x[1]
    
    #data:
    Rs  = args[0]
    zs  = args[1]

    #fixed parameters:
    sf     = args[2]
    sftype = args[3]
    xgl    = args[4]
    wgl    = args[5]

    #define function func(R,z) to integrate --> rho_MiyamotoNagai(R,z), eq. (2.69b) in B&T2008
    #rho = lambda R,z: b**2 * (a*R**2+(a+3.*numpy.sqrt(z**2+b**2))*(a+numpy.sqrt(z**2+b**2))**2) / \
    #                         ((R**2+(a+numpy.sqrt(z**2+b**2))**2)**(5./2.)*(z**2+b**2)**(3./2.))

    pot = MiyamotoNagaiPotential(a=a,b=b,normalize=1.)
    rho = lambda rr,zz: pot._dens(rr,zz)
    
    # normalized log of tracer distribution / likelihood:
    if sftype == 4: #Incomplete shell

       #total mass in selection function:
        Mtot = sf._fastGLint_IncompleteShell(rho,xgl,wgl)

    else:
        sys.exit("Error in mloglike_dens_MNdisk(): sftype = "+str(sftype)+" is not defined yet.")

    log_p = numpy.log(rho(Rs,zs)) - numpy.log(Mtot)

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

#---------------------------------



def estimate_initial_df_parameters(R_kpc,vR_kms,phi_deg,vT_kms,z_kpc,vz_kms,
                                sftype=None,sf=None,Rsun_kpc=None,
                                plot=False,plotfilename=None,
                                fit_with_velocity_scale_lengths=True):
    """
        NAME:
           estimate_fiducial_qdf
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2017-01-05 - Written. - Trick (MPIA)
    """

    #_____global constants_____  
    _REFR0 = 8.        #spatial scaling
    _REFV0 = 220.      #velocity scaling

    if plot: fig = plt.figure(figsize=(6,5))

    #_____fitting velocity disersion profiles_____
    print "    ... fitting velocity dispersions"

    #...radial velocity dispersion...

    if fit_with_velocity_scale_lengths:
    
        #initial guess:
        x0 = [0.5*(30.+60.)/_REFV0,8./_REFR0] #[km/s/_REFV0,kpc/_REFR0]
        #boundary conditions:
        bounds = ((10./_REFV0,70./_REFV0),(1./_REFR0,20./_REFR0))
        #   this comes from fig. 6 in Bovy & Rix 2013, second panel
        #   and p.10, second column, hsr = 8 kpc.
        
        #minimize:
        res = scipy.optimize.minimize(mloglike_vel_disp, x0, args=(R_kpc/_REFR0,vR_kms/_REFV0,Rsun_kpc/_REFR0), bounds=bounds)
        sig_R   = res.x[0] * _REFV0
        h_sig_R = res.x[1] * _REFR0
        if not res.success:
            print "Problem in radial velocity dispersion: ",res.message

        if plot:
            ax = fig.add_subplot(121)
            vR_mean, R_kpc_edges, binnumber = scipy.stats.binned_statistic(R_kpc,vR_kms,statistic=(lambda x: numpy.std(x)),bins=10)
            R_kpc_bins = R_kpc_edges[0:-1] + 0.5 * (R_kpc_edges[1] - R_kpc_edges[0])
            ax.plot(R_kpc_bins,vR_mean,color='cornflowerblue',linestyle='None',marker='o')
            ax.plot(R_kpc_bins,sig_R*numpy.exp(-(R_kpc_bins-Rsun_kpc)/h_sig_R),color='crimson',linewidth=2,marker=None)
            ax.set_xlabel('R [kpc]')
            ax.set_ylabel('$\sigma_R$ [km/s]')

    else:

        sig_R = numpy.std(vR_kms)

        if plot:
            ax = fig.add_subplot(121)
            hist, vR_kms_edges = numpy.histogram(vR_kms,normed=True,bins=10)
            vR_kms_edges = vR_kms_edges[0:-1] + 0.5 * (vR_kms_edges[1] - vR_kms_edges[0])
            ax.plot(vR_kms_edges,hist,color='cornflowerblue',linestyle='None',marker='o')
            ax.plot(vR_kms_edges,numpy.exp(-(vR_kms_edges)**2/(2.*sig_R**2))/numpy.sqrt(2.*math.pi*sig_R**2),color='crimson',linewidth=2,marker=None)
            ax.set_xlabel('$v_R$ [km/s]')

    #...vertical velocity dispersion...

    if fit_with_velocity_scale_lengths:

        #initial guess:
        x0 = [0.5*(16.+80.)/_REFV0,5./_REFR0] #[km/s/_REFV0,kpc/_REFR0]
        #boundary conditions:
        bounds = ((1./_REFV0,90./_REFV0),(1./_REFR0,20./_REFR0))
        #   this comes from fig. 6 in Bovy & Rix 2013, third panel
        #   and p.10, second column for the fitting limits of hsr
        
        #minimize:
        res = scipy.optimize.minimize(mloglike_vel_disp, x0, args=(R_kpc/_REFR0,vz_kms/_REFV0,Rsun_kpc/_REFR0), bounds=bounds)
        sig_z   = res.x[0] * _REFV0
        h_sig_z = res.x[1] * _REFR0
        if not res.success:
            print "Problem in vertical velocity dispersion: ",res.message

        if plot:
            ax = fig.add_subplot(122)
            vz_mean, R_kpc_edges, binnumber = scipy.stats.binned_statistic(R_kpc,vz_kms,statistic=(lambda x: numpy.std(x)),bins=10)
            R_kpc_bins = R_kpc_edges[0:-1] + 0.5 * (R_kpc_edges[1] - R_kpc_edges[0])
            ax.plot(R_kpc_bins,vz_mean,color='cornflowerblue',linestyle='None',marker='o')
            ax.plot(R_kpc_bins,sig_z*numpy.exp(-(R_kpc_bins-Rsun_kpc)/h_sig_z),color='crimson',linewidth=2,marker=None)
            ax.set_xlabel('R [kpc]')
            ax.set_ylabel('$\sigma_z$ [km/s]')

    else:
        
        sig_z = numpy.std(vz_kms)

        if plot:
            ax = fig.add_subplot(122)
            hist, vz_kms_edges = numpy.histogram(vz_kms,normed=True,bins=10)
            vz_kms_edges = vz_kms_edges[0:-1] + 0.5 * (vz_kms_edges[1] - vz_kms_edges[0])
            ax.plot(vz_kms_edges,hist,color='cornflowerblue',linestyle='None',marker='o')
            ax.plot(vz_kms_edges,numpy.exp(-(vz_kms_edges)**2/(2.*sig_z**2))/numpy.sqrt(2.*math.pi*sig_z**2),color='crimson',linewidth=2,marker=None)
            ax.set_xlabel('$v_z$ [km/s]')

    #____fitting the tracer scale length and height_____
    print "    ... fitting double exponential disk"

    #boundary conditions:
    bounds = [(1.2/_REFR0,20./_REFR0),(0.001/_REFR0,5./_REFR0)]
    #initial guess:
    x0 = [0.5*(1.6+4.85)/_REFR0,0.3/_REFR0] #[kpc/_REFR0]
    #   this comes from fig. 6 in Bovy & Rix 2013, first panel
    #   and p.10, second column for the fitting limits.
    #   And random initial value for tracer scale height.
    
    #for Gauss Legendre integration over selection function:
    _XGL,_WGL = numpy.polynomial.legendre.leggauss(40)
   
    #minimize:
    res = scipy.optimize.minimize(mloglike_dens_expdisk, x0, args=(R_kpc/_REFR0,z_kpc/_REFR0,sf,sftype,_XGL,_WGL), bounds=bounds)
    h_R_disk   = res.x[0] * _REFR0
    h_z_disk   = res.x[1] * _REFR0
    if not res.success:
        print "Problem in tracer scale length & height: ",res.message

    """

    #____fitting the Miyamoto-Nagai disk_____
    print "    ... fitting Miyamoto-Nagai disk"

    #initial guess:
    x0 = [h_R_disk/_REFR0,h_z_disk/_REFR0] #[kpc/_REFR0]

    #boundary conditions:
    bounds = [(0.5/_REFR0,20./_REFR0),(0.0001/_REFR0,10./_REFR0)]

   
    #minimize:
    res = scipy.optimize.minimize(mloglike_dens_MNdisk, x0, args=(R_kpc/_REFR0,z_kpc/_REFR0,sf,sftype,_XGL,_WGL), bounds=bounds)
    a_disk  = res.x[0] * _REFR0
    b_disk  = res.x[1] * _REFR0
    if not res.success:
        print "Problem in fitting MN disk: ",res.message

    """

    if plot: 
        plt.tight_layout()
        plt.savefig(plotfilename)

    if fit_with_velocity_scale_lengths: return (sig_R,sig_z,h_sig_R,h_sig_z,h_R_disk,h_z_disk)
    else: return (sig_R,sig_z,h_R_disk,h_z_disk)
    
    
    

    
