#_____import packages_____
import math
import numpy
import sys

def calculate_outlier_model(dftype,dfPar_galpy,
                            ro,vo,
                            data_galpy=None,sf=None,norm_out=None):   #parameters used for drtype=12

    """
        NAME:
        PURPOSE:
        INPUT:
            dftype - int scalar - defines the DF (and outlier) model to use
            dfPar_galpy - float array - contains the qDF parameters, and the parameters used for the outlier model
        OUTPUT:
        HISTORY:
            2017-01-02 - Written - Trick (MPIA)
    """

    if dftype == 12:

        #____outlier mixture model_____
        #analogous to Bovy & Rix (2013), mimicking halo population:
        #velocity distribution: 
        #   broad 3D Gaussian around (vR=0,vz=0,vT=mean(vT)),
        #   velocity dispersion decreases exponentially with R.
        #spatial distribution: 
        #   constant spatial density.
        #--> om(x,v) = N(vR|0,sig) * N(vz|0,sig) * N(vT|mean(vT),sig)
        #
        #requirement for normalisation of the likelihood: 
        #   int L_out(x,v) dx dv = 1
        #   --> L_out(x,v) = sf(x) * om(x,v) / int sf(x) * om(x,v) dx dv
        #   for one data point.
        #   with 
        #        norm_out = int sf(x) * om(x,v) dx dv 
        #                 = int sf(x) dv
        #  for the above om(x,v).

        #*Note:* In the physical distribution function model likelihood
        #        we ignore the model-independent prefactor Sum_i sf(x_i) 
        #        in the likelihood 
        #        L_DF({x,v}|p) = Sum_i sf(x_i) * df(x_i,v_i) / norm_df 
        #        with norm_df = int sf(x) * df(x,v) dx dv 
        #        and only evaluate L_DF(x_i,v_i|p) = df(x_i,v_i) / norm_df
        #        for each star.
        #        For the DF and the outlier model to be on the same absolute scale
        #        we do not include this prefactor in the outlier, either
        #        and only evaluate L_out(x_i,v_i) = om(x,v) / norm_out

        if data_galpy is None or sf is None or norm_out is None:
            sys.exit("Error in calculate_outlier_model(), dftype=12 "+\
                     "Keywords data_galpy, sf and norm_out need to be set.")

        #data (not in galpy units):
        R_galpy  = data_galpy[0,:]
        vR_galpy = data_galpy[1,:]
        vT_galpy = data_galpy[2,:]
        vz_galpy = data_galpy[3,:]
        #model parameters:
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out]
        sv_out  = dfPar_galpy[6]
        hsv_out = dfPar_galpy[7]

        #velocity dispersion in (vR,vT,vz):
        sig_out = sv_out * numpy.exp(-(R_galpy-ro)/hsv_out)
        
        #central value of Gaussian, here: 0
        peak_vR, peak_vz, peak_vT = 0., 0., 0. 

        #probability of each of the velocities to be drawn from 1D Gaussian:
        ln_om_vR_i = -numpy.log(numpy.sqrt(2.*math.pi)*sig_out) - (vR_galpy-peak_vR)**2 / (2.*sig_out**2)
        ln_om_vz_i = -numpy.log(numpy.sqrt(2.*math.pi)*sig_out) - (vz_galpy-peak_vz)**2 / (2.*sig_out**2)
        ln_om_vT_i = -numpy.log(numpy.sqrt(2.*math.pi)*sig_out) - (vT_galpy-peak_vT)**2 / (2.*sig_out**2)


        #product (or sum in log space), i.e. probability of data to be 
        #drawn from 3D velocity Gaussian:
        lnL_out_i = ln_om_vR_i + ln_om_vz_i + ln_om_vT_i

        #normalisation according to selection function, 
        #norm_out = int sf(x) dx
        #to get outlier ikelihood on same scale as physical DF model likelihood:
        norm_out_sftot = norm_out # = sf.sftot(xgl=_XGL,wgl=_WGL)

        #properly normalize outlier likelihood:
        lnL_out_i -= numpy.log(norm_out_sftot)

        #_____units of the likelihood_____
        # [L_i dx^3 dv^3] = 1 --> [L_i] = [xv]^{-3}
        logunits = 3. * numpy.log(ro*vo)
        lnL_out_i -= logunits

        return lnL_out_i

    else:
        sys.exit("Error in calculate_outlier_model(): dftype = "+str(dftype)+" is not defined yet.")

#----------------

def scale_df_fit_to_galpy(dftype,ro,vo,dfPar_fit):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-01-02 - Written - Trick (MPIA)
    """

    #_____accounting for correct array shape of input_____
    npar = [5,8]
    transpose = False
    if dfPar_fit.ndim == 1 and dfPar_fit.shape[0] in npar:
        dfPar_fit = dfPar_fit.reshape((1,dfPar_fit.shape[0]))
    if dfPar_fit.ndim == 2 and dfPar_fit.shape[1] in npar:
        pass    #all fine
    elif dfPar_fit.ndim == 2 and dfPar_fit.shape[0] in npar:
        dfPar_fit = dfPar_fit.T
        transpose = True
    else:
        sys.exit("Error in scale_df_fit_to_galpy(): Input array does not have the correct shape.")

    #_____transforming the df parameters_____
    if dftype in [0,11]:
        #dfPar = [hr,sr,sz,hsr,hsz] --> 5 parameters

        traf = numpy.array([ro,vo,vo,ro,ro])
        dfPar_galpy = numpy.exp(numpy.array(dfPar_fit,dtype=float)) / traf

    elif dftype in [12]:
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out] --> 8 parameters
        dfPar_galpy = numpy.zeros_like(dfPar_fit)

        #...[hr,sr,sz,hsr,hsz]
        traf_qdf = numpy.array([ro,vo,vo,ro,ro])
        dfPar_galpy[:,0:5] = numpy.exp(numpy.array(dfPar_fit[:,0:5],dtype=float)) / traf_qdf
        #...[p_out]
        dfPar_galpy[:,5] = dfPar_fit[:,5]
        #...[sv_out,hv_out]
        traf_out = numpy.array([vo,ro])
        dfPar_galpy[:,6:8] = numpy.exp(numpy.array(dfPar_fit[:,6:8],dtype=float)) / traf_out
    else:
        sys.exit("Error in scale_df_fit_to_galpy(): dftype = "+\
                 str(dftype)+" is not defined.")

    if transpose: dfPar_galpy = dfPar_galpy.T
    return numpy.squeeze(dfPar_galpy)

#----------------

def scale_df_galpy_to_phys(dftype,ro,vo,dfPar_galpy):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-01-03 - Written - Trick (MPIA)
    """

    #_____accounting for correct array shape of input_____
    npar = [5,8]
    transpose = False
    if dfPar_galpy.ndim == 1 and dfPar_galpy.shape[0] in npar:
        dfPar_galpy = dfPar_galpy.reshape((1,dfPar_galpy.shape[0]))
    if dfPar_galpy.ndim == 2 and dfPar_galpy.shape[1] in npar:
        pass    #all fine
    elif dfPar_galpy.ndim == 2 and dfPar_galpy.shape[0] in npar:
        dfPar_galpy = dfPar_galpy.T
        transpose = True
    else:
        sys.exit("Error in scale_df_galpy_to_phys(): Input array does not have the correct shape.")

    #_____global constants_____
    _REFR0 = 8.     #spatial scaling
    _REFV0 = 220.   #velocity scaling


    #_____transforming the df parameters_____
    if dftype in [0,11]:
        #dfPar = [hr,sr,sz,hsr,hsz] --> 5 parameters

        traf = numpy.array([ro*_REFR0,vo*_REFV0,vo*_REFV0,ro*_REFR0,ro*_REFR0])
        dfPar_phys = numpy.array(dfPar_galpy,dtype=float) * traf

    elif dftype in [12]:
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out] --> 8 parameters
        dfPar_phys = numpy.zeros_like(dfPar_galpy)

        #...[hr,sr,sz,hsr,hsz]
        traf_qdf = numpy.array([ro*_REFR0,vo*_REFV0,vo*_REFV0,ro*_REFR0,ro*_REFR0])
        dfPar_phys[:,0:5] = numpy.array(dfPar_galpy[:,0:5],dtype=float) * traf_qdf
        #...[p_out]
        dfPar_phys[:,5] = dfPar_galpy[:,5]
        #...[sv_out,hv_out]
        traf_out = numpy.array([vo*_REFV0,ro*_REFR0])
        dfPar_phys[:,6:8] = numpy.array(dfPar_galpy[:,6:8],dtype=float) * traf_out
    else:
        sys.exit("Error in scale_df_galpy_to_phys(): dftype = "+\
                 str(dftype)+" is not defined.")

    if transpose: dfPar_phys = dfPar_phys.T
    return numpy.squeeze(dfPar_phys)

#----------------

def scale_df_phys_to_fit(dftype,dfPar_phys):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-01-03 - Written - Trick (MPIA)
    """

    #_____accounting for correct array shape of input_____
    npar = [5,8]
    transpose = False
    if dfPar_phys.ndim == 1 and dfPar_phys.shape[0] in npar:
        dfPar_phys = dfPar_phys.reshape((1,dfPar_phys.shape[0]))
    if dfPar_phys.ndim == 2 and dfPar_phys.shape[1] in npar:
        pass    #all fine
    elif dfPar_phys.ndim == 2 and dfPar_phys.shape[0] in npar:
        dfPar_phys = dfPar_phys.T
        transpose = True
    else:
        sys.exit("Error in scale_df_phys_to_fit(): Input array does not have the correct shape.")

    #_____global constants_____
    _REFR0 = 8.     #spatial scaling
    _REFV0 = 220.   #velocity scaling

    #_____transforming the df parameters_____
    if dftype in [0,11]:
        #dfPar = [hr,sr,sz,hsr,hsz] --> 5 parameters

        traf = numpy.array([_REFR0,_REFV0,_REFV0,_REFR0,_REFR0])
        dfPar_fit = numpy.log(numpy.array(dfPar_phys,dtype=float) / traf)

    elif dftype in [12]:
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out] --> 8 parameters
        dfPar_fit = numpy.zeros_like(dfPar_phys)

        #...[hr,sr,sz,hsr,hsz]
        traf_qdf = numpy.array([_REFR0,_REFV0,_REFV0,_REFR0,_REFR0])
        dfPar_fit[:,0:5] = numpy.log(numpy.array(dfPar_phys[:,0:5],dtype=float) / traf_qdf)
        #...[p_out]
        dfPar_fit[:,5] = dfPar_phys[:,5]
        #...[sv_out,hv_out]
        traf_out = numpy.array([_REFV0,_REFR0])
        dfPar_fit[:,6:8] = numpy.log(numpy.array(dfPar_phys[:,6:8],dtype=float) / traf_out)
    else:
        sys.exit("Error in scale_df_phys_to_fit(): dftype = "+\
                 str(dftype)+" is not defined.")

    if transpose: dfPar_fit = dfPar_fit.T
    return numpy.squeeze(dfPar_fit)

#----------------

def scale_df_fit_to_phys(dftype,dfPar_fit):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-01-03 - Written - Trick (MPIA)
    """

    #_____accounting for correct array shape of input_____
    npar = [5,8]
    transpose = False
    if dfPar_fit.ndim == 1 and dfPar_fit.shape[0] in npar:
        dfPar_fit = dfPar_fit.reshape((1,dfPar_fit.shape[0]))
    if dfPar_fit.ndim == 2 and dfPar_fit.shape[1] in npar:
        pass    #all fine
    elif dfPar_fit.ndim == 2 and dfPar_fit.shape[0] in npar:
        dfPar_fit = dfPar_fit.T
        transpose = True
    else:
        sys.exit("Error in scale_df_fit_to_phys(): Input array does not have the correct shape.")

    #_____global constants_____
    _REFR0 = 8.     #spatial scaling
    _REFV0 = 220.   #velocity scaling

    #_____transforming the df parameters_____
    if dftype in [0,11]:
        #dfPar = [hr,sr,sz,hsr,hsz] --> 5 parameters

        traf = numpy.array([_REFR0,_REFV0,_REFV0,_REFR0,_REFR0])
        dfPar_phys = numpy.exp(numpy.array(dfPar_fit,dtype=float)) * traf

    elif dftype in [12]:
        #dfPar = [hr,sr,sz,hsr,hsz,p_out,sv_out,hv_out] --> 8 parameters
        dfPar_phys = numpy.zeros_like(dfPar_fit)

        #...[hr,sr,sz,hsr,hsz]
        traf_qdf = numpy.array([_REFR0,_REFV0,_REFV0,_REFR0,_REFR0])
        dfPar_phys[:,0:5] = numpy.exp(numpy.array(dfPar_fit[:,0:5],dtype=float)) * traf_qdf
        #...[p_out]
        dfPar_phys[:,5] = dfPar_fit[:,5]
        #...[sv_out,hv_out]
        traf_out = numpy.array([_REFV0,_REFR0])
        dfPar_phys[:,6:8] = numpy.exp(numpy.array(dfPar_fit[:,6:8],dtype=float)) * traf_out
    else:
        sys.exit("scale_df_fit_to_phys(): dftype = "+\
                 str(dftype)+" is not defined.")

    if transpose: dfPar_phys = dfPar_phys.T
    return numpy.squeeze(dfPar_phys)



