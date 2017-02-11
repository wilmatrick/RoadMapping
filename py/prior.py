#_____import packages_____
from galpy import potential
import numpy
import sys
from outlier_model import scale_df_galpy_to_phys

def calculate_logprior_potential(priortype,pottype,potPar_phys,pot_physical,pot=None):

    """
        NAME:
            calculate_logprior_potential
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2016-12-27 - Written - Trick (MPIA)
            2017-01-05 - Corrected missing minus in d ln v_circ / d ln R. - Trick (MPIA)
            2017-01-08 - Added flag pot_physical. - Trick (MPIA)
            2017-02-10 - Added priortype 11, which is flat rotation curve + boundaries in hR. - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    #_____scale position of sun and circular velocity to galpy units_____
    ro = potPar_phys[0] / _REFR0
    vo = potPar_phys[1] / _REFV0

    if not pot_physical:
        #_____unphysical potential --> prior = 0_____
        logprior = -numpy.inf

    else:
        if priortype == 0:
            #_____potential parameters_____
            #flat prior
            #zero outside the given grid limits and in unphysical regions 
            #(taken care of in likelhood function with keyword pot_physical)
            logprior = 0.

        elif priortype in [1,11]:
            logprior = 0.

            #_____potential parameters_____ 
            #prior on d (ln v_circ(R0)) / d (ln R) analogous to Bovy & Rix (2013), equation (41)
            if pottype in [8,81,82,821]:
                if pot is None:
                    sys.exit("Error in calculate_logprior_potential(): pot keyword needs to "+\
                             "be set with potential object for priortype = "+str(priortype)+\
                             " and potential type = "+str(pottype)+\
                             ".")
                elif ro != 1.:
                    sys.exit("Error in calculate_logprior_potential(): In priortype = "+str(priortype)+\
                             " and potential type = "+str(pottype)+". Prior on rotation curve "+\
                             "slope is evaluated at R=1 [galpy units]. If this "+\
                             "is not what is required, then the code needs to "+\
                             "be changed.")
                else:
                    #calculate slope of total rotation curve at R=1 [galpy units]
                    # --> d (ln v_circ) / d (ln R) |R=1
                    # 1. v_circ^2                 = R * d Phi / d R 
                    #                          = - R * F_R
                    #
                    # 2. d (ln v_circ) / d (ln R)
                    #    = R / v_circ * dv_circ / dR 
                    #    = R / v_circ * d/dR sqrt(R * dPhi/dR)
                    #    = R / v_circ * 0.5 / v_circ * d/dR (R * dPhi/dR)
                    #    = 0.5 * R / v_circ^2 * (dPhi/dR + R * d^2Phi/dR^2)
                    #    = 0.5 / (-F_R) * (-F_R + R * d^2Phi/dR^2)
                    #    = 0.5 * (F_R - R * d^2Phi/dR^2) / F_R
                    #
                    # 3. Evaluated at R = 1, where F_R = -1, according to galpy unit definition:
                    #    --> d (ln v_circ) / d (ln R) = 0.5 * (1 + d^2 Phi / d R^2|R=1)
                    
                    dlnvc_dlnR = 0.5 * (1. + potential.evaluateR2derivs(pot,1.,0.))  #R=1, z=0

                    #Prior following equation (41) in Bovy & Rix (2013):
                    if dlnvc_dlnR > 0.04: 
                        logprior += -numpy.inf
                    else:
                        W = (1. - 1./0.04 * dlnvc_dlnR)
                        logprior += numpy.log(W) - W
            else: 
                sys.exit("Error in calculate_logprior_potential(): priortype = "+str(priortype)+\
                         " is not defined for potential type = "+str(pottype)+\
                         ".")

            #print "potPar_phys = ",potPar_phys
            #print "dlnvc_dlnR = ",dlnvc_dlnR 
            #print "logprior = ", logprior
            #print "\n"

        else:
            sys.exit("Error in logprior(): priortype = "+str(priortype)+" is not defined (yet).")

    return logprior

#-----------------------------------------------------------------

def calculate_logprior_df(priortype,dftype,dfPar_galpy,ro,vo):

    """
        NAME:
            calculate_logprior_df
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-02-10 - Written - Trick (MPIA)
    """

    #_____global constants_____
    _REFR0 = 8.
    _REFV0 = 220.

    if priortype in [0,1]:

        #_____df parameters_____
        #logarithmically flat priors
        #(because the fit parameters are actually log(df parameters))
        logprior = 0.

    elif priortype == 11:

        #_____df parameters_____
        #logarithmically flat priors
        #but h_r is limited to the range [0.5kpc,20kpc]

        if dftype in [0,11,12]:

            #transform to physical units and pick only hr parameter:
            dfPar_phys = scale_df_galpy_to_phys(dftype,ro,vo,dfPar_galpy)
            hr_kpc     = dfPar_phys[0]

            if (hr_kpc < 0.5) or (hr_kpc > 20.):
                logprior = -numpy.inf
            else:
                logprior = 0.
  
        else:
            sys.exit("Error in calculate_logprior_df(): priortype = "+str(priortype)+\
                     " is not defined for df type = "+str(dftype)+\
                     ".")

    else:
        sys.exit("Error in calculate_logprior_df(): priortype = "+str(priortype)+" is not defined (yet).")

    return logprior   





