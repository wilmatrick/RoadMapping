#_____import packages_____
from galpy import potential
import numpy
import sys

def calculate_logprior(priortype,pottype,potPar_phys,pot_physical,pot=None):

    """
        NAME:
            calculate_logprior
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2016-12-27 - Written - Trick (MPIA)
            2017-01-05 - Corrected missing minus in d ln v_circ / d ln R. - Trick (MPIA)
            2017-01-08 - Added flag pot_physical. - Trick (MPIA)
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
            #_____df parameters_____
            #logarithmically flat priors
            #(because the fit parameters are actually log(df parameters))
            logprior = 0.
        elif priortype == 1:
            logprior = 0.

            #_____potenial parameters_____ 
            #prior on d (ln v_circ(R0)) / d (ln R) analogous to Bovy & Rix (2013), equation (41)
            if pottype in [8,81,82,821]:
                if pot is None:
                    sys.exit("Error in logprior(): pot keyword needs to "+\
                             "be set with potential object for priortype = "+str(priortype)+\
                             " and potential type = "+str(pottype)+\
                             ".")
                elif ro != 1.:
                    sys.exit("Error in logprior(): In priortype = "+str(priortype)+\
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
                sys.exit("Error in logprior(): priortype = "+str(priortype)+\
                         " is not defined for potential type = "+str(pottype)+\
                         ".")

            #print "potPar_phys = ",potPar_phys
            #print "dlnvc_dlnR = ",dlnvc_dlnR 
            #print "logprior = ", logprior
            #print "\n"

        else:
            sys.exit("Error in logprior(): priortype = "+str(priortype)+" is not defined (yet).")

    return logprior





