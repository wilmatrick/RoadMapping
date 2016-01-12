#_____import packages____
import pickle
import numpy
import sys
import os
import math
import scipy
from write_RoadMapping_parameters import write_RoadMapping_parameters
from read_RoadMapping_parameters import read_RoadMapping_parameters

def gauss(x, A, mu, sigma):
    return A * numpy.exp(-(x-mu)**2/(2.*sigma**2))

#==================================================

def adapt_fitting_range(datasetname,testname=None,analysis_output_filename=None,n_sigma_range=3.,n_gridpoints_final=9,mockdatapath='../data/',force_fine_grid=False,method='GRID'):

    """
        NAME:
           adapt_fitting_range
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-12-01 - Started adapt_fitting_range.py on the basis of BovyCode/py/adapt_fitting_range_flexible.py - Trick (MPIA)
           2015-12-10 - Included special treatment for case, where parameter is pegged at a limit. - Trick (MPIA)
    """

    #_____reference scales_____
    _REFR0 = 8.  #[kpc]
    _REFV0 = 220.   #[km/s]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = "../out/"+dataname+"_analysis_output_"+method+".sav"
        else:                analysis_output_filename = "../out/"+dataname+"_"+testname+"_analysis_output_"+method+".sav"
    if os.path.exists(analysis_output_filename):
        savefile= open(analysis_output_filename,'rb')
        if method == 'GRID':   
            loglike_out      = pickle.load(savefile)  #likelihood in correct grid shape
            ii               = pickle.load(savefile)  #next potential to calculate
            fitParNamesLatex = pickle.load(savefile)  #names of axes in Latex
            gridPointNo      = pickle.load(savefile)  #number of points along each axis
            gridAxesPoints   = pickle.load(savefile)  #all axes in one flattened array
            gridAxesIndex    = pickle.load(savefile)  #indices of start and end of each axis in above array               
            potParFitBool    = pickle.load(savefile)  #boolean array that indicates which of all potential parameters are fitted
            dfParFitBool     = pickle.load(savefile)                    #boolean array that indicates which of all DF parameters are fitted
        elif method == 'MCMC':
            chain_out      = pickle.load(savefile)          #MCMC chain (nwalker,nsteps,ndim)
            fitParNamesLatex   = pickle.load(savefile)      #names of axes in Latex
            gridPointNo        = pickle.load(savefile)      #number of points along each axis
            gridAxesPoints     = pickle.load(savefile)      #all axes in one flattened array
            gridAxesIndex      = pickle.load(savefile)      #indices of start and end of each axis in above array
            potParFitBool      = pickle.load(savefile)      #boolean array that indicates which of all potential parameters are fitted
            dfParFitBool       = pickle.load(savefile)      #boolean array that indicates which of all DF parameters are fitted
            potParTrue_phys    = pickle.load(savefile)      #true potential parameters in physical units
            dfParTrue_fit      = pickle.load(savefile)      #true df parameters in logarithmic fit units
        else:
            sys.exit("Error in adapt_fitting_range: method = "+method+" not known. Use GRID or MCMC.")
        savefile.close()
    else:
        sys.exit("Analysis file "+analysis_output_filename+" does not exist.")

    if method == 'GRID':
        #_____rescale as probabilities are too small to plot_____
        loglike_out -= loglike_out.max()

        nquant = len(gridPointNo)
    elif method == 'MCMC':
        #_____burnin + bounds_____
        ANALYSIS = read_RoadMapping_parameters(
                datasetname,testname=testname,
                mockdatapath=mockdatapath
                )
        burnin_steps = ANALYSIS['noMCMCburnin']
        lower_bounds = numpy.append(ANALYSIS['potParLowerBound_phys'][potParFitBool],
                                    ANALYSIS['dfParLowerBound_fit'  ][dfParFitBool ])
        upper_bounds = numpy.append(ANALYSIS['potParUpperBound_phys'][potParFitBool],
                                    ANALYSIS['dfParUpperBound_fit'  ][dfParFitBool ])
        nquant    = numpy.shape(chain_out)[2]
        chain = chain_out[:, burnin_steps:, :].reshape((-1, nquant))

    #_____quantities_____
    axislist = numpy.array(range(nquant),dtype=int)



    ##################################################

    #_____offset and stddev_____
    qmin = numpy.zeros(nquant)
    qmax = numpy.zeros(nquant)

    #____flat to check if already a fine grid is required______
    fine_grid = 1

    print "===== adapt fitting range ====="

    #_____iterate over all quantities_____
    for ii in range(nquant):

        if method == 'GRID':

            #_____x coordinate_____
            xp = gridAxesPoints[gridAxesIndex[ii]:gridAxesIndex[ii+1]]  #grid axes points

            #____current pixel width_____
            dx = numpy.fabs(xp[1] - xp[0])
            Delta_x = numpy.fabs(xp[-1] - xp[0])

            #_____marginalized probability_____
            axistuple = tuple(axislist[axislist != ii]) #axes over which we marginalize
            logprob = numpy.log(numpy.sum(numpy.exp(loglike_out),axis=axistuple))
            prob = numpy.exp(logprob)

            #_____mean and stddev from marginalized pdf_____
            x_mean = numpy.sum(xp * prob) / numpy.sum(prob)
            x_stddev = numpy.sum(xp * xp * prob) / numpy.sum(prob) - x_mean**2

            #_____best value of the marginalized pdf_____
            indices = numpy.where(logprob == logprob.max())
            i_x = indices[0]

            #_____find new fit range_____
            prob_sort = numpy.sort(prob)
            test = numpy.exp(-0.5 * (n_sigma_range)**2)
            frac = prob_sort[-2] / prob_sort[-1]


            #_____find new fit range_____
            if (frac < test) and not force_fine_grid: #the second highest bin is smaller than a gaussian at N*sigma.
                fine_grid *= 0
                print "Zoom in\t",
                #zoom into highest bin, slightly shift it to mean:
                xmin = x_mean - 0.5 * numpy.fabs(dx)
                xmax = x_mean + 0.5 * numpy.fabs(dx)
                #test if peak is at border, if yes, add a bin:
                if i_x == 0: xmin -= 0.5 * numpy.fabs(dx)
                elif i_x == (len(xp)-1): xmax += 0.5 * numpy.fabs(dx)

            elif (frac >= test) or force_fine_grid:  
                #already reached approximately right resolution
                if len(xp) % 2 == 0:
                    sys.exit("Error in adapt_fitting_range(): Better choose an odd number of grid points.")
                else:
                    mid = (len(xp)-1)/2
                    biggest = numpy.sum(prob[mid] < prob)
                    if (numpy.sum(biggest) > 0) and not force_fine_grid:  
                        #peak is not in the middle: shift
                        print "shift range\t",
                        mu = x_mean
                        sigma = 0.5 * Delta_x / (n_sigma_range)
                        fine_grid *= 0
                    else:   
                        #peak is in the middle: fit Gauss
                        if len(xp) == 3:
                            #special case: pegged at limit:
                            if prob[2] < 2e-308:
                                print "Pegged at upper limit\t",
                                mu = xp[0] + 0.45 * Delta_x
                                sigma = 0.45 * Delta_x / n_sigma_range
                                fine_grid *= 0
                            elif prob[0] < 2e-308:
                                print "Pegged at lower limit\t",
                                mu = xp[2] - 0.45 * Delta_x
                                sigma = 0.45 * Delta_x / n_sigma_range
                                fine_grid *= 0
                            else:
                                print "Gauss through 3 points\t",
                                print "(",prob,")\t",
                                term1 = numpy.log(prob[2]/prob[1]) * (xp[1]**2-xp[0]**2) - numpy.log(prob[0]/prob[1]) * (xp[1]**2-xp[2]**2)
                                term2 = numpy.log(prob[2]/prob[1]) * (xp[1]   -xp[0]   ) - numpy.log(prob[0]/prob[1]) * (xp[1]   -xp[2]   )
                                mu = 0.5*term1/term2
                                sigma = numpy.sqrt((xp[1]**2-xp[0]**2-2.*mu*(xp[1]-xp[0]))/(2.*numpy.log(prob[0]/prob[1])))
                                fine_grid *= 1
                        elif len(xp) > 3:
                            try:
                                print "fit Gauss\t",
                                popt, pcov = scipy.optimize.curve_fit(gauss, xp, prob,p0=[1.,x_mean,x_stddev])
                                A = popt[0]
                                mu = popt[1]
                                sigma = numpy.fabs(popt[2])
                                fine_grid *= 1
                            except RuntimeError:
                                print "estimate Gauss\t",
                                n = 0.5 * ( numpy.sqrt(-2.*numpy.log(prob[0]/prob[mid])) + numpy.sqrt(-2.*numpy.log(prob[-1]/prob[mid])))
                                mu    = x_mean
                                sigma = Delta_x / n
                                fine_grid *= 0
                    xmin = mu - n_sigma_range * sigma
                    xmax = mu + n_sigma_range * sigma

            else:
                #something went wrong, try again with slightly smaller fit range
                print "try again\t",
                fine_grid *= 0
                xmin = x_mean - 0.45*Delta_x
                xmax = x_mean + 0.45*Delta_x

            if numpy.isnan(xmin) or numpy.isnan(xmax):
                #something went terribly wrong, try again with slightly larger fit range:
                print "and try again\t",
                fine_grid *= 0
                mid = (len(xp)-1)/2
                xmin = xp[mid] - 0.52*Delta_x
                xmax = xp[mid] + 0.52*Delta_x

        elif method == 'MCMC':

            #_____MCMC sampels_____
            xs = chain[:,ii] 

            #_____mean and stddev from marginalized pdf_____
            x_mean = numpy.mean(xs)
            x_stddev = numpy.std(xs)

            #optimal bin width (1D)
            bar_width = 3.5 * x_stddev / len(xs)**(1./3.)  
            Delta = n_sigma_range*x_stddev
            #number of bins (1D)
            kx = int(math.ceil(Delta / (0.5 * bar_width)))   
            if ((kx % 2) == 0): kx += 1
            #histogram range (1D and plot)
            xmax = x_mean + kx * 0.5 * bar_width
            xmin = x_mean - kx * 0.5 * bar_width
            x_range = [xmin,xmax]

            #_____histogram_____
            N, b = numpy.histogram(xs, bins=kx, range=x_range,normed=True)
            #coordinates:
            dx = 0.5 * numpy.fabs(b[1] - b[0])
            xp = b[0:-1] + dx

            #_____find new fit range_____
            print "fit Gauss\t",
            popt, pcov = scipy.optimize.curve_fit(gauss, xp, N, p0=[max(N),x_mean,x_stddev])
            A = popt[0]
            mu = popt[1]
            sigma = numpy.fabs(popt[2])
            fine_grid *= 1
            xmin = mu - n_sigma_range * sigma
            xmax = mu + n_sigma_range * sigma

        print "     --> ",fitParNamesLatex[ii]

        qmin[ii] = xmin
        qmax[ii] = xmax


    ##################################################

    #_____which are potential, which df parameters_____
    npotpar = numpy.sum(potParFitBool)
    potParMin_phys = qmin[0:npotpar]
    potParMax_phys = qmax[0:npotpar]
    dfParMin_fit   = qmin[npotpar::]
    dfParMax_fit   = qmax[npotpar::]

    #_____read and apply physical boundaries in potential parameters_____
    out = read_RoadMapping_parameters(datasetname,testname=testname,mockdatapath=mockdatapath)
    potParLowerBound_phys = out['potParLowerBound_phys'][potParFitBool]
    potParUpperBound_phys = out['potParUpperBound_phys'][potParFitBool]
    for ii in range(len(potParMin_phys)):
        if (potParLowerBound_phys[ii] is not None) and \
           (potParMin_phys[ii] < potParLowerBound_phys[ii]):
            potParMin_phys[ii] = potParLowerBound_phys[ii]
        if (potParUpperBound_phys[ii] is not None) and \
           (potParMax_phys[ii] > potParUpperBound_phys[ii]):
            potParMax_phys[ii] = potParUpperBound_phys[ii]


    ##################################################

    if force_fine_grid: fine_grid = 1

    if fine_grid == 0:
        write_RoadMapping_parameters(datasetname,testname=testname,
                            potParMin_phys=potParMin_phys,
                            potParMax_phys=potParMax_phys,
                            dfParMin_fit  =dfParMin_fit,
                            dfParMax_fit  =dfParMax_fit,
                            potParFitNo=3+numpy.zeros(npotpar       ,dtype=int),
                            dfParFitNo =3+numpy.zeros(nquant-npotpar,dtype=int),
                            default_dfParFid=True,
                            mockdatapath=mockdatapath,
                            update=True
                            )
    elif fine_grid == 1:
        write_RoadMapping_parameters(datasetname,testname=testname,
                            potParMin_phys=potParMin_phys,
                            potParMax_phys=potParMax_phys,
                            dfParMin_fit  =dfParMin_fit,
                            dfParMax_fit  =dfParMax_fit,
                            potParFitNo=n_gridpoints_final+numpy.zeros(npotpar       ,dtype=int),
                            dfParFitNo =n_gridpoints_final+numpy.zeros(nquant-npotpar,dtype=int),
                            default_dfParFid=True,
                            mockdatapath=mockdatapath,
                            update=True
                            )

    return fine_grid

