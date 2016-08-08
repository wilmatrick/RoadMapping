#_____import packages_____
import pickle
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib
import sys
from scipy.misc import logsumexp
from matplotlib.patches import Polygon
import scipy.optimize
import scipy.stats
sys.path.insert(0,'/home/trick/RoadMapping/py')
from read_RoadMapping_parameters import read_RoadMapping_parameters


#-------------------------------------------------

def gauss(x, A, mu, sigma):
    return A * numpy.exp(-(x-mu)**2/(2.*sigma**2))

def oplot_violines(datasetname,pos_in_plot,
                            testname=None,
                            method='GRID',
                            quantities_to_plot=None,
                            list_of_axis_objects=None,
                            width=0.5,
                            analysis_output_filename=None,
                            color=None,
                            columns=1,
                            this_column=1,
                            sigma_levels=False,
                            plot_violins=True,
                            plot_true_values=False,
                            show_labels=False,
                            burnin_steps=None,
                            mockdatapath=None,
                            fulldatapath=None,
                            Gaussian_fit=False,
                            **kwargs):

    #_____reference scales_____
    _REFR0 = 8.     #[kpc]
    _REFV0 = 220.   #[km/s]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = "../out/"+datasetname+"_analysis_output_"+method+".sav"
        else:                analysis_output_filename = "../out/"+datasetname+"_"+testname+"_analysis_output_"+method+".sav"
    savefile= open(analysis_output_filename,'rb')    
    if method == 'MCMC':
        chain_out      = pickle.load(savefile)                #MCMC chain
    elif method == 'GRID':
        savefile       = open(analysis_output_filename,'rb')
        loglike_out    = pickle.load(savefile)             #likelihood grid
        ii             = pickle.load(savefile)             #iteration index
    fitParNamesLatex   = pickle.load(savefile)      #names of axes in Latex
    gridPointNo        = pickle.load(savefile)      #number of points along each axis
    gridAxesPoints     = pickle.load(savefile)      #all axes in one flattened array
    gridAxesIndex      = pickle.load(savefile)      #indices of start and end of each axis in above array
    potParFitBool      = pickle.load(savefile)      #boolean array that indicates which of all potential parameters are fitted
    dfParFitBool       = pickle.load(savefile)      #boolean array that indicates which of all DF parameters are fitted
    potParTrue_phys    = pickle.load(savefile)      #true potential parameters in physical units
    dfParTrue_fit      = pickle.load(savefile)      #true df parameters in logarithmic fit units
    savefile.close()

    if method == 'MCMC':
        ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            mockdatapath=mockdatapath,
            fulldatapath=fulldatapath
            )
        #____parameter boundaries_____
        lower_bounds = numpy.append(ANALYSIS['potParLowerBound_phys'][potParFitBool],
                                    ANALYSIS['dfParLowerBound_fit'  ][dfParFitBool ])
        upper_bounds = numpy.append(ANALYSIS['potParUpperBound_phys'][potParFitBool],
                                    ANALYSIS['dfParUpperBound_fit'  ][dfParFitBool ])
        #_____remove burn-in & flatten walkers_____
        ndim  = numpy.shape(chain_out)[2]
        if burnin_steps is None:
            burnin_steps = ANALYSIS['noMCMCburnin']
        chain = chain_out[:, burnin_steps:, :].reshape((-1, ndim))
    elif method == 'GRID':
        #_____number of dimensions_____
        ndim = len(loglike_out.shape)
        #_____rescale as probabilities are too small to plot_____
        loglike_out[numpy.isnan(loglike_out)] = -numpy.finfo(numpy.dtype(numpy.float64)).max
        loglike_out -= loglike_out.max()

    #_____true values_____
    trueValues = numpy.append(potParTrue_phys[potParFitBool],dfParTrue_fit[dfParFitBool])

    ##################################################

    #_____quantities to plot_____
    if quantities_to_plot is None: quantities_to_plot = range(ndim)
    n_quant = len(quantities_to_plot)
    axislist = numpy.array(range(ndim),dtype=int)

    #_____parse additional plotting parameters_____
    if kwargs.has_key('edgecolor'):
        edgecolor=kwargs['edgecolor']
        kwargs.pop('edgecolor')
    else:
        edgecolor = 'k'
    if kwargs.has_key('alpha'):
        alpha=kwargs['alpha']
        kwargs.pop('alpha')
    else:
        alpha = 1.
    if sigma_levels:
        if kwargs.has_key('markercolor'):
            markercolor=kwargs['markercolor']
            kwargs.pop('markercolor')
        else:
            markercolor='white'
        if kwargs.has_key('marker'):
            marker=kwargs['marker']
            kwargs.pop('marker')
            print markercolor,marker
        else:
            marker='.'


    #_____iterate over all combinations of quantities_____
    for ii in range(n_quant):

        #_____x coordinate_____
        ix = quantities_to_plot[ii]
        if method == 'GRID':
            xp = gridAxesPoints[gridAxesIndex[ix]:gridAxesIndex[ix+1]]  #grid axes points
        if method == 'MCMC': 
            xs = chain[:,ix] #MCMC sampels
            mean = numpy.mean(xs)
            stddev = numpy.std(xs)
            xmin = mean-3.*stddev
            xmax = mean+3.*stddev
            if numpy.isfinite(lower_bounds[ix]): xmin = numpy.max([lower_bounds[ix],xmin])                           
            if numpy.isfinite(upper_bounds[ix]): xmax = numpy.min([upper_bounds[ix],xmax])                      
            xp = numpy.linspace(xmin,xmax,30)
            
          
        #========== 1D probability distribution==========

        #_____marginalized probability_____
        if method == 'MCMC':
            #kernel density estimation:
            kernel = scipy.stats.gaussian_kde(xs)
            prob   = kernel(xp)
        elif method == 'GRID':
            #marginalize over axes:
            axistuple = tuple(axislist[axislist != ix]) #axes over which we marginalize
            logprob = numpy.log(numpy.sum(numpy.exp(loglike_out),axis=axistuple))
            prob = numpy.exp(logprob)
            #pixel width:
            #dx = numpy.fabs(xp[1] - xp[0])
            #normalize:
            #prob /= numpy.sum(prob) * dx


        #_____best value of the marginalized pdf_____
        if numpy.sum(numpy.isnan(prob)) == prob.size:
            x_best = truevalues[ix]
        else:
            indices = numpy.where(prob == prob.max())
            i_x = indices[0]
            x_best = xp[i_x]

        #_____position of next plot _____
        if list_of_axis_objects is None:
            plotpos =  ii * columns + this_column
            ax1 = plt.subplot(n_quant,columns,plotpos)
        else:
            ax1 = list_of_axis_objects[ii]

        if plot_violins:

            #_____prepare plot arrays for polygon_____
            if method == 'MCMC':    #smooth distribution
                a1 = prob / numpy.sum(prob)
                b1 = xp
            elif method == 'GRID':  #histogram
                dx = 0.5 * (xp[1] - xp[0])
                a1 = [0.]
                b1 = [xp[0]-dx]
                for mm in range(len(xp)):
                    a1 = numpy.append(a1,[prob[mm],prob[mm]])
                    b1 = numpy.append(b1,[xp[mm]-dx,xp[mm]+dx])
                a1 = numpy.append(a1,[0.]) / numpy.sum(prob)
                b1 = numpy.append(b1,[xp[mm]+dx])
            a2 = -a1[::-1]
            b2 = b1[::-1]
            ap = numpy.append(a1,a2) * width + pos_in_plot
            bp = numpy.append(b1,b2)
            d = numpy.zeros((len(ap),2))
            d[:,0] = ap
            d[:,1] = bp

            #_____color_____
            if color is None: color = 'red'

            #_____plot pdf_____
            ax1.add_patch(Polygon(d, closed=True,
                          fill=True, facecolor=color, edgecolor=edgecolor, alpha=alpha,zorder=3))

        if sigma_levels:

            if Gaussian_fit:
                #_____fit gaussian_____
                try:
                    popt, pcov = scipy.optimize.curve_fit(gauss, xp, prob,p0=[1.,x_best,0.3*(max(xp)-min(xp))])
                except RuntimeError as e:
                    pass #do nothing
                else:
                    A = popt[0]
                    mu = popt[1]
                    sigma = numpy.fabs(popt[2])

                #_____sigma confidence levels_____
                ax1.errorbar([pos_in_plot], [mu], yerr=[sigma], color=markercolor,marker=marker,zorder=4,ecolor=markercolor,elinewidth=2,markeredgecolor='None',capsize=0,markersize=10)

            else:
                percentile = numpy.percentile(xs, [15.87,50.,84.13])
                #percentile2 = percentile
                #print ii, percentile2[1],', +',percentile2[2]-percentile2[1],', -',percentile2[1]-percentile2[0]
                
                #_____sigma confidence levels_____
                err_l = percentile[1]-percentile[0]
                err_u = percentile[2]-percentile[1]
                ax1.errorbar([pos_in_plot], [percentile[1]], yerr=[[err_l],[err_u]], color=markercolor,marker=marker,zorder=4,ecolor=markercolor,elinewidth=2,markeredgecolor='None',capsize=0,markersize=10)

        #_____plot true values_____
        if plot_true_values:
            ax1.hlines([trueValues[ix]],[pos_in_plot-2.*width],[pos_in_plot+2.*width],linestyle='dotted',color='0.7',linewidth=2,zorder=4)

        #_____axis labels_____
        if show_labels and this_column == 1:
            ax1.set_ylabel(fitParNamesLatex[ix])


