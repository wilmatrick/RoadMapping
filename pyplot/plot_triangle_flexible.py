#_____import packages_____
import pickle
import numpy
import math
import matplotlib.pyplot as plt
import sys
import scipy.optimize
import scipy.stats
sys.path.insert(0,'/home/trick/RoadMapping/py')
from read_RoadMapping_parameters import read_RoadMapping_parameters

#-------------------------------------------------

def gauss(x, A, mu, sigma):
    return A * numpy.exp(-(x-mu)**2/(2.*sigma**2))

def plot_triangle_flexible(datasetname,plotfilename,testname=None,
                                quantities_to_plot=None,size=15,analysis_output_filename=None,
                                method='MCMC',
                                burnin_steps=None,color=True,
                                fit_gaussian=True,
                                datapath='/home/trick/ElenaSim/out/'):

    """
        NAME:
           plot_triangle_flexible
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-12-07 - Started plot_triangle_flexible.py on the basis of BovyCode/pyplots/plot_triangle_flexible.py - Trick (MPIA)
    """

    #_____reference scales_____
    _REFR0 = 8.     #[kpc]
    _REFV0 = 220.   #[km s$^{-1}$]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = datapath+datasetname+"_analysis_output_"+method+".sav"
        else:                analysis_output_filename = datapath+datasetname+"_"+testname+"_analysis_output_"+method+".sav"
    savefile= open(analysis_output_filename,'rb')    
    if method == 'MCMC':
        chain_out      = pickle.load(savefile)             #MCMC chain
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

    #??????
    #fitParNamesLatex = ['$v_{circ}(R_\odot)$ [km s$^{-1}$]','$b$ [kpc]','ln($h_R$/8kpc)','ln($\sigma_{R,0}$/220km s$^{-1}$)','ln($\sigma_{z,0}$/220km s$^{-1}$)','ln($h_{\sigma,R}$/8kpc)','ln($h_{\sigma,z}$/8kpc)']
    #?????

    if method == 'MCMC':
        #_____remove burn-in & flatten walkers_____
        ndim  = numpy.shape(chain_out)[2]
        if burnin_steps is None:
            ANALYSIS = read_RoadMapping_parameters(
                datasetname,testname=testname,
                fulldatapath=datapath
                )
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
  
    #_____prepare plot_____
    if quantities_to_plot is None: quantities_to_plot = range(ndim)
    n_quant = len(quantities_to_plot)
    n_plots = math.factorial(n_quant)
    axislist = numpy.array(range(ndim),dtype=int)
    labels = fitParNamesLatex
    fig = plt.figure(figsize=(size, size))
    fontsize = 15

    #_____iterate over all combinations of quantities_____
    for jj in range(n_quant):
        for ii in range(jj+1):

            #_____x coordinate_____
            ix = quantities_to_plot[ii]
            if method == 'GRID':
                xp = gridAxesPoints[gridAxesIndex[ix]:gridAxesIndex[ix+1]]  #grid axes points
            elif method == 'MCMC': 
                xs = chain[:,ix] #MCMC sampels

            #____is x coordinate potential?_____
            xpotflag = (ix < numpy.sum(potParFitBool))

            #_____x range_____
            if method == 'MCMC':
                #stddev and mean of the sample
                sigma_x     = numpy.std(xs)                   
                mean        = numpy.mean(xs)

                #optimal bin width (1D)
                bar_width = 3.5 * sigma_x / len(xs)**(1./3.)  
                Delta = 4.*sigma_x#max([numpy.fabs(mean-max(xs)),numpy.fabs(mean-min(xs))])
                #number of bins (1D)
                kx = int(math.ceil(Delta / (0.5 * bar_width)))   
                if ((kx % 2) == 0): kx += 1
                #histogram range (1D and plot)
                xmax = mean + kx * 0.5 * bar_width
                xmin = mean - kx * 0.5 * bar_width
                x_range = [xmin,xmax]

                #optimal bin width (2D)
                bin_width_x_2D = 3.5 * sigma_x / len(xs)**(1./4.)
                #number of bins (2D)
                kx_2D = int(math.ceil(Delta / (0.5 * bin_width_x_2D)))   
                if ((kx_2D % 2) == 0): kx_2D += 1
                #histogram range (2D)
                xmax = mean + kx_2D * 0.5 * bin_width_x_2D
                xmin = mean - kx_2D * 0.5 * bin_width_x_2D
                x_range_2D = [xmin,xmax]

            elif method == 'GRID':
                dx = 0.5 * numpy.fabs(xp[1] - xp[0])
                xmin = xp[0]  - dx
                xmax = xp[-1] + dx
                x_range = [xmin,xmax]
                x_range_2D = [xmin,xmax]
                


            #_____next plot _____
            plotpos = jj * n_quant + ii + 1
            ax = plt.subplot(n_quant,n_quant,plotpos)

            if ii == jj:
                #========== 1D probability distribution==========

                #_____marginalized probability_____
                #N: number of MCMC samples / probability at Grid point
                #b: bin edges
                if method == 'MCMC':
                    #_____histogram_____
                    N, b = numpy.histogram(xs, bins=kx, range=x_range,normed=True)
                    #coordinates:
                    dx = 0.5 * numpy.fabs(b[1] - b[0])
                    xp = b[0:-1] + dx
                elif method == 'GRID':
                    #_____marginalize grid over axes_____
                    axistuple = tuple(axislist[axislist != ix]) #axes over which we marginalize
                    logprob = numpy.log(numpy.sum(numpy.exp(loglike_out),axis=axistuple))
                    N = numpy.exp(logprob)
                    #pixel width:
                    dx_pix = numpy.fabs(xp[1] - xp[0])
                    #normalize:
                    N /= numpy.sum(N) * dx_pix
                    #bin edges:
                    b = numpy.append([xp[0]-dx],xp+dx)

                #____plot____
                xN = numpy.repeat(N,2)
                xN = numpy.concatenate(([0.],xN,[0.]))
                xb = numpy.repeat(b,2)
                #ax.plot(xb,xN,color='0.7',linewidth=1)
                if color:
                    if xpotflag: ax.fill_between(xb,0,xN,color='cornflowerblue') #'mediumpurple'
                    else:        ax.fill_between(xb,0,xN,color='limegreen')
                else:
                    ax.fill_between(xb,0,xN,color='0.7')

                #if method == 'MCMC':
                #   _____kernel density estimation_____
                #   kernel = scipy.stats.gaussian_kde(xs)
                #   values = kernel(cp)
                #   plt.plot(xp,values,color='darkorchid',linestyle='dashed',linewidth=2)

                if fit_gaussian:

                    #_____best value of the marginalized pdf_____
                    if numpy.sum(numpy.isnan(N)) == N.size:
                        x_best = truevalues[ix]
                    else:
                        indices = numpy.where(N == N.max())
                        i_x = indices[0][0]
                        x_best = xp[i_x]

                    #_____fit gaussian_____
                    popt, pcov = scipy.optimize.curve_fit(gauss, xp, N, p0=[max(N),x_best,0.3*(x_range[1]-x_range[0])])
                    A = popt[0]
                    mu = popt[1]
                    sigma = popt[2]
                    print "Attention: The Gauss curve fitting to the triangle plot does not account for possible physical upper and lower limits."+ \
                          "Therefore the result might be slightly different than what the violin plots and get_MCMC_mean_SE() calculate."         
                    xtemp = numpy.linspace(x_range[0],x_range[1],200)
                    gtemp = gauss(xtemp, A, mu, sigma)
                    plt.plot(xtemp,gtemp,color='k',linestyle='dashed',linewidth=2)
                    print labels[ix]," = ",numpy.round(mu,2)," +/- ",numpy.round(sigma,2)


                #_____format_____
                probrange = [0.,max(N)]
                ax.set_xlim(x_range)
                ax.set_ylim(probrange)

                #_____true value_____
                if color:
                    plt.vlines(trueValues[ix],probrange[0],probrange[1],color='0.2',linestyle='dotted',linewidth=2)
                else:
                    plt.vlines(trueValues[ix],probrange[0],probrange[1],color='0.1',linestyle='dotted',linewidth=2)

                #_____axis label and ticks_____
                ax =  plt.gca()
                ax.locator_params(nbins=5)
                if n_quant == 1:
                    #x axis:
                    plt.xlabel(labels[ix],fontsize=fontsize)
                    plt.xticks(rotation=45)
                    #y axis:
                    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                elif ii == 0:
                    #x axis:
                    ax.xaxis.tick_top()
                    plt.xticks(rotation=45) 
                    ax.set_xlabel(labels[ix])
                    ax.xaxis.set_label_position('top')
                    #y axis:
                    plt.tick_params(\
                        axis='y',          # changes apply to the y-axis
                        which='both',      # both major and minor ticks are affected
                        left='off',      # ticks along the left edge are off
                        right='off',         # ticks along the right edge are off
                        labelleft='off') # labels along the left edge are off
                elif ii == n_quant-1:
                    #bottom x axis:
                    ax.set_xlabel(labels[ix],fontsize=fontsize)
                    ax.xaxis.set_label_coords(0.5, -0.25)
                    plt.xticks(rotation=45)
                    #y axis:
                    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                    #top x axis:
                    ax2 = ax.twiny()
                    ax2.plot(xb,xN, alpha=0)
                    ax2.axis([x_range[0],x_range[1],probrange[0],probrange[1]])
                    ax2.locator_params(nbins=5)
                    ax2.set_xlabel(labels[ix],fontsize=fontsize)
                    plt.xticks(rotation=45)
                else:
                    #bottom x axis:
                    ax.xaxis.set_ticklabels([])
                    #y axis:
                    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                    #top x axis:
                    ax2 = ax.twiny()
                    ax2.plot(xb,xN, alpha=0)
                    ax2.axis([x_range[0],x_range[1],probrange[0],probrange[1]])
                    ax2.set_xlabel(labels[ix],fontsize=fontsize)
                    plt.xticks(rotation=45)
                    ax2.locator_params(nbins=5)
                    ax2.tick_params(width=1.5,which='both',axis='x')

                #thickness of axes & ticks:
                ax.spines['top'].set_linewidth(1.5)
                ax.spines['right'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                ax.xaxis.set_tick_params(width=1.5,which='both')
                ax.yaxis.set_tick_params(width=1.5,which='both')

            
            else:
            #========== 2D probability distribution==========

                #_____y coordinate_____
                jy = quantities_to_plot[jj]
                if method == 'GRID':
                    yp = gridAxesPoints[gridAxesIndex[jy]:gridAxesIndex[jy+1]]  #grid axes points
                elif method == 'MCMC': 
                    ys = chain[:,jy] #MCMC sampels

                #____is x coordinate potential?_____
                ypotflag = (jy < numpy.sum(potParFitBool))


                #_____y range_____
                if method == 'MCMC':
                    #stddev and mean of the sample
                    sigma_y     = numpy.std(ys)                   
                    mean      = numpy.mean(ys)

                    #optimal bin width (1D)
                    bar_width = 3.5 * sigma_y / len(ys)**(1./3.)  
                    Delta = 4.*sigma_y#max([numpy.fabs(mean-max(ys)),numpy.fabs(mean-min(ys))])
                    #number of bins (1D)
                    ky = int(math.ceil(Delta / (0.5 * bar_width)))   
                    if ((ky % 2) == 0): ky += 1
                    #histogram range (1D and plot)
                    ymax = mean + ky * 0.5 * bar_width
                    ymin = mean - ky * 0.5 * bar_width
                    y_range = [ymin,ymax]

                    #optimal bin width (2D)
                    bin_width_y_2D = 3.5 * sigma_y / len(ys)**(1./4.)
                    #number of bins (2D)
                    ky_2D = int(math.ceil(Delta / (0.5 * bin_width_y_2D)))   
                    if ((ky_2D % 2) == 0): ky_2D += 1
                    #histogram range (2D)
                    ymax = mean + ky_2D * 0.5 * bin_width_y_2D
                    ymin = mean - ky_2D * 0.5 * bin_width_y_2D
                    y_range_2D = [ymin,ymax]

                elif method == 'GRID':
                    dy = 0.5 * numpy.fabs(yp[1] - yp[0])
                    ymin = yp[0]  - dy
                    ymax = yp[-1] + dy
                    y_range = [ymin,ymax]
                    y_range_2D = [ymin,ymax]


                #_____marginalized probability_____
                #N: number of MCMC samples / probability at Grid point
                #bx,by: bin edges
                if method == 'MCMC':
                    #_____2D histogram______
                    N, bx, by = numpy.histogram2d(xs,ys,bins=[kx_2D,ky_2D],range=[x_range_2D,y_range_2D])
                    #coordinates:
                    dx = 0.5 * (bx[1] - bx[0])
                    dy = 0.5 * (by[1] - by[0])
                    xx,yy = bx[0:-1]+dx, by[0:-1]+dy
                elif method == 'GRID':
                    #_____marginalize grid over axes_____
                    axistuple = tuple(axislist[(axislist != ix) * (axislist != jy)]) #axes over which we marginalize
                    logprob = numpy.log(numpy.sum(numpy.exp(loglike_out),axis=axistuple))
                    N = numpy.exp(logprob)
                    #pixel width:
                    dx_pix = numpy.fabs(xp[1] - xp[0])
                    dy_pix = numpy.fabs(yp[1] - yp[0])
                    #normalize:
                    N /= numpy.sum(N) * dx_pix *dy_pix
                    #bin edges:
                    bx = numpy.append([xp[0]-dx],xp+dx)
                    by = numpy.append([yp[0]-dy],yp+dy)
                    #coordinates:
                    xx = xp
                    yy = yp

                #if method == 'MCMC':
                #   _____kernel density estimation_____
                #   data = numpy.vstack([xs, ys])
                #   kernel = scipy.stats.gaussian_kde(data)
                #   xm,ym = numpy.meshgrid(xx,yy)
                #   positions = numpy.vstack([xm.ravel(), ym.ravel()])
                #   values = kernel(positions)
                #   Z = numpy.reshape(kernel(positions).T, xm.shape)


                #_____calculate levels of confidence intervals_____
                sortarr = numpy.sort(N.flatten())
                cumulative = numpy.cumsum(sortarr)  #how much percenter are BELOW the current element
                con68upper = numpy.min(sortarr[cumulative >= (1.-0.6827) * cumulative[-1]])
                con68lower = numpy.max(sortarr[cumulative <= (1.-0.6827) * cumulative[-1]])
                con68 = 0.5 * (con68upper + con68lower)
                con95upper = numpy.min(sortarr[cumulative >= (1.-0.9545) * cumulative[-1]])
                con95lower = numpy.max(sortarr[cumulative <= (1.-0.9545) * cumulative[-1]])
                con95 = 0.5 * (con95upper + con95lower)
                con99upper = numpy.min(sortarr[cumulative >= (1.-0.9973) * cumulative[-1]])
                con99lower = numpy.max(sortarr[cumulative <= (1.-0.9973) * cumulative[-1]])
                con99 = 0.5 * (con99upper + con99lower)

                #_____plot contours_____
                if color:
                    if   xpotflag and ypotflag:         ax.contourf(xx,yy,N.T,[con99,con95,con68,cumulative[-1]],colors=('lightskyblue','cornflowerblue','royalblue'))
                    elif not xpotflag and not ypotflag: ax.contourf(xx,yy,N.T,[con99,con95,con68,cumulative[-1]],colors=('lightsage','limegreen','seagreen'))
                    elif xpotflag or ypotflag:          ax.contourf(xx,yy,N.T,[con99,con95,con68,cumulative[-1]],colors=('mediumpurple','darkorchid','indigo'))
                else:
                    ax.contourf(xx,yy,N.T,[con99,con95,con68,cumulative[-1]],colors=('0.7','0.5','0.3'))
                #ax.contour(xx,yy,N.T,[con68,con95,con99],colors=['g','b','r'],linestyles=['solid','solid','solid'])

                #_____scatter plot_____
                #plt.scatter(xs,ys,marker='o',edgecolor='None',s=5,alpha=0.1,color='0.5')

                #_____true value_____
                if color:
                    plt.vlines(trueValues[ix],y_range[0],y_range[1],color='0.2',linestyle='dotted',linewidth=2)
                    plt.hlines(trueValues[jy],x_range[0],x_range[1],color='0.2',linestyle='dotted',linewidth=2)
                else:
                    plt.vlines(trueValues[ix],y_range[0],y_range[1],color='0.1',linestyle='dotted',linewidth=2)
                    plt.hlines(trueValues[jy],x_range[0],x_range[1],color='0.1',linestyle='dotted',linewidth=2)

                #_____axes_____
                plt.axis([x_range[0],x_range[1],y_range[0],y_range[1]])
                ax =  plt.gca()
                ax.locator_params(nbins=5)
                if (ii == 0) & (jj == (n_quant-1)):
                    ax.set_xlabel(labels[ix],fontsize=fontsize)
                    ax.xaxis.set_label_coords(0.5, -0.25)
                    plt.xticks(rotation=45)
                    plt.ylabel(labels[jy],fontsize=fontsize)
                elif ii == 0:
                    plt.ylabel(labels[jy],fontsize=fontsize)
                    ax.xaxis.set_ticklabels([])
                elif jj == n_quant-1:
                    ax.set_xlabel(labels[ix],fontsize=fontsize)
                    ax.xaxis.set_label_coords(0.5, -0.25)
                    plt.xticks(rotation=45)
                    ax.yaxis.set_ticklabels([])
                else:
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])

                #thickness of axes & ticks:
                ax.spines['top'].set_linewidth(1.5)
                ax.spines['right'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                ax.xaxis.set_tick_params(width=1.5,which='both')
                ax.yaxis.set_tick_params(width=1.5,which='both')

                    
    
    #_____adjust and save plot_____
    plt.subplots_adjust(hspace=0,wspace=0,bottom=0.07, right=0.99, top=0.95,left=0.07) 
    #plt.tight_layout()
    plt.savefig(plotfilename,format='eps', dpi=300)

#--------------------------------------------------------------

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        plotfilename = '../out/'+sys.argv[1]+'_triangle.eps'
        plot_triangle_flexible(sys.argv[1],plotfilename,method=sys.argv[2],testname=None)
    elif len(sys.argv) == 4:
        plot_triangle_flexible(sys.argv[1],sys.argv[2],method=sys.argv[3],testname=None)
    elif len(sys.argv) == 5:
        plot_triangle_flexible(sys.argv[1],sys.argv[2],method=sys.argv[3],testname=sys.argv[4])
    elif len(sys.argv) == 6:
        testname = sys.argv[4]
        if testname == 'None':
            plot_triangle_flexible(sys.argv[1],sys.argv[2],method=sys.argv[3],testname=None,datapath=sys.argv[5]+'/')
        else:
            plot_triangle_flexible(sys.argv[1],sys.argv[2],method=sys.argv[3],testname=sys.argv[4],datapath=sys.argv[5]+'/')
    else:
        print "Error in plot_triangle_flexible(): Wrong number of input parameters."

