#_____import packages_____
import numpy
import colormaps as cmaps
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import pickle
import math

def plot_mockdata_xv_triangle(datasetname,plotfilename,
                              quantities_to_plot=[0,1,2,3,4,5,6,7],size=15,
                              mockdatapath='../data/'):

    #_____load data from file_____
    datafilename = mockdatapath+datasetname+"/"+datasetname+"_mockdata.sav"
    savefile= open(datafilename,'rb')
    Rs_kpc = pickle.load(savefile)
    ndata = len(Rs_kpc)
    data = numpy.zeros((8,ndata))
    data[0,:] = Rs_kpc                  #R   [kpc]
    data[1,:] = pickle.load(savefile)   #vR  [km/s]
    phis_deg  = pickle.load(savefile)   
    data[2,:] = phis_deg                #phi [deg]
    data[3,:] = pickle.load(savefile)   #vT  [km/s] 
    data[4,:] = pickle.load(savefile)   #z   [kpc]
    data[5,:] = pickle.load(savefile)   #vz  [km/s]
    savefile.close()
    data[6,:] = Rs_kpc * numpy.cos(phis_deg/180.*math.pi)   #x [kpc]
    data[7,:] = Rs_kpc * numpy.sin(phis_deg/180.*math.pi)   #y [kpc]
    labels = ['$R$ [kpc]','$v_R$ [km s$^{-1}$]','$\phi$ [deg]','$v_T$ [km s$^{-1}$]','$z$ [kpc]','$v_z$ [km s$^{-1}$]','$x$ [kpc]','$y$ [kpc]']

    ##################################################
  
    #_____prepare plot_____
    n_quant = len(quantities_to_plot)
    n_plots = math.factorial(n_quant)
    fig = plt.figure(figsize=(size, size))
    fontsize = 15

    #_____iterate over all combinations of quantities_____
    for jj in range(n_quant):
        for ii in range(jj+1):

            #_____x coordinate_____
            ix = quantities_to_plot[ii]
            xs = data[ix,:]
            """x_range = [min(xs),max(xs)]

            #stddev of the sample
            sigma_x     = numpy.std(xs)
            #optimal bin width (1D)
            bar_width = 3.5 * sigma_x / len(xs)**(1./3.)
            #number of bins (1D)
            kx = int(math.ceil((x_range[1]-x_range[0])/bar_width))
            #optimal bin width (2D)
            bin_width_x_2D = 3.5 * sigma_x / len(xs)**(1./4.)
            #number of bins (2D)
            kx_2D = int(math.ceil((x_range[1]-x_range[0])/ bin_width_x_2D))"""
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

            #_____next plot _____
            plotpos = jj * n_quant + ii + 1
            ax = plt.subplot(n_quant,n_quant,plotpos)

            if ii == jj:
                #========== 1D distribution==========

                #_____marginalized probability_____
                #N: number of stars
                #b: bin edges         
                N, b = numpy.histogram(xs, bins=kx, range=x_range,normed=True)
                dx = 0.5 * numpy.fabs(b[1] - b[0])
                xp = b[0:-1] + dx

                #____plot____
                xN = numpy.repeat(N,2)
                xN = numpy.concatenate(([0.],xN,[0.]))
                xb = numpy.repeat(b,2)
                ax.fill_between(xb,0,xN,color='mediumvioletred')

                #_____format_____
                probrange = [0.,max(N)]
                ax.set_xlim(x_range)
                ax.set_ylim(probrange)

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
                ys = data[jy,:]
                """y_range = [min(ys),max(ys)]

                #stddev of the sample
                sigma_y     = numpy.std(ys)  
                #optimal bin width (2D)
                bin_width_y_2D = 3.5 * sigma_y / len(ys)**(1./4.)
                #number of bins (2D)
                ky_2D = int(math.ceil((y_range[1]-y_range[0]) / bin_width_y_2D)) """
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

                #_____2D histogram_____
                #N: number of MCMC samples / probability at Grid point
                #bx,by: bin edges
                N, bx, by = numpy.histogram2d(xs,ys,bins=[kx_2D,ky_2D],range=[x_range,y_range])
                #coordinates:
                dx = 0.5 * (bx[1] - bx[0])
                dy = 0.5 * (by[1] - by[0])
                xx,yy = bx[0:-1]+dx, by[0:-1]+dy

                #_____plot_____
                N[N == 0] = numpy.nan
                ax.imshow(N.T,origin='lower',cmap=cmaps.plasma,extent=[x_range[0],x_range[1],y_range[0],y_range[1]],aspect='auto',interpolation='nearest')

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
    plt.savefig(plotfilename,format='pdf', dpi=300)
