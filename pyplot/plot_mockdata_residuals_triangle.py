#_____import packages_____
from __past__ import division
import numpy
import colormaps as cmaps
import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import pickle
import math

def plot_mockdata_residuals_triangle(datasetname_original,testname,datasetname_reference,plotfilename,
                              quantities_to_plot=[0,1,2,3,4,5,6,7],size=15,
                              mockdatapath='../data/',with_actions=False):

    #_____load data from file_____
    #original data:
    originaldatafilename = mockdatapath+datasetname_original+"/"+datasetname_original+"_mockdata.sav"
    savefile= open(originaldatafilename,'rb')
    Rd_kpc    = pickle.load(savefile)   #R   [kpc] 
    data      = numpy.zeros((11,len(Rd_kpc)))
    data[0,:] = Rd_kpc 
    data[1,:] = pickle.load(savefile)   #vR  [km/s]
    data[2,:] = pickle.load(savefile)   #phi [deg]
    data[3,:] = pickle.load(savefile)   #vT  [km/s] 
    data[4,:] = pickle.load(savefile)   #z   [kpc] 
    data[5,:] = pickle.load(savefile)   #vz  [km/s]
    savefile.close()
    data[6,:]  = data[0,:] * numpy.cos(data[2,:]/180.*math.pi) #x [kpc]
    data[7,:]  = data[0,:] * numpy.sin(data[2,:]/180.*math.pi) #y [kpc]
    if with_actions:
        dataactionsfilename = mockdatapath+datasetname_original+"/"+datasetname_original+"_"+testname+"_mockdata_actions.sav"
        savefile  = open(dataactionsfilename,'rb')
        data[8,:]  = pickle.load(savefile)/8./220.   #Jr [kpc km/s]
        data[9,:]  = pickle.load(savefile)/8./220.   #Lz [kpc km/s]
        data[10,:] = pickle.load(savefile)/8./220.   #Jz [kpc km/s]
        savefile.close()

    #reference data:
    referencedatafilename = mockdatapath+datasetname_reference+"/"+datasetname_reference+"_mockdata.sav"
    savefile  = open(referencedatafilename,'rb')
    Rr_kpc    = pickle.load(savefile)   #R   [kpc]
    ref       = numpy.zeros((11,len(Rr_kpc)))
    ref[0,:]  = Rr_kpc
    ref[1,:]  = pickle.load(savefile)   #vR  [km/s]
    ref[2,:]  = pickle.load(savefile)   #phi [deg]
    ref[3,:]  = pickle.load(savefile)   #vT  [km/s] 
    ref[4,:]  = pickle.load(savefile)   #z   [kpc] 
    ref[5,:]  = pickle.load(savefile)   #vz  [km/s]
    savefile.close()
    ref[6,:]  = ref[0,:] * numpy.cos(ref[2,:]/180.*math.pi) #x [kpc]
    ref[7,:]  = ref[0,:] * numpy.sin(ref[2,:]/180.*math.pi) #y [kpc]
    if with_actions:
        referenceactionsfilename = mockdatapath+datasetname_reference+"/"+datasetname_reference+"_mockdata_actions.sav"
        savefile  = open(referenceactionsfilename,'rb')
        ref[8,:]  = pickle.load(savefile)/8./220.   #Jr [kpc km/s]
        ref[9,:]  = pickle.load(savefile)/8./220.   #Lz [kpc km/s]
        ref[10,:] = pickle.load(savefile)/8./220.   #Jz [kpc km/s]
        savefile.close()

    #scaling for reference data set to same number of stars as the actual data set
    frac = float(len(data[0,:])) / float(len(ref[0,:])) 

    labels = ['$R$ [kpc]','$v_R$ [km s$^{-1}$]','$\phi$ [deg]','$v_T$ [km s$^{-1}$]','$z$ [kpc]','$v_z$ [km s$^{-1}$]','$x$ [kpc]','$y$ [kpc]','$J_R$ [8 kpc 220 km s$^{-1}$]','$L_z$ [8 kpc 220 km s$^{-1}$]','$J_z$ [8 kpc 220 km s$^{-1}$]']

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
            xs_ref = ref[ix,:]

            #stddev and mean of the sample
            sigma_x     = numpy.std(xs)                   
            mean        = numpy.mean(xs)
            #optimal bin width (1D)
            bar_width = 3.5 * sigma_x / len(xs)**(1./3.)  
            Delta = numpy.max(xs) - numpy.min(xs)
            #number of bins (1D)
            kx = int(math.ceil(Delta / bar_width))
            #histogram range (1D and plot)
            x_range = [numpy.min(xs),numpy.max(xs)]
            #optimal bin width (2D)
            bin_width_x_2D = 3.5 * sigma_x / len(xs)**(1./4.)
            #number of bins (2D)
            kx_2D = int(math.ceil(Delta / bin_width_x_2D))
            #histogram range (2D)
            x_range_2D = [numpy.min(xs),numpy.max(xs)]

            #_____next plot _____
            plotpos = jj * n_quant + ii + 1
            ax = plt.subplot(n_quant,n_quant,plotpos)

            if ii == jj:
                #========== 1D distribution==========

                #_____1D residual significance_____
                #N: number of stars
                #b: bin edges         
                N, b = numpy.histogram(xs, bins=kx, range=x_range,normed=False)
                dx = 0.5 * numpy.fabs(b[1] - b[0])
                xp = b[0:-1] + dx
                #reference data histogram:
                Nref, b = numpy.histogram(xs_ref, bins=kx, range=x_range,normed=False)
                #scale histograms to same number:
                Nref = Nref * frac  
                #Residual significance:
                Nsig = (N-Nref)/numpy.sqrt(Nref)

                #____plot____
                xN = numpy.repeat(Nsig,2)
                xN = numpy.concatenate(([0.],xN,[0.]))
                xb = numpy.repeat(b,2)
                ax.plot(xb,xN,color='crimson')
                ax.plot(xb,numpy.zeros_like(xb),linestyle='dotted',color='k',linewidth=2)

                #_____format_____
                probrange = [1.1*min(xN),1.1*max(xN)]
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
                ys_ref = ref[jy,:]

                #stddev and mean of the sample
                sigma_y     = numpy.std(ys)                   
                mean      = numpy.mean(ys)

                #optimal bin width (1D)
                bar_width = 3.5 * sigma_y / len(ys)**(1./3.)  
                Delta = numpy.max(ys) - numpy.min(ys)
                #number of bins (1D)
                ky = int(math.ceil(Delta / bar_width))
                #histogram range (1D and plot)
                y_range = [numpy.min(ys),numpy.max(ys)]

                #optimal bin width (2D)
                bin_width_y_2D = 3.5 * sigma_y / len(ys)**(1./4.)
                #number of bins (2D)
                ky_2D = int(math.ceil(Delta / bin_width_y_2D))
                #histogram range (2D)
                y_range_2D = [numpy.min(ys),numpy.max(ys)]


                #_____2D histogram_____
                #N: number of MCMC samples / probability at Grid point
                #bx,by: bin edges
                N, bx, by = numpy.histogram2d(xs,ys,bins=[kx_2D,ky_2D],range=[x_range,y_range])
                #coordinates:
                dx = 0.5 * (bx[1] - bx[0])
                dy = 0.5 * (by[1] - by[0])
                xx,yy = bx[0:-1]+dx, by[0:-1]+dy
                #reference histogram:
                Nref, bx, by = numpy.histogram2d(xs_ref,ys_ref,bins=[kx_2D,ky_2D],range=[x_range,y_range])
                #scale histograms to same number:
                Nref = Nref * frac  
                #Residual significance:
                Nsig = (N-Nref)/numpy.sqrt(Nref)

                #_____plot_____
                Nsig[Nsig == 0] = numpy.nan
                vmax = 4.#numpy.nanmax(Nsig.flatten())
                ax.imshow(Nsig.T,origin='lower',cmap='RdBu',extent=[x_range[0],x_range[1],y_range[0],y_range[1]],aspect='auto',interpolation='nearest',vmin=-vmax,vmax=vmax)

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
