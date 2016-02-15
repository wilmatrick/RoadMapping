import matplotlib.pyplot as plt
import numpy
import sys
import pickle
import colormaps as cmaps
sys.path.insert(0,'/home/trick/RoadMapping/py')
from read_RoadMapping_parameters import read_RoadMapping_parameters

datasetname = "isoSphFlex_short_hot_2kpc_1a"

def plot_progress_of_MCMC(datasetname,plotfilename,testname=None,datapath='/home/trick/ElenaSim/out/'):

    """
        NAME:
           plot_progress_of_MCMC
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-12-07 - Started plot_progress_of_MCMC.py on the basis of BovyCode/pyplots/plot_progress_of_MCMC.py - Trick (MPIA)
    """

    #_____read MCMC chain_____
    if testname is None: filename = datapath+datasetname+"_chain_MCMC.dat"
    else:                filename = datapath+datasetname+"_"+testname+"_chain_MCMC.dat"
    out = numpy.loadtxt(filename)

    #____Read names & True values_____
    if testname is None: filename = datapath+datasetname+"_parameters_MCMC.sav"
    else:                filename = datapath+datasetname+"_"+testname+"_parameters_MCMC.sav"
    savefile= open(filename,'rb')
    fitParNamesLatex   = pickle.load(savefile)      #names of axes in Latex
    gridPointNo        = pickle.load(savefile)      #number of points along each axis
    gridAxesPoints     = pickle.load(savefile)      #all axes in one flattened array
    gridAxesIndex      = pickle.load(savefile)      #indices of start and end of each axis in above array
    potParFitBool      = pickle.load(savefile)      #boolean array that indicates which of all potential parameters are fitted
    dfParFitBool       = pickle.load(savefile)      #boolean array that indicates which of all DF parameters are fitted
    potParTrue_phys    = pickle.load(savefile)      #true potential parameters in physical units
    dfParTrue_fit      = pickle.load(savefile)      #true df parameters in logarithmic fit units
    savefile.close()
    trueValues = numpy.append(potParTrue_phys[potParFitBool],dfParTrue_fit[dfParFitBool])

    #_____Estimates_____
    #read analysis parameters:
    ANALYSIS = read_RoadMapping_parameters(
            datasetname,
            testname=testname,
            fulldatapath=datapath
            )
    estimatedValues = numpy.append(ANALYSIS['potParEst_phys'][potParFitBool],ANALYSIS['dfParEst_fit'][dfParFitBool])
    fiducialValues  = ANALYSIS['dfParFid_fit'][dfParFitBool]


    #_____format MCMC chain for plotting_____
    npos = len(out[:,0])
    print npos
    nwalkers = 100
    nsteps = npos / nwalkers
    chain  = out[0:nwalkers*nsteps,0:-1]
    steps = numpy.repeat(range(nsteps),nwalkers)
    colors = out[0:nwalkers*nsteps,-1]
    colors = colors-numpy.max(colors)
    colors = -numpy.sqrt(2.*numpy.fabs(colors))
    nplots = len(chain[0,:])
    

    #_____plot_____
    fig = plt.figure(figsize=(10, 2*nplots))
    for ii in range(nplots):
        ax = plt.subplot(nplots,1,ii+1)
        index = sorted(range(nwalkers*nsteps),key=lambda x: colors[x])
        plt.scatter(steps[index],chain[:,ii][index],c=colors[index],cmap=cmaps.magma,edgecolor='None',vmin=-5.,vmax=0.,alpha=0.3)
        ax.set_xlim([min(steps),max(steps)])
        if ii > len(trueValues)-1:
            plt.ylabel("auxiliary")
        else:
            plt.ylabel(fitParNamesLatex[ii])
            npot = numpy.sum(potParFitBool) #number of potential fit parameters
            if ii >= npot: #DF parameter
                plt.hlines(estimatedValues[ii],min(steps),max(steps),color='gold',linewidth=3,linestyle='dashed')
                plt.hlines(fiducialValues[ii-npot],min(steps),max(steps),color='red',linewidth=3,linestyle='dashed')
            elif ii < npot: #pot parameter
                plt.hlines(estimatedValues[ii],min(steps),max(steps),color='gold',linewidth=3,linestyle='dashed')
                plt.hlines(trueValues[ii],min(steps),max(steps),color='limegreen',linewidth=3,linestyle='dashed')
        plt.locator_params(nbins=5)
        if ii < nplots-1:
            ax.axes.get_xaxis().set_ticklabels([])
    plt.colorbar(orientation='horizontal')

    #_____adjust and save plot_____
    plt.subplots_adjust(hspace=0.1) 
    plt.tight_layout()
    plt.savefig(plotfilename)

#--------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) == 2:
        plotfilename = '../out/'+sys.argv[1]+'_progress_MCMC.png'
        plot_progress_of_MCMC(sys.argv[1],plotfilename,testname=None)
    elif len(sys.argv) == 3:
        plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=None)
    elif len(sys.argv) == 4:
        plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=sys.argv[3])
