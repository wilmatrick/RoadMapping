import matplotlib.pyplot as plt
import numpy
import sys
import pickle
sys.path.insert(0,'/home/trick/RoadMapping/py')
from read_RoadMapping_parameters import read_RoadMapping_parameters

def plot_progress_of_MCMC(datasetname,plotfilename,testname=None,datapath='../data/',outputpath='../out/',nwalkers=64):

    """
        NAME:
           plot_progress_of_MCMC
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-12-07 - Started plot_progress_of_MCMC.py on the basis of 
                        BovyCode/pyplots/plot_progress_of_MCMC.py - Trick (MPIA)
           2021-10-20 - Made file paths consistent. - Trick (MPA)
    """

    #_____read MCMC chain_____
    if testname is None: filename = outputpath+datasetname+"_chain_MCMC.dat"
    else:                filename = outputpath+datasetname+"_"+testname+"_chain_MCMC.dat"
    out = numpy.loadtxt(filename)

    #____Read names & True values_____
    if testname is None: filename = outputpath+datasetname+"_parameters_MCMC.sav"
    else:                filename = outputpath+datasetname+"_"+testname+"_parameters_MCMC.sav"
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
            mockdatapath=datapath
            )
    estimatedValues = numpy.append(ANALYSIS['potParEst_phys'][potParFitBool],ANALYSIS['dfParEst_fit'][dfParFitBool])
    fiducialValues  = ANALYSIS['dfParFid_fit'][dfParFitBool]


    #_____format MCMC chain for plotting_____
    npos = len(out[:,0])
    print(numpy.shape(out))
    nsteps = npos // nwalkers
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
        im = ax.scatter(steps[index],chain[:,ii][index],c=colors[index],cmap='magma',edgecolor='None',vmin=-5.,vmax=0.,alpha=0.3)
        ax.set_xlim([min(steps),max(steps)])
        if ii > len(trueValues)-1:
            ax.set_ylabel("auxiliary")
        else:
            ax.set_ylabel(fitParNamesLatex[ii])
            npot = numpy.sum(potParFitBool) #number of potential fit parameters
            if ii >= npot: #DF parameter
                ax.hlines(estimatedValues[ii],min(steps),max(steps),color='gold',linewidth=3,
                          linestyle='dashed',label='estimated')
                ax.hlines(fiducialValues[ii-npot],min(steps),max(steps),color='red',linewidth=3,
                          linestyle='dashed',label='fiducial')
                ax.hlines(trueValues[ii],min(steps),max(steps),color='limegreen',linewidth=3,
                          linestyle='dashed',label='true')
            elif ii < npot: #pot parameter
                ax.hlines(estimatedValues[ii],min(steps),max(steps),color='gold',linewidth=3,linestyle='dashed')
                ax.hlines(trueValues[ii],min(steps),max(steps),color='limegreen',linewidth=3,linestyle='dashed')
        ax.locator_params(nbins=5)
        if ii < nplots-1:
            ax.axes.get_xaxis().set_ticklabels([])
        elif ii == nplots-1:
            ax.legend()
            #cb = plt.colorbar(im,orientation='horizontal')
            #cb.set_label('~ log prob')
            ax.set_xlabel('MCMC step')

    #_____adjust and save plot_____
    plt.subplots_adjust(hspace=0.2) 
    plt.tight_layout()
    plt.savefig(plotfilename)

#--------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) == 2:
        plotfilename = '../out/'+sys.argv[1]+'_progress_MCMC.png'
        plot_progress_of_MCMC(sys.argv[1],plotfilename,testname=None,nwalkers=100)
    elif len(sys.argv) == 3:
        plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=None,nwalkers=100)
    elif len(sys.argv) == 5:
        plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=sys.argv[3],nwalkers=int(sys.argv[4]))
    elif len(sys.argv) == 6:
        testname = sys.argv[3]
        if testname == 'None':
            plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=None,datapath=sys.argv[4]+'/',nwalkers=int(sys.argv[5]))
        else:
            plot_progress_of_MCMC(sys.argv[1],sys.argv[2],testname=testname,datapath=sys.argv[4]+'/',nwalkers=int(sys.argv[5]))
