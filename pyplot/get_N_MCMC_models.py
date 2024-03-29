#_____import packages_____
#from __past__ import division
#from __future__ import print_function
import pickle
import sys
sys.path.insert(0,'/home/trick/RoadMapping/py')
from read_RoadMapping_parameters import read_RoadMapping_parameters
import numpy
import scipy
import scipy.stats
from outlier_model import scale_df_fit_to_phys

def get_N_MCMC_models(datasetname,testname=None,N=12,analysis_output_filename=None,mockdatapath=None,fulldatapath='../out/',randomseed=None,with_replacement=False):

    """
        NAME:
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
            2017-01-03 - Now uses scale_df_fit_to_phys to allow for flexible number of parameters. - Trick (MPIA)
            2017-03-24 - Code allows now samples to be drawn from MCMC with or without replacement. - Trick (MPIA)
            2017-04-10 - Minor bug introduced on 2017-01-03 in return value dfParModels_phys was corrected. - Trick (MPIA)
    """

    #_____reference scales_____
    _REFR0 = 8.     #[kpc]
    _REFV0 = 220.   #[km/s]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = fulldatapath+datasetname+"_analysis_output_MCMC.sav"
        else:                analysis_output_filename = fulldatapath+datasetname+"_"+testname+"_analysis_output_MCMC.sav"
    savefile= open(analysis_output_filename,'rb')    
    chain_out      = pickle.load(savefile)          #MCMC chain (nwalker,nsteps,ndim)
    fitParNamesLatex   = pickle.load(savefile)      #names of axes in Latex
    gridPointNo        = pickle.load(savefile)      #number of points along each axis
    gridAxesPoints     = pickle.load(savefile)      #all axes in one flattened array
    gridAxesIndex      = pickle.load(savefile)      #indices of start and end of each axis in above array
    potParFitBool      = pickle.load(savefile)      #boolean array that indicates which of all potential parameters are fitted
    dfParFitBool       = pickle.load(savefile)      #boolean array that indicates which of all DF parameters are fitted
    potParTrue_phys    = pickle.load(savefile)      #true potential parameters in physical units
    dfParTrue_fit      = pickle.load(savefile)      #true df parameters in logarithmic fit units
    savefile.close()

    #_____read model parameters_____
    if mockdatapath is None:
        ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            fulldatapath=fulldatapath
            )
    else:
        ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            mockdatapath=mockdatapath
            )
    potParEst_phys = ANALYSIS['potParEst_phys']
    dfParEst_fit = ANALYSIS['dfParEst_fit']
    burnin_steps = ANALYSIS['noMCMCburnin']

    #_____pick N random models from MCMC after burn-in_____
    nwalker = numpy.shape(chain_out)[0]
    nsteps  = numpy.shape(chain_out)[1]
    ndim    = numpy.shape(chain_out)[2]

    if randomseed is not None:
        numpy.random.seed(seed=randomseed)

    #old version (delete later):
    #iwalker = numpy.random.randint(0,high=nwalker,size=N)
    #istep   = numpy.random.randint(burnin_steps,high=nsteps,size=N)

    #this version of drawing random samples allows with/without replacement:
    total_samples = nwalker * (nsteps-burnin_steps)
    sample_numbers = range(total_samples)
    ind = numpy.random.choice(sample_numbers,size=N,replace=with_replacement)
    istep = ind // nwalker + burnin_steps
    iwalker = ind % nwalker 
    if len(numpy.unique(istep*nwalker+iwalker )) < N:
        sys.exit("Error in get_N_MCMC_models(): Elements to draw from MCMC are not unique.")

    models  = chain_out[iwalker,istep,:].reshape((-1,ndim))

    #_____potential parameters_____
    npotpar    = len(potParEst_phys)
    npotfitpar = numpy.sum(potParFitBool)
    potParModels_phys = numpy.tile(potParEst_phys,(N,1))
    potParModels_phys[:,potParFitBool] = models[:,0:npotfitpar]

    #_____DF parameters_____
    ndfpar    = len(dfParEst_fit)
    ndffitpar = numpy.sum(dfParFitBool)
    dfParModels_fit = numpy.tile(dfParEst_fit,(N,1))
    dfParModels_fit[:,dfParFitBool] = models[:,npotfitpar::]
    dfParModels_phys = scale_df_fit_to_phys(ANALYSIS['dftype'],dfParModels_fit)

    return potParModels_phys, dfParModels_phys
    
#----------------------------------------------------------------------------------------------

def gauss(x, A, mu, sigma):
    return A * numpy.exp(-(x-mu)**2/(2.*sigma**2))

def get_MCMC_mean_SE(datasetname,testname=None,analysis_output_filename=None,mockdatapath='../data/',fulldatapath=None,quantities_to_calculate=None,Gaussian_fit=False):

    #_____reference scales_____
    _REFR0 = 8.     #[kpc]
    _REFV0 = 220.   #[km/s]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = "../out/"+datasetname+"_analysis_output_MCMC.sav"
        else:                analysis_output_filename = "../out/"+datasetname+"_"+testname+"_analysis_output_MCMC.sav"
    savefile= open(analysis_output_filename,'rb')    
    chain_out      = pickle.load(savefile)          #MCMC chain (nwalker,nsteps,ndim)
    fitParNamesLatex   = pickle.load(savefile)      #names of axes in Latex
    gridPointNo        = pickle.load(savefile)      #number of points along each axis
    gridAxesPoints     = pickle.load(savefile)      #all axes in one flattened array
    gridAxesIndex      = pickle.load(savefile)      #indices of start and end of each axis in above array
    potParFitBool      = pickle.load(savefile)      #boolean array that indicates which of all potential parameters are fitted
    dfParFitBool       = pickle.load(savefile)      #boolean array that indicates which of all DF parameters are fitted
    potParTrue_phys    = pickle.load(savefile)      #true potential parameters in physical units
    dfParTrue_fit      = pickle.load(savefile)      #true df parameters in logarithmic fit units
    savefile.close()

    #_____burnin + bounds_____
    ANALYSIS = read_RoadMapping_parameters(
            datasetname,testname=testname,
            mockdatapath=mockdatapath,
            fulldatapath=fulldatapath
            )
    burnin_steps = ANALYSIS['noMCMCburnin']
    lower_bounds = numpy.append(ANALYSIS['potParLowerBound_phys'][potParFitBool],
                                ANALYSIS['dfParLowerBound_fit'  ][dfParFitBool ])
    upper_bounds = numpy.append(ANALYSIS['potParUpperBound_phys'][potParFitBool],
                                ANALYSIS['dfParUpperBound_fit'  ][dfParFitBool ])
    ndim    = numpy.shape(chain_out)[2]
    chain = chain_out[:, burnin_steps:, :].reshape((-1, ndim))


    #_____iterate over quantities_____
    if quantities_to_calculate is None: quantities_to_calculate = range(ndim)
    n_quant = len(quantities_to_calculate)
    means   = numpy.zeros(n_quant) + numpy.nan
    stddevs = numpy.zeros(n_quant) + numpy.nan
    medians = numpy.zeros(n_quant) + numpy.nan
    for ii in range(n_quant):

        #MCMC sampels
        ix = quantities_to_calculate[ii]
        xs = chain[:,ix] 
        mean = numpy.mean(xs)
        stddev = numpy.std(xs)

        if Gaussian_fit:
            xmin = mean-3.*stddev
            xmax = mean+3.*stddev
            if numpy.isfinite(lower_bounds[ix]): xmin = numpy.max([lower_bounds[ix],xmin])                           
            if numpy.isfinite(upper_bounds[ix]): xmax = numpy.min([upper_bounds[ix],xmax])                 
            xp = numpy.linspace(xmin,xmax,30)

            #kernel density estimation:
            kernel = scipy.stats.gaussian_kde(xs)
            prob   = kernel(xp)

            indices = numpy.where(prob == prob.max())
            i_x = indices[0]
            x_best = xp[i_x]

        
            #_____fit gaussian_____
            try:
                popt, pcov = scipy.optimize.curve_fit(gauss, xp, prob,p0=[1.,x_best,0.3*(max(xp)-min(xp))])
            except RuntimeError as e:
                pass #do nothing
            else:
                A = popt[0]
                mu = popt[1]
                sigma = numpy.fabs(popt[2])
                print(A, mu, sigma)
                if mu >= xmin:
                    means[ii] = mu
                    stddevs[ii] = sigma
                else:
                    sys.exit("Error in get_MCMC_mean_SE(): mu is smaller than xmin.")
        else:
            means[ii] = mean
            stddevs[ii] = stddev
        medians[ii] = numpy.median(xs)
            

    return means, stddevs, medians

#--------------------------------------------

def get_GRID_midpoint(datasetname,testname=None,analysis_output_filename=None,mockdatapath='../data/',fulldatapath=None,quantities_to_calculate=None):

    #_____reference scales_____
    _REFR0 = 8.     #[kpc]
    _REFV0 = 220.   #[km s$^{-1}$]

    #_____load data from file_____
    if analysis_output_filename is None:
        if testname is None: analysis_output_filename = datapath+datasetname+"_analysis_output_"+method+".sav"
        else:                analysis_output_filename = datapath+datasetname+"_"+testname+"_analysis_output_"+method+".sav"
    savefile= open(analysis_output_filename,'rb')    
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

    #_____number of dimensions_____
    ndim = len(loglike_out.shape)
    #_____rescale as probabilities are too small to plot_____
    loglike_out[numpy.isnan(loglike_out)] = -numpy.finfo(numpy.dtype(numpy.float64)).max
    loglike_out -= loglike_out.max()

    #_____iterate over quantities_____
    if quantities_to_calculate is None: quantities_to_calculate = range(ndim)
    axislist = numpy.array(range(ndim),dtype=int)
    n_quant = len(quantities_to_calculate)
    midpoint   = numpy.zeros(n_quant) + numpy.nan
    for ii in range(n_quant):

        ix = quantities_to_calculate[ii]
        xp = gridAxesPoints[gridAxesIndex[ix]:gridAxesIndex[ix+1]]  #grid axes points
        if len(xp) == 3:
            midpoint[ii] = xp[1]
        else:
            sys.exit('Error in get_GRID_midpoint: not implemented for more than 3 grid points yet.')
        midpoint
        
    return midpoint
