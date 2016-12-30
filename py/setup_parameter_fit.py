#_____import packages_____
import numpy
import sys
from read_RoadMapping_parameters import read_RoadMapping_parameters

def setup_parameter_fit(datasetname,testname=None,mockdatapath='../data/',print_to_screen=False):

    """
        NAME:
           setup_parameter_fit
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2015-11-30 - Started setup_parameter_fit.py on the basis of BovyCode/py/setup_parameter_fit.py - Trick (MPIA)
           2016-01-18 - Added pottype 5 and 51, Miyamoto-Nagai disk, Hernquist halo + Hernquist bulge for Elena D'Onghias Simulation - Trick (MPIA)
           2015-02-16 - Added pottype 6,61,7,71, DISK+HALO+BULGE potentials. - Trick (MPIA)
           2016-09-22 - Added pottype 4 and 41, MWPotential2014 by Bovy (2015) - Trick (MPIA)
           2016-09-25 - Added pottype 42 and 421, MWPotential from galpy - Trick (MPIA)
           2016-12-30 - Added pottype 8, 81 (for fitting to Gaia data). - Trick (MPIA)
    """

    #read analysis parameters:
    out = read_RoadMapping_parameters(
                datasetname,testname=testname,
                mockdatapath=mockdatapath,
                print_to_screen=False
                )

    #Reference scales:
    _REFR0 = 8.                 #[kpc], Reference radius 
    _REFV0 = 220.               #[km/s], Reference velocity

    #names of fit parameters:
    fitParNamesLatex  = []
    fitParNamesScreen = []

    #prepare one flattened array that will contain 
    #all points on the axes of the fitting range:
    potParFitNo   = out['potParFitNo']
    potParFitBool = out['potParFitBool']
    dfParFitNo    = out['dfParFitNo']
    dfParFitBool  = out['dfParFitBool']
    gridPointNo     = numpy.append(potParFitNo[potParFitBool],dfParFitNo[dfParFitBool])
    #this array gives start and end index of each axis in flattened array:   
    gridPointCumsum = numpy.cumsum(gridPointNo)
    gridAxesIndex = numpy.append([0],gridPointCumsum)
    ind = gridAxesIndex
    #this array will contain all the axes within one flattened array:
    gridAxesPoints  = numpy.zeros(numpy.sum(gridPointNo))
    kk = 0  #index that counts the fit parameters

    #width and min and max of potential grid:
    d_potcoords   = []
    min_potcoords = []
    max_potcoords = []

    #min and max for initializing MCMC walkers:
    min_walkerpos = []
    max_walkerpos = []

    #===================
    #=====POTENTIAL=====
    #===================

    #_____get fit limits & potential type_____
    pottype        = out['pottype' ]
    potParMin_phys = out['potParMin_phys']
    potParMax_phys = out['potParMax_phys']
    potParEst_phys = out['potParEst_phys']

    #_____test if number of grid points is reasonable_____
    npotpar = len(potParFitNo)
    for ii in range(npotpar):
        N   = potParFitNo   [ii]
        Min = potParMin_phys[ii]
        Max = potParMax_phys[ii]
        Est = potParEst_phys[ii]
        if N < 0:
            print "potParFitNo = ",potParFitNo
            sys.exit("Error in setup_parameter_fit(): "+\
             "Only postive integers allowed as number of grid points.")
        if N == 1 and ((Min != Max) or (Min != Est)):
            sys.exit("Error in setup_parameter_fit(): "+\
             "It is unclear which value to use in analysis for the "+str(ii)+\
             "th potential parameter (kept fixed): Min = "+str(Min)+\
             ", Max = "+str(Max)+"or Estimate = "+str(Est))

    potNamesLatex  = out['potNamesLatex']
    potNamesScreen = out['potNamesScreen']

    #setup vectors for each parameter:
    for ii in range(npotpar):
        N = potParFitNo[ii]
        if N > 1:
            xs = numpy.linspace(potParMin_phys[ii],potParMax_phys[ii],N)
            fitParNamesLatex.append(potNamesLatex[ii])
            fitParNamesScreen.append(potNamesScreen[ii])
            #save axes in physical units:
            gridAxesPoints[ind[kk]:ind[kk+1]] = xs
            #width of grid in physical units:
            dx = 0.5 * numpy.fabs(xs[1]-xs[0])      
            d_potcoords.extend([dx])
            #limits of the potential grid in physical units: 
            min_potcoords.extend([(min(xs)-dx)])
            max_potcoords.extend([(max(xs)+dx)])
            #inital walker positions:
            mid = (len(xs)-1)/2
            min_walkerpos.extend([xs[mid]-dx])
            max_walkerpos.extend([xs[mid]+dx])
            #count fit parameters
            kk += 1
            if print_to_screen:
                print potNamesScreen[ii],'\t:\t',N,'\tin\t',
                print '[',potParMin_phys[ii],',',potParMax_phys[ii],']'
        elif N == 1:
            xs = numpy.array([potParEst_phys[ii]])
        #assign arrays to quantities:
        if ii == 0: rs = xs 
        if ii == 1: vs = xs 
        if pottype == 1:    #ISOCHRONE, potPar = [R0_kpc,vc_kms,b_kpc]
            if ii == 2: bs = xs
        elif pottype == 2 or pottype == 21:  #2-COMP KK STAECKEL POTENTIAL, potPar = [R0_kpc,vc_kms,Delta,ac_D,ac_H,k]
            if   ii == 2: Deltas = xs
            elif ii == 3: ac_Ds  = xs
            elif ii == 4: ac_Hs  = xs
            elif ii == 5: ks     = xs
        elif pottype == 3 or pottype == 31: #MW-LIKE POTENTIAL, potPar = [R0_kpc,vc_kms,Rd_kpc,zh_kpc,fh,dlnvcdlnr]
            if   ii == 2: Rds   = xs
            elif ii == 3: zhs   = xs
            elif ii == 4: fhs   = xs
            elif ii == 5: dvdrs = xs
        elif pottype in numpy.array([4,41,42,421],dtype=int):  #MWPotential2014 or MWPotential from galpy, potPar = [R0_kpc,vc_kms]
            pass
        elif pottype in numpy.array([5,51,6,61,7,71],dtype=int): #DISK+HALO+BULGE POTENTIAL, potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
            #5: MNd Hh Hb POTENTIAL
            #6: expd Hh Hb POTENTIAL (here a_disk_kpc = hr_disk_kpc, b_disk_kpc = hz_disk_kpc)
            #7: MNd NFWh Hb POTENTIAL
            if   ii == 2: a_ds = xs
            elif ii == 3: b_ds = xs
            elif ii == 4: f_hs = xs
            elif ii == 5: a_hs = xs  
        elif pottype in numpy.array([8,81],dtype=int): #MN-DISK+NFW-HALO+H-BULGE POTENTIAL, potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc,M_bulge_1010Msun,a_bulge_kpc]
            if   ii == 2: a_ds = xs
            elif ii == 3: b_ds = xs
            elif ii == 4: f_hs = xs
            elif ii == 5: a_hs = xs  
            elif ii == 6: M_bs = xs
            elif ii == 7: a_bs = xs   
        else:
            sys.exit("Error in setup_parameter_fit(): "+\
             "potential type "+str(pottype)+" is not defined.")
    
    #setup parameter grid:
    potParArr_phys = numpy.zeros((numpy.prod(potParFitNo),npotpar))
    if pottype == 1:    #ISOCHRONE, potPar = [R0_kpc,vc_kms,b_kpc]
        r1,v1,b1 = numpy.meshgrid(rs,vs,bs,indexing='ij')
        potParArr_phys[:,2] = b1.flatten()
    elif pottype == 2 or pottype == 21:  #2-COMP KK STAECKEL POTENTIAL, potPar = [R0_kpc,vc_kms,Delta,ac_D,ac_H,k]
        r1,v1,D1,ad1,ah1,k1 = numpy.meshgrid(rs,vs,Deltas,ac_Ds,ac_Hs,ks,indexing='ij')
        potParArr_phys[:,2] =  D1.flatten()
        potParArr_phys[:,3] = ad1.flatten()
        potParArr_phys[:,4] = ah1.flatten()
        potParArr_phys[:,5] =  k1.flatten()
    elif pottype == 3 or pottype == 31: #MW-LIKE POTENTIAL, potPar = [R0_kpc,vc_kms,Rd_kpc,zh_kpc,fh,dlnvcdlnr]
        r1,v1,Rd1,zh1,fh1,dvdr1 = numpy.meshgrid(rs,vs,Rds,zhs,fhs,dvdrs,indexing='ij')
        potParArr_phys[:,2] =   Rd1.flatten()
        potParArr_phys[:,3] =   zh1.flatten()
        potParArr_phys[:,4] =   fh1.flatten()
        potParArr_phys[:,5] = dvdr1.flatten()
    elif pottype in numpy.array([4,41,42,421],dtype=int): #MWPotential2014 or MWPotential from galpy, potPar = [R0_kpc,vc_kms]
        r1,v1 = numpy.meshgrid(rs,vs,indexing='ij')
    elif pottype in numpy.array([5,51,6,61,7,71],dtype=int): #DISK+HALO+BULGE POTENTIAL, potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc]
        r1,v1,a_d1,b_d1,f_h1,a_h1 = numpy.meshgrid(rs,vs,a_ds,b_ds,f_hs,a_hs,indexing='ij')
        potParArr_phys[:,2] = a_d1.flatten()
        potParArr_phys[:,3] = b_d1.flatten()
        potParArr_phys[:,4] = f_h1.flatten()
        potParArr_phys[:,5] = a_h1.flatten()
    elif pottype in numpy.array([8,81],dtype=int): #MN-DISK+NFW-HALO+H-BULGE POTENTIAL, potPar = [R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc,M_bulge_1010Msun,a_bulge_kpc]
        r1,v1,a_d1,b_d1,f_h1,a_h1,M_b1,a_b1 = numpy.meshgrid(rs,vs,a_ds,b_ds,f_hs,a_hs,M_bs,a_bs,indexing='ij')
        potParArr_phys[:,2] = a_d1.flatten()
        potParArr_phys[:,3] = b_d1.flatten()
        potParArr_phys[:,4] = f_h1.flatten()
        potParArr_phys[:,5] = a_h1.flatten()
        potParArr_phys[:,6] = M_b1.flatten()
        potParArr_phys[:,7] = a_b1.flatten()
    else:
        sys.exit("Error in setup_parameter_fit(): "+\
             "potential type "+str(pottype)+\
             " of data set "+datasetname+" in test "+testname+" is not defined.")
    potParArr_phys[:,0] = r1.flatten()
    potParArr_phys[:,1] = v1.flatten()

    #shape tuple for potentials:
    potShape = numpy.squeeze(r1).shape

    #transform to numpy arrays:
    d_potcoords   = numpy.array(d_potcoords)
    min_potcoords = numpy.array(min_potcoords)
    max_potcoords = numpy.array(max_potcoords)


    #=============
    #=====QDF=====
    #=============

    #_____get fit limits_____
    dfParMin_fit = out['dfParMin_fit']
    dfParMax_fit = out['dfParMax_fit']
    dfParEst_fit = out['dfParEst_fit']

    #_____test if number of grid points is reasonable_____
    ndfpar = len(dfParFitNo)
    for ii in range(ndfpar):
        N   = dfParFitNo  [ii]
        Min = dfParMin_fit[ii]
        Max = dfParMax_fit[ii]
        Est = dfParEst_fit[ii]
        if N < 0:
            print "dfParFitNo = ",dfParFitNo
            sys.exit("Error in setup_parameter_fit(): "+\
             "Only postive integers allowed as number of grid points.")
        if N == 1 and ((Min != Max) or (Min != Est)):
            sys.exit("Error in setup_parameter_fit(): "+\
             "It is unclear which value to use in analysis for the "+str(ii)+\
             "th DF parameter (kept fixed): Min = "+str(Min)+\
             ", Max = "+str(Max)+"or Estimate = "+str(Est))

    #_____setup fit grid_____
    #dfPar_phys = [hr_kpc,sr_kms,sz_kms,hsr_kpc,hsz_kpc]
    dfNamesLatex  = out['dfNamesLatex']
    dfNamesScreen = out['dfNamesScreen']
    
    #setup vectors for each parameter:
    for ii in range(ndfpar):
        N = dfParFitNo[ii]
        if N > 1:
            xs = numpy.linspace(dfParMin_fit[ii],dfParMax_fit[ii],N)
            fitParNamesLatex.append(dfNamesLatex[ii])
            fitParNamesScreen.append(dfNamesScreen[ii])
            #save axes in logarithmic fit units
            gridAxesPoints[ind[kk]:ind[kk+1]] = xs
            #width of grid in fit units:
            dx = 0.5 * numpy.fabs(xs[1]-xs[0])
            #inital walker positions:
            mid = (len(xs)-1)/2
            min_walkerpos.extend([xs[mid]-dx])
            max_walkerpos.extend([xs[mid]+dx])
            #count fit parameter
            kk += 1 
            if print_to_screen:
                print dfNamesScreen[ii],'\t:\t',N,'\tin\t',
                print '[',dfParMin_fit[ii],',',dfParMax_fit[ii],']'
        elif N == 1:
            xs = numpy.array([dfParEst_fit[ii]])
        #assign arrays to quantities:
        if ii == 0: lnhrs  = xs
        if ii == 1: lnsrs  = xs
        if ii == 2: lnszs  = xs
        if ii == 3: lnhsrs = xs
        if ii == 4: lnhszs = xs
    #setup parameter grid:
    lnhrs1,lnsrs1,lnszs1,lnhsrs1,lnhszs1 = \
               numpy.meshgrid(lnhrs,lnsrs,lnszs,lnhsrs,lnhszs,indexing='ij')
    dfParArr_fit = numpy.zeros((numpy.prod(dfParFitNo),ndfpar))
    dfParArr_fit[:,0] = lnhrs1.flatten()
    dfParArr_fit[:,1] = lnsrs1.flatten()
    dfParArr_fit[:,2] = lnszs1.flatten()
    dfParArr_fit[:,3] = lnhsrs1.flatten()
    dfParArr_fit[:,4] = lnhszs1.flatten()
    #shape tuple for distribution function
    dfShape = numpy.squeeze(lnhrs1).shape

    #tranform to numpy arrays:
    min_walkerpos = numpy.array(min_walkerpos)
    max_walkerpos = numpy.array(max_walkerpos)


    return {'potParArr_phys':potParArr_phys, 'potShape':potShape,
            'dfParArr_fit'  :dfParArr_fit , 'dfShape' :dfShape,
            'd_potcoords':d_potcoords,'min_potcoords':min_potcoords,'max_potcoords':max_potcoords,
            'min_walkerpos':min_walkerpos,'max_walkerpos':max_walkerpos,
            'fitParNamesLatex':fitParNamesLatex,'fitParNamesScreen':fitParNamesScreen,
            'gridAxesPoints':gridAxesPoints,'gridPointNo':gridPointNo,'gridAxesIndex':gridAxesIndex}

#----------------------------------------------------------------
