import galpy
from galpy.util import bovy_coords
import math
import numpy
import sys

"""
    Note on coordinate systems:
        (R,z,phi):
            galactocentric cylindrical coordinates
            in [kpc,kpc,rad]
        (x,y,z):
            rectangular/cartesian coordinates with sun at the center 
            in [kpc]
            x twds Galactic center
            y twds Galactic rotation
            z twds Galactic north pole
            moves with sun around Galactic center
        (l,b,d):
            spherical Galactic coordinates wrt to sun 
            in [rad,rad,kpc]
            l = Galactic longitude
            b = Galactic latitude
            d = distance from sun
        (vR,vz,vT):
            velocities in (R,z,phi) direction
            in [km/s]
        (vx,vy,vz):
            velocities in (x,y,z) direction
            in [km/s]
            when calculating these from Galactic coordinates, 
            the rotation velocity of the sun has to be given, 
            as these velocities are as observed from the sun
        (pm_l,pm_b,v_los):
            "velocities" in (l,b,d) direction
            in [km/s,mas/yr,mas/yr]
            pm_l = proper motion in Galactic longitude direction
                 = mu_l * cos(b), where mu_l describes the change in 
                   l at a given b. At higher latitude b the longitude 
                   circle parallel to the Galactic plane becomes 
                   smaller. The same change in l corresponds there-
                   fore to a slower velocity if observed at a higher 
                   b. To account for this mu_l has to be multiplied 
                   by cos(b), when we want a proper velocity and not 
                   just a change in angle. The galpy coordinate 
                   transformations use pm_l everywhere.
            pm_b = proper motion in Galactic latitude
            v_los = line of sight velocity / radial velocity
        (pm_ra,pm_dec):
            proper motions in equatorial coordinate system.
            in [mas/yr,mas/yr]
            pm_ra = proper motion in direction of right ascension
            pm_dec = proper motion in direction of declination
            (transformation from (pm_l,pm_b) to (pm_ra,pm_dec) is
            simply a rotation on the plane of the sky)
"""
#-------------------------------------------------------------------------------------------------

def galcencyl_to_radecDM(R_kpc,phi_rad,z_kpc,
                         quiet=True,
                         Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.): #position of Sun in galactocentric (X,Y,Z) coordinates

    if float(galpy.__version__) < 1.2:
        sys.exit("Error in galcencyl_to_galcencyl_to_radecDM(): Make sure to use the newest (>=1.2) version of galpy. There have been changes in the coordinate transformations. This is galpy version: "+galpy.__version__)

    if isinstance(R_kpc,float):
        R_kpc = numpy.array([R_kpc])
        phi_rad = numpy.array([phi_rad])
        z_kpc = numpy.array([z_kpc])
    ndata = len(R_kpc)


    #_____a. covert spatial coordinates to (ra,dec,DM)_____

    if not quiet:
        for ii in range(ndata):
            print ii, "(R,phi,z)_true \t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc,rad,kpc]" % (R_kpc[ii],phi_rad[ii],z_kpc[ii])

    # (R,z,phi) --> (x,y,z):
    xyz = bovy_coords.galcencyl_to_XYZ(
                R_kpc, phi_rad, z_kpc, 
                Xsun=Xgc_sun_kpc, Ysun=Ygc_sun_kpc, Zsun=Zgc_sun_kpc
                )
    Xs_kpc = xyz[:,0]
    Ys_kpc = xyz[:,1]
    Zs_kpc = xyz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii,"(x,y,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc]" % (Xs_kpc[ii],Ys_kpc[ii],Zs_kpc[ii])

    # (x,y,z) --> (l,b,d):
    lbd = bovy_coords.XYZ_to_lbd(
                Xs_kpc, Ys_kpc, Zs_kpc, 
                degree=False
                )
    l_rad = lbd[:,0]
    b_rad = lbd[:,1]
    d_kpc = lbd[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(l,b,d) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,kpc]" % (l_rad[ii], b_rad[ii], d_kpc[ii])

    # (l,b) --> (ra,dec)
    radec = bovy_coords.lb_to_radec(
                l_rad,b_rad,
                degree=False,epoch=2000.0
                )
    ra_rad  = radec[:,0]
    dec_rad = radec[:,1]
    if not quiet:
        for ii in range(ndata):
            print ii, "(ra,dec) \t\t= "+\
              "(%5.3f, %5.3f) \t\tin [rad]" % (ra_rad[ii],dec_rad[ii])

    # d --> DM (distance modulus):
    DM_mag = 5. * numpy.log10(d_kpc * 1000.) - 5.
    if not quiet:
       for ii in range(ndata):
            print ii, "(DM,d) \t\t= "+\
              "(%5.3f,%5.3f) \t\tin [mag,kpc]" % (DM_mag[ii],d_kpc[ii])

  
    if ndata == 1:
        return (ra_rad[0],dec_rad[0],DM_mag[0])
    else:
        return (ra_rad,dec_rad,DM_mag)

#-------------------------------------------------------------------------------------------------

def galcencyl_to_radecDMvlospmradec(R_kpc,phi_rad,z_kpc,
                                    vR_kms,vT_kms, vz_kms,
                                    quiet=True,
                                    Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.,
                                    vXgc_sun_kms=0.,vYgc_sun_kms=230.,vZgc_sun_kms=0.): #position & velocity of Sun in galactocentric (X,Y,Z) frame, i.e. (X,Y,Z)_gc=(8.,0.,0.025)kpc, (vX,vY,vZ)_gc=(-U,V+vcirc,W)~(-10,240,7) km/s

    """
        NAME:
            galcencyl_to_radecDMvlospmradec
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2016-??-?? - Written. - Trick (MPIA)
           2016-12-19 - bovy_coords.galcencyl_to_vxvyvz() requires input of solar position. Corrected the code for that. - Trick (MPIA)
    """

    if float(galpy.__version__) < 1.2:
        sys.exit("Error in galcencyl_to_radecDMvlospmradec(): Make sure to use the newest version of galpy. There have been changes in the coordinate transformations. This is galpy version: "+galpy.__version__)

    if Ygc_sun_kpc != 0.:
        sys.exit("Error in galcencyl_to_radecDMvlospmradec(): Galpy unit conversions do not work for Y_gc_sun != 0.")

    if isinstance(R_kpc,float):
        R_kpc = numpy.array([R_kpc])
        phi_rad = numpy.array([phi_rad])
        z_kpc = numpy.array([z_kpc])
        vR_kms = numpy.array([vR_kms])
        vT_kms = numpy.array([vT_kms])
        vz_kms = numpy.array([vz_kms])
    ndata = len(R_kpc)


    #_____a. covert spatial coordinates to (ra,dec,DM)_____

    if not quiet:
        for ii in range(ndata):
            print ii, "(R,phi,z)_true \t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc,rad,kpc]" % (R_kpc[ii],phi_rad[ii],z_kpc[ii])

    # (R,z,phi) --> (x,y,z):
    xyz = bovy_coords.galcencyl_to_XYZ(
                R_kpc, phi_rad, z_kpc, 
                Xsun=Xgc_sun_kpc, Zsun=Zgc_sun_kpc
                )
    Xs_kpc = xyz[:,0]
    Ys_kpc = xyz[:,1]
    Zs_kpc = xyz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii,"(x,y,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc]" % (Xs_kpc[ii],Ys_kpc[ii],Zs_kpc[ii])

    # (x,y,z) --> (l,b,d):
    lbd = bovy_coords.XYZ_to_lbd(
                Xs_kpc, Ys_kpc, Zs_kpc, 
                degree=False
                )
    l_rad = lbd[:,0]
    b_rad = lbd[:,1]
    d_kpc = lbd[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(l,b,d) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,kpc]" % (l_rad[ii], b_rad[ii], d_kpc[ii])

    # (l,b) --> (ra,dec)
    radec = bovy_coords.lb_to_radec(
                l_rad,b_rad,
                degree=False,epoch=2000.0
                )
    ra_rad  = radec[:,0]
    dec_rad = radec[:,1]
    if not quiet:
        for ii in range(ndata):
            print ii, "(ra,dec) \t\t= "+\
              "(%5.3f, %5.3f) \t\tin [rad]" % (ra_rad[ii],dec_rad[ii])

    # d --> DM (distance modulus):
    DM_mag = 5. * numpy.log10(d_kpc * 1000.) - 5.
    if not quiet:
       for ii in range(ndata):
            print ii, "(DM,d) \t\t= "+\
              "(%5.3f,%5.3f) \t\tin [mag,kpc]" % (DM_mag[ii],d_kpc[ii])

    #_____b. convert velocities to los-velocity and proper motions_____

    if not quiet:
        for ii in range(ndata):
            print ii, "(vR,vT,vz)_true \t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [km/s]" % (vR_kms[ii],vT_kms[ii],vz_kms[ii])

    # (vR,vz,vT) --> (vx,vy,vz):
    vxyz = bovy_coords.galcencyl_to_vxvyvz(
                    vR_kms,vT_kms,vz_kms,
                    phi_rad,
                    Xsun=Xgc_sun_kpc,Zsun=Zgc_sun_kpc,
                    vsun=[vXgc_sun_kms,vYgc_sun_kms,vZgc_sun_kms])
    vx_kms = vxyz[:,0]
    vy_kms = vxyz[:,1]
    vz_kms = vxyz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(vx,vy,vz) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [km/s]" % (vx_kms[ii], vy_kms[ii],vz_kms[ii])

    # (vx,vy,vz) --> (vlos,pm_l,pm_b):
    vrpmlpmb = bovy_coords.vxvyvz_to_vrpmllpmbb(
                    vx_kms,vy_kms,vz_kms,
                    l_rad,b_rad,d_kpc,
                    XYZ=False,degree=False
                    )
    vlos_kms  = vrpmlpmb[:,0]
    pml_masyr = vrpmlpmb[:,1]
    pmb_masyr = vrpmlpmb[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(v_los,pm_l,pm_b) \t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [km/s,mas/yr,mas/yr]" % (vlos_kms[ii], pml_masyr[ii], pmb_masyr[ii])

    # (pm_l,pm_b) --> (pm_ra,pm_dec):
    pmradec = bovy_coords.pmllpmbb_to_pmrapmdec(
                    pml_masyr,pmb_masyr,
                    l_rad,b_rad,
                    degree=False,epoch=2000.0
                    )
    pm_ra_masyr  = pmradec[:,0]
    pm_dec_masyr = pmradec[:,1]
    if not quiet:
        for ii in range(ndata):
            print ii, "(pm_ra,pm_dec) \t= "+\
              "(%5.3f, %5.3f) \t\tin [mas/yr]" % (pm_ra_masyr[ii],pm_dec_masyr[ii])
  
    if ndata == 1:
        return (ra_rad[0],dec_rad[0],DM_mag[0],vlos_kms[0],pm_ra_masyr[0],pm_dec_masyr[0])
    else:
        return (ra_rad,dec_rad,DM_mag,vlos_kms,pm_ra_masyr,pm_dec_masyr)

#-------------------------------------------------------------------------------

def radecDM_to_galcencyl(ra_rad,dec_rad,DM_mag,
                         quiet=True,
                         Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.):

    if float(galpy.__version__) < 1.2:
        sys.exit("Error in radecDM_to_galcencyl(): Make sure to use the newest version of galpy. There have been changes in the coordinate transformations. This is galpy version: "+galpy.__version__)

    if Ygc_sun_kpc != 0.:
        sys.exit("Error in radecDM_to_galcencyl(): Galpy unit conversions do not work for Y_gc_sun != 0.")

    if isinstance(ra_rad,float):
        ra_rad = numpy.array([ra_rad])
        dec_rad = numpy.array([dec_rad])
        DM_mag = numpy.array([DM_mag])
    ndata = len(ra_rad)

    #_____a. convert spatial coordinates (ra,dec,DM) to (R,z,phi)_____"

    if not quiet:
        for ii in range(ndata):
            print ii, "(ra,dec,DM) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,mag]" % (ra_rad[ii],dec_rad[ii],DM_mag[ii])

    #(DM) --> (d):
    d_kpc = 10.**(0.2 * DM_mag + 1.) * 10.**(-3)
    
    #(l,b) --> (ra,dec):
    lb = bovy_coords.radec_to_lb(
                    ra_rad,dec_rad,
                    degree=False,epoch=2000.0
                    )
    l_rad = lb[:,0]
    b_rad = lb[:,1]

    if not quiet:
        for ii in range(ndata):
            print ii, "(l,b,d) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,kpc]" % (l_rad[ii],b_rad[ii],d_kpc[ii])

    # (l,b,d) --> (x,y,z):
    xyz = bovy_coords.lbd_to_XYZ(
                    l_rad,b_rad,
                    d_kpc,
                    degree=False)
    x_kpc = xyz[:,0]
    y_kpc = xyz[:,1]
    z_kpc = xyz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(x,y,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc]" % (x_kpc[ii],y_kpc[ii],z_kpc[ii])

    # (x,y,z) --> (R,z,phi):
    Rzphi = bovy_coords.XYZ_to_galcencyl(
                    x_kpc, y_kpc, z_kpc, 
                    Xsun=Xgc_sun_kpc, Zsun=Zgc_sun_kpc
                    )
    R_kpc   = Rzphi[:,0]
    phi_rad = Rzphi[:,1]
    z_kpc   = Rzphi[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii,"(R,phi,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc,rad,kpc]" % (R_kpc[ii], phi_rad[ii], z_kpc[ii])

    if ndata == 1:
        return (R_kpc[0], phi_rad[0], z_kpc[0])
    else:
        return (R_kpc, phi_rad, z_kpc)

#-------------------------------------------------------------------------------

def radecDMvlospmradec_to_galcencyl(ra_rad,dec_rad,DM_mag,
                                    vlos_kms,pm_ra_masyr,pm_dec_masyr,quiet=True,
                                    Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.,
                                    vXgc_sun_kms=0.,vYgc_sun_kms=230.,vZgc_sun_kms=0.): #position & velocity of Sun in galactocentric (X,Y,Z) frame, i.e. (X,Y,Z)_gc=(8.,0.,0.025)kpc, (vX,vY,vZ)_gc=(-U,V+vcirc,W)~(-10,240,7) km/s

    """
        NAME:
            radecDMvlospmradec_to_galcencyl
        PURPOSE:
        INPUT:
        OUTPUT:
        HISTORY:
           2016-??-?? - Written. - Trick (MPIA)
           2016-12-19 - bovy_coords.vxvyvz_to_galcencyl() requires input of solar position. Corrected the code for that. - Trick (MPIA)
    """

    if float(galpy.__version__) < 1.2:
        sys.exit("Error in radecDMvlospmradec_to_galcencyl(): Make sure to use the newest version of galpy. There have been changes in the coordinate transformations. This is galpy version: "+galpy.__version__)

    if Ygc_sun_kpc != 0.:
        sys.exit("Error in radecDMvlospmradec_to_galcencyl(): Galpy unit conversions do not work for Y_gc_sun != 0.")

    if isinstance(ra_rad,float):
        ra_rad = numpy.array([ra_rad])
        dec_rad = numpy.array([dec_rad])
        DM_mag = numpy.array([DM_mag])
        vlos_kms = numpy.array([vlos_kms])
        pm_ra_masyr = numpy.array([pm_ra_masyr])
        pm_dec_masyr = numpy.array([pm_dec_masyr])  
    ndata = len(ra_rad)

    #_____a. convert spatial coordinates (ra,dec,DM) to (R,z,phi)_____"

    if not quiet:
        for ii in range(ndata):
            print ii, "(ra,dec,DM) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,mag]" % (ra_rad[ii],dec_rad[ii],DM_mag[ii])

    #(DM) --> (d):
    d_kpc = 10.**(0.2 * DM_mag + 1.) * 10.**(-3)
    
    #(ra,dec) --> (l,b):
    lb = bovy_coords.radec_to_lb(
                    ra_rad,dec_rad,
                    degree=False,epoch=2000.0
                    )
    l_rad = lb[:,0]
    b_rad = lb[:,1]

    if not quiet:
        for ii in range(ndata):
            print ii, "(l,b,d) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [rad,rad,kpc]" % (l_rad[ii],b_rad[ii],d_kpc[ii])

    # (l,b,d) --> (x,y,z):
    xyz = bovy_coords.lbd_to_XYZ(
                    l_rad,b_rad,
                    d_kpc,
                    degree=False)
    x_kpc = xyz[:,0]
    y_kpc = xyz[:,1]
    z_kpc = xyz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(x,y,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc]" % (x_kpc[ii],y_kpc[ii],z_kpc[ii])

    # (x,y,z) --> (R,z,phi):
    Rzphi = bovy_coords.XYZ_to_galcencyl(
                    x_kpc, y_kpc, z_kpc, 
                    Xsun=Xgc_sun_kpc, Zsun=Zgc_sun_kpc
                    )
    R_kpc   = Rzphi[:,0]
    phi_rad = Rzphi[:,1]
    z_kpc   = Rzphi[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii,"(R,phi,z) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [kpc,rad,kpc]" % (R_kpc[ii], phi_rad[ii], z_kpc[ii])

    #_____b. convert velocities (pm_ra,pm_dec,vlos) to (vR,vz,vT)_____

    # (pm_ra,pm_dec) --> (pm_l,pm_b):
    pmlpmb = bovy_coords.pmrapmdec_to_pmllpmbb(
                        pm_ra_masyr,
                        pm_dec_masyr,
                        ra_rad,dec_rad,
                        degree=False,epoch=2000.0
                        )
    pml_masyr = pmlpmb[:,0]
    pmb_masyr = pmlpmb[:,1]
    if not quiet:
        for ii in range(ndata):
            print ii, "(v_los,pm_l,pm_b) \t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [mas/yr]" % (vlos_kms[ii],pml_masyr[ii], pmb_masyr[ii])

    # (v_los,pm_l,pm_b) & (l,b,d) --> (vx,vy,vz):
    vxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(
                    vlos_kms,
                    pml_masyr,pmb_masyr,
                    l_rad,b_rad,
                    d_kpc,
                    XYZ=False,degree=False
                    )
    vx_kms = vxvyvz[:,0]
    vy_kms = vxvyvz[:,1]
    vz_kms = vxvyvz[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(vx,vy,vz) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [km/s]" % (vx_kms[ii],vy_kms[ii],vz_kms[ii])

    # (vx,vy,vz) & (x,y,z) --> (vR,vT,vz):
    vRvTvZ = bovy_coords.vxvyvz_to_galcencyl(
                    vx_kms, 
                    vy_kms, 
                    vz_kms, 
                    R_kpc,
                    phi_rad, 
                    z_kpc, 
                    Xsun=Xgc_sun_kpc,Zsun=Zgc_sun_kpc,
                    vsun=[vXgc_sun_kms,vYgc_sun_kms,vZgc_sun_kms], 
                    galcen=True
                    )
    vR_kms = vRvTvZ[:,0]
    vT_kms = vRvTvZ[:,1]
    vz_kms = vRvTvZ[:,2]
    if not quiet:
        for ii in range(ndata):
            print ii, "(vR,vT,vz) \t\t= "+\
              "(%5.3f, %5.3f, %5.3f) \tin [km/s]" % (vR_kms[ii], vT_kms[ii], vz_kms[ii])

    if ndata == 1:
        return (R_kpc[0], phi_rad[0], z_kpc[0], vR_kms[0], vT_kms[0], vz_kms[0])
    else:
        return (R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms)

#------------------------------------------------------------------------

if __name__ == "__main__":

    R_kpc = numpy.array([7.9,9.2])
    z_kpc = numpy.array([-2.,0.1])
    phi_rad = numpy.array([-0.111,0.])
    vR_kms = numpy.array([3.,-0.1])
    vz_kms = numpy.array([10.,-20.])
    vT_kms = numpy.array([240.,200.])
    ra_rad,dec_rad,DM_mag,vlos_kms,pm_ra_masyr,pm_dec_masyr = \
                galcencyl_to_radecDMvlospmradec(R_kpc,phi_rad,z_kpc,
                                    vR_kms,vT_kms,vz_kms,
                                    quiet=False,
                                    Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.025,
                                    vXgc_sun_kms=-11.1,vYgc_sun_kms=12.24+220.,vZgc_sun_kms=7.25)

    R_kpc, phi_rad, z_kpc,vR_kms,vT_kms, vz_kms = \
                radecDMvlospmradec_to_galcencyl(ra_rad,dec_rad,DM_mag,
                                    vlos_kms,pm_ra_masyr,pm_dec_masyr,
                                    quiet=False,
                                    Xgc_sun_kpc=8.,Ygc_sun_kpc=0.,Zgc_sun_kpc=0.025,
                                    vXgc_sun_kms=-11.1,vYgc_sun_kms=12.24+220.,vZgc_sun_kms=7.25)




