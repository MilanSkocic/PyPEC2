# -*- coding: utf-8 -*-
r""" Modules - Iph Functions

This module contains functions requiered for computing and optimizing the photo-current values from
input parameters i.e. triplets :math:`(K_i, \theta_i, Eg_i)` and experimental data for each semi-conductive phase. 

"""
__author__ = 'M.Skocic'

import os
import platform
import sys
import numpy as np
from scipy.optimize import fmin as nelder_mead
from scipy.stats import linregress, t
from scipy.optimize.slsqp import approx_jacobian

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

# Solve the problem for compilation with PyInstaller
from scipy.sparse.csgraph import _validation

import datetime
import shutil

if sys.version_info >= (3,0):
    xrange = range

FIT_SETTING_EXT = 'FitSet'
PRM_INIT_EXT = 'PrmInit'
PRM_END_EXT = 'PrmEnd'
PRM_MIN_EXT = 'PrmMin'
DATA_MIN_EXT = 'DataMin'
DATA_END_EXT = 'DataEnd'
PRM_ALL_RUN_EXT = 'PrmAll'
DATA_FILE_EXTS = ['dot', 'data']
SUMMARY_END_EXT = 'SumEnd'
SUMMARY_END_Eg_EXT = 'SumEndEg'
SUMMARY_MIN_EXT = 'SumMin'
SUMMARY_MIN_Eg_EXT = 'SumMinEg'

# _ARCHITECTURE, _OS = platform.architecture()
# _FLOAT = np.float32
# _FLOAT_COMPLEX = np.complex64
#if _ARCHITECTURE == '64bit':
#    _FLOAT = np.float64
#    _FLOAT_COMPLEX = np.complex128
_FLOAT = np.float64
_FLOAT_COMPLEX = np.complex128
_EPSILON = np.sqrt(np.finfo(_FLOAT).eps)


def get_header_footer_dot_file(filepath):

    r"""

    Find the number of lines in header and footer in .dot files.
    
    Parameters
    -----------
    filepath: path to the dot file
    
    Returns
    -------
    skip_header: int
        number of lines in header
    
    skip_footer: int
        number of lines in footer
    
    nbpoints: int
        number of data lines
        
    """
    datafile = open(filepath,'r')
    lines = datafile.readlines()
    datafile.close()
    n_0 = int(lines[0])
    n_1 = int(lines[n_0+1])
    N = len(lines)
    k=0
    for i in range(n_0+2+n_1, N):
        if lines[i] not in ['\n','\r\n']:
            k = k+1
    skip_header = n_0+2
    skip_footer = k
    nbpoints = n_1
    return (skip_header,skip_footer, nbpoints)

def get_exp_data(filepath):

    r"""

    Get the data array of data files according to their extension.

    Supported files are .dot files recorded by PECLab software and .data files
    were the first three columns represent :math:`h\nu`, :math:`\left | Iph^{\ast} \right |`, :math:`\theta`.
    
    Parameters
    -----------
    filepath: string
        Path to the data file.

    Returns
    --------
    data_array: 2d array
        Experimental data.
        
    """
    
    name, ext = os.path.basename(os.path.abspath(filepath)).split('.')
    extensions = ['dot', 'data']
    flag_iphB = True
    if ext.lower() in extensions:
        
        if ext.lower() == 'dot':
            skip_header, skip_footer, nbpoints = get_header_footer_dot_file(filepath)
            usecols = (2,3,4,6,9)
            delimiter = ','
            converters = None

        elif ext.lower() == 'data':
            skip_header, skip_footer = 0,0
            usecols = (0,1,2)
            delimiter = '\t'
            converters = None
        
        elif ext.lower() == 'txt':
            skip_header, skip_footer = 0,0
            usecols = (0,1,2)
            delimiter = '\t'
            converters = None
        
        try:
            hv, Iph, IphN, phase, IphB = np.genfromtxt(fname=filepath, dtype=_FLOAT, comments='#', delimiter=delimiter,\
                      skip_header=skip_header, skip_footer=skip_footer,\
                      converters=converters, missing_values=None, filling_values=None,\
                      usecols=usecols, names=None, excludelist=None,\
                      deletechars=None, replace_space='_', autostrip=False, case_sensitive=True,\
                      defaultfmt='f%i', unpack=True, usemask=False, loose=True, invalid_raise=True)
        except ValueError as e:
            flag_iphB = False
            hv, Iph, phase = np.genfromtxt(fname=filepath, dtype=_FLOAT, comments='#', delimiter=delimiter,\
                      skip_header=skip_header, skip_footer=skip_footer,\
                      converters=converters, missing_values=None, filling_values=None,\
                      usecols=usecols, names=None, excludelist=None,\
                      deletechars=None, replace_space='_', autostrip=False, case_sensitive=True,\
                      defaultfmt='f%i', unpack=True, usemask=False, loose=True, invalid_raise=True)
            IphB = np.zeros(shape=hv.shape, dtype=_FLOAT)
            IphB[:] = Iph[:]
            print(Iph[0:5])
            IphN = Iph/np.max(Iph)

        return hv, Iph, IphN, phase, IphB, flag_iphB
    else:
        raise Exception('FileType Error: File was not recognized')

def import_prm_file(filepath):
    r"""

    Import the triplets :math:`(K_i, \theta_i, Eg_i)` from text file where each line represents a
    contributing semi-conductive phase.

    Parameters
    -----------
    filepath: string
        Absolute or relative file path to the text file.

    Returns
    -------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    """

    path = os.path.abspath(filepath)

    #Import all lines from the text file
    try:
        prm = np.loadtxt(path,
                           dtype=_FLOAT,
                           comments='#',
                           converters=None,
                           skiprows=0,
                           ndmin=2,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                           unpack=False)
    except IndexError as error:
        #print(error.message)
        prm = np.loadtxt(path,
                           dtype=_FLOAT,
                           comments='#',
                           converters=None,
                           skiprows=0,
                           ndmin=2,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7),
                           unpack=False)
    
    #Check if zeros lines are zeros lines
    #Zeros lines are ignored
    sum_array = np.sum(prm[:,0:6], axis=1)
    mask, = np.where(sum_array != 0.0)
    nb_SC = mask.size

    prm_array = np.zeros(shape=(nb_SC, 11))
    prm_array[:,8:11] = -1 # Initialize to -1 the errors
    
    # compatibility layer for parameter files that do not have
    # the 3 extra columns for the parameter errors
    row, col = prm.shape
    if col == 11:
        prm_array[:] = prm[mask,:]
    else:
        prm_array[:, 0:8] = prm[mask,:]

    return prm_array




def get_random_prm_values(prm_array,\
                          K_bound=(10**-12, 10**-1),\
                          theta_bound=(-180.0, 180.0),\
                          Eg_bound=(0.1, 6.2),\
                          phase_flag = True):
    r"""

    Generates random values for the triplets :math:`(K_i, \theta_i, Eg_i)` to be fitted based on the states
    given by the `prm_array`.

    By default, the limits are:
    
    * :math:`K_i`: :math:`[10^{-12},10^{-1}]`
    * :math:`\theta _i`: :math:`[-\pi,+\pi]`
    * :math:`Eg_i`: :math:`[0.1, 6.0]`

    Parameters
    ----------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.
        
    K_bound:tuple
        Contains the lower and upper limits for the :math:`K_i` values.

    theta_bound: tuple
        Contains the lower and upper limits for the :math:`\theta _i` values.

    Eg_bound:tuple
        Contains the lower and upper limits for the :math:`Eg_i` values.

    phase_flag: bool
        Indicates if the values of :math:`\theta _i` have to be randomized.

    Returns
    -------
    random_prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    """
    K_low, K_up = K_bound
    log_K_low = np.log10(K_low)
    log_K_up = np.log10(K_up)
    theta_low, theta_up = theta_bound
    Eg_low, Eg_up = Eg_bound
    
    random_prm_array = np.zeros(shape=prm_array.shape, dtype=_FLOAT)
    random_prm_array[:] = prm_array[:]

    if phase_flag == True:
        mask, = np.where(random_prm_array[:,1] == 1)
        random_prm_array[mask, 0] = 10**(log_K_low+np.random.rand(len(mask))*(log_K_up - log_K_low))
    
        mask, = np.where(random_prm_array[:,3] == 1)
        random_prm_array[mask, 2] = theta_low + np.random.rand(len(mask))*(theta_up-theta_low)

        mask, = np.where(random_prm_array[:,5] == 1)
        random_prm_array[mask, 4] = Eg_low + np.random.rand(len(mask))*(Eg_up-Eg_low)
        
    elif phase_flag == False:
        mask, = np.where(random_prm_array[:,1] == 1)
        random_prm_array[mask, 0] = 10**(log_K_low+np.random.rand(len(mask))*(log_K_up - log_K_low))

        mask, = np.where(random_prm_array[:,5] == 1)
        random_prm_array[mask, 4] = Eg_low + np.random.rand(len(mask))*(Eg_up-Eg_low)

    random_prm_array = sort_prm_Eg(random_prm_array)

    return random_prm_array

def sort_prm_Eg(prm_array):
    r"""

    Sort the ``prm_array`` based on values of :math:`Eg_i`.

    Parameters
    -----------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    Returns
    -------
    prm_array: 2d array
        Represents the sorted values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    """
    
    mask = np.argsort(prm_array[:,4])
    prm_array = prm_array[mask]

    return prm_array

def shift_phase(prm_array, theta_bound=(-180.0, 180.0)):
    r"""

    Compute the modulo of :math:`\theta _i` values with :math:`2\pi` and then
    shift the values of :math:`\theta _i` by the amplitude of the boundaries in order
    to be in between the boundaries.

    By default, the boundaries for :math:`\theta _i` are set to :math:`[-\pi,+\pi]`.

    Parameters
    ----------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    theta_bound: tuple
        Contains the lower and upper limits for the :math`theta _i` values.

    Returns
    -------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)` where the
        :math:`\theta _i` values were shifted.

    """

    theta_low, theta_up = theta_bound

    mask, = np.where((prm_array[:,2] > theta_up) | (prm_array[:,2] < theta_low))
    prm_array[mask,2]  = np.mod(prm_array[mask,2], 360)
        
    mask, = np.where(prm_array[:,2] > theta_up)
    prm_array[mask,2] = prm_array[mask,2] - (theta_up-theta_low)
        
    mask, = np.where(prm_array[:,2] < theta_low)
    prm_array[mask,2] = prm_array[mask,2] + (theta_up-theta_low)

    return prm_array


def validate_prm(prm_array,\
                 K_bound=(10**-12, 10**-1),\
                 Eg_bound=(0.1, 6.2)):
    r"""

    Check if the values of :math:`K_i` and :math:`Eg_i` are within the boundaries.

    Parameters
    ----------
    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    K_bound:tuple
        Contains the lower and upper limits for the :math:`K_i` values.

    Eg_bound:tuple
        Contains the lower and upper limits for the :math:`Eg_i` values.

    Returns
    -------
    valid: bool
        Set to True if value of :math:`K_i` or :math:`Eg_i` is out of the boundaries.
    
    """
    K_low, K_up = K_bound
    log_K_low = np.log10(K_low)
    log_K_up = np.log10(K_up)
    Eg_low, Eg_up = Eg_bound

    mask_K, = np.where((prm_array[:,0] > K_up) | (prm_array[:,0] < K_low))
    mask_Eg, = np.where((prm_array[:,4] >  Eg_up) | (prm_array[:,4] < Eg_low))
    
    if (len(mask_K)+len(mask_Eg) == 0):
        valid = True
    else:
        valid = False

    return valid
    

def get_distance(iph_exp_complex, iph_calc_complex):
    r"""

    Compute the distance :math:`D` between :math:`Iph_{exp}` and
    :math:`Iph_{calc}`.
    The distance is computed by multiplying the distances on real and imaginary
    parts of :math:`Iph`:

    .. math::
        
        \Delta Re & = Re \, Iph_{exp} - Re \, Iph_{calc} \\
        \Delta Im & = Im \, Iph_{exp} - Im \, Iph_{calc} \\
        D_{Re} & = \sqrt{\sum{\Delta Re ^2}}\\
        D_{Im} & = \sqrt{\sum{\Delta Im ^2}}\\
        D & = D_{Re} \cdot D_{Im}

    Parameters
    ----------
    iph_exp: 1d numpy array
            Contains the complex values of the :math:`Iph_{exp}`.

    iph_calc: 1d numpy array
            Contains the complex values of the :math:`Iph_{calc}`.

    Returns
    -------
    D: float
            The computed distance on real and imaginary parts of :math:`Iph`:.

    """
    Re_iph_exp = np.real(iph_exp_complex)
    Im_iph_exp = np.imag(iph_exp_complex)

    Re_iph_calc = np.real(iph_calc_complex)
    Im_iph_calc = np.imag(iph_calc_complex)
    
    delta_Re = (Re_iph_exp - Re_iph_calc)
    delta_Im = (Im_iph_exp - Im_iph_calc)

    D_Re = np.sqrt(np.sum(delta_Re**2))
    D_Im = np.sqrt(np.sum(delta_Im**2))
    D = D_Re*D_Im

    return D


def _get_residuals(p, hv, prm_array, iph_exp_complex, phi_N, weights=1.0):

    r"""DocString"""

    row, col = prm_array.shape
    prm_array[:,0:6:2] = p.reshape(row, 3)
    return np.absolute((get_Iph_calc(hv, prm_array, phi_N) - iph_exp_complex)*weights)


def _get_chi2(p, hv, prm_array, iph_exp_complex, phi_N, weights=1.0):

    r"""DocString"""

    return np.sum(_get_residuals(p, hv, prm_array, iph_exp_complex, phi_N, weights)**2)


def get_Iph_calc(hv,  prm_array, phi_N):
    r"""
    Compute the complex values of :math:`Iph` based on the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    .. math::
        Iph^{\ast} = \frac{Iph}{\Phi_{N}}


    Parameters
    ----------
    hv: 1d array
        Vector of energies for which the complex :math:`Iph` has to be computed.

    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    phi_N: 1d array
        Represents the values of the normalized photon flux to the maximum value.
        If nphf is a unity vector, the true photo-current is returned otherwise the 
        as-measured photo-current is returned.

    Returns
    -------
    iph_calc_complex: 1d array
        Vector of the computed complex values of :math:`Iph`.
    """

    nb_SC, cols = prm_array.shape
    
    iph_calc_complex = np.zeros(shape=hv.shape, dtype=np.complex128)

    for k in xrange(nb_SC):
        d_hv = np.where(hv > prm_array[k,4], (hv-prm_array[k,4]), 0.0)
        iph_calc_complex[:] += ((prm_array[k,0]*d_hv)**prm_array[k,6])/(hv**prm_array[k,7])*np.exp(1j*np.deg2rad(prm_array[k,2]))
    
    iph_calc_complex *= phi_N

    return iph_calc_complex


def _jacobian(hv, prm_array):

    nb_SC, cols = prm_array.shape
    Dfun = np.zeros(shape=(hv.size, prm_array[:,0:6:2].flatten()), dtype=np.complex128)
    iph_i = np.zeros(shape=hv.shape, dtype=np.complex128)

    for k in xrange(nb_SC):
        d_hv = np.where(hv > prm_array[k,4], (hv-prm_array[k,4]), 0.0)
        iph_i = ((prm_array[k,0]*d_hv)**prm_array[k,6])/(hv**prm_array[k,7])*np.exp(1j*np.deg2rad(prm_array[k,2]))
        
        Dfun[:,0+k*3] = prm_array[k,6]*iph_i/prm_array[k,0]
        Dfun[:,1+k*3] = 1j * iph_i
        Dfun[:,2+k*3] = np.where(d_hv>0.0, (-1*prm_array[k,6]*iph_i/d_hv), 0)

    return np.absolute(Dfun)


def _random_scan(hv,
                 prm_array,
                 iph_exp_complex,
                 phi_N,
                 weights,
                 loops=1,
                 K_bound=(10**-12, 10**-1),
                 theta_bound=(-180.0, 180.0),
                 Eg_bound=(0.1, 6.2),
                 phase_flag = True):

    r"""

    Performs a random scan of the values of the triplets :math:`(K_i, \theta_i, Eg_i)`. The number of random loops is
    set with the parameter `loops`.

    Parameters
    ----------
    hv: 1d array
        Vector of energies for which the complex :math:`Iph` has to be computed.

    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    iph_exp: 1d numpy array
        Contains the complex values of the experimental :math:`Iph`.

    phi_N: 1d array
        Represents the values of the normalized photon flux to the maximum value.
    
    loops: int
        Number of loops for random scan.

    K_bound:tuple
        Contains the lower and upper limits for the :math:`K_i` values.

    theta_bound: tuple
        Contains the lower and upper limits for the :math:`\theta _i` values.

    Eg_bound:tuple
        Contains the lower and upper limits for the :math:`Eg_i` values.

    phase_flag: bool
        Indicates if the values of :math:`\theta _i` have to be randomized.

    Returns
    -------
    prm_array_random_min: 2d array
        Represents the best values and states of the triplets :math:`(K_i, \theta_i, Eg_i)` after random scan.

    """

    prm_array_random = np.zeros(shape=prm_array.shape, dtype=_FLOAT)
    prm_array_random_min = np.zeros(shape=prm_array.shape, dtype=_FLOAT)

    #1st random scan
    prm_array_random = get_random_prm_values(prm_array,\
                                             K_bound = K_bound,\
                                             theta_bound = theta_bound,\
                                             Eg_bound = Eg_bound,\
                                             phase_flag = True)
    prm_array_random = sort_prm_Eg(prm_array_random)
    iph_calc_complex = get_Iph_calc(hv, prm_array_random, phi_N)

    prm_array_random_min[:] = prm_array_random[:]
    distance = get_distance(iph_exp_complex, iph_calc_complex)
    #distance = _get_chi2(prm_array[:,0:6:2].flatten(), hv, prm_array, iph_exp_complex, phi_N, weights)

    distance_min = distance

    for i in range(loops-1):
        prm_array_random = get_random_prm_values(prm_array_random,\
                                                 K_bound = K_bound,\
                                                 theta_bound = theta_bound,\
                                                 Eg_bound = Eg_bound,\
                                                 phase_flag = True)
        prm_array_random = sort_prm_Eg(prm_array_random)
        iph_calc_complex = get_Iph_calc(hv, prm_array_random, phi_N)
        distance = get_distance(iph_exp_complex, iph_calc_complex)
        #distance = _get_chi2(prm_array[:,0:6:2].flatten(), hv, prm_array, iph_exp_complex, phi_N, weights)

        valid = validate_prm(prm_array_random, K_bound = K_bound , Eg_bound=Eg_bound)
        prm_array_random = shift_phase(prm_array_random, theta_bound=theta_bound)

        if valid == True:
            if distance < distance_min:
                distance_min = distance
                prm_array_random_min[:] = prm_array_random[:]

    return prm_array_random_min


def target_func(p, hv, prm_array, iph_exp_complex, phi_N, weights, Ki_log_flag=True):


    r"""

    Update the triplets :math:`(K_i, \theta_i, Eg_i)` from the flattened parameter
    vector ``p`` sent by the optimization algorithm. The ``prm_array`` will be flattened
    and the indexes of the parameters to be fitted will be updated with ``p`` vector.

    The calculated complex values of :math:`Iph` will be sent along the
    experimental values to the :func:`get_distance` function. The value of the
    distance between the experimental and calculated data will sent back to the
    optimization algorithm.
    

    Parameters
    ----------
    p: 1d array
        Parameter vector sent by the optimization algorithm which is always.
        flattened.

    hv: 1d array
        Vector of energies for which the complex values of :math:`Iph` have to be calculated.

    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    iph_exp_complex: 1d numpy array
        Contains the complex values of the experimental :math:`Iph`.

    phi_N: 1d array
        Represents the values of the normalized photon flux to the maximum value.

    Ki_log_flag: bool
        Indicates if the :math:`K_i` values are in logarithmic space. 
    
    Returns
    -------
    distance: float
        Calculated distance between experimental and calculated data values.
        See the :func:`get_distance` function.

    """
    row, col = prm_array.shape
    prm_values_flatten = prm_array[:,0:6:2].flatten()
    prm_states_flatten = prm_array[:,1:7:2].flatten()
    prm_to_fit_mask, = np.where(prm_states_flatten == 1.0)

    prm_values_flatten[prm_to_fit_mask] = p
    prm_array[:,0:6:2] = prm_values_flatten.reshape(row, 3)

    mask_K_to_fit = np.where(prm_array[:,1] == 1.0)
    if Ki_log_flag == True:
        prm_array[mask_K_to_fit,0] = 10**prm_array[mask_K_to_fit,0]

    iph_calc_complex = get_Iph_calc(hv,  prm_array, phi_N)

    return get_distance(iph_exp_complex, iph_calc_complex)
    #return _get_chi2(prm_array[:,0:6:2].flatten(), hv, prm_array, iph_exp_complex, phi_N, weights)

    

def minimize(hv, iph_exp_complex, phi_N, weights, prm_array,\
             Ki_log_flag=True,\
             maxiter=None, maxfun=None, xtol=1e-11, ftol=1e-23,\
             full_output=True, retall=False, disp=False, callback=None):
    r"""

    Execute the Nelder-Mead algorithm based on parameter values given by
    ``prm_array`` and energy vector :math:`h\nu`.
    
    First, the ``prm_array`` is flattened and the parameters :math:`(K_i, \theta_i,
    Eg_i)` to be fitted are extracted and sent to the :func:`target_function`
    through the Nelder-Mead algorithm. 

    Once the parameters were computed by the Nelder-Mead algorithm, ``the prm_array`` is updated
    with the new values.

    Parameters
    ----------
    hv: 1d numpy array
        Contains the energy vector.
        
    iph_exp: 1d numpy array
            Contains the complex values of the experimental photo-current.

    prm_array: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    Ki_log_flag: bool
        Indicates if the :math:`K_i` values are in logarithmic space.

    maxiter : int, optional
        Maximum number of iterations to perform.

    maxfun : number, optional
        Maximum number of function evaluations to make.

    xtol : float, optional
        Relative error in xopt acceptable for convergence.
        
    ftol : number, optional
        Relative error in func(xopt) acceptable for convergence.

    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.

    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        
    disp : bool, optional
        Set to True to print convergence messages.

    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    prm_array: 2d array
        Represents the updated values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.
        
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
        
    """
    mask_K_to_fit = np.where(prm_array[:,1] == 1.0)
    if Ki_log_flag == True:
            prm_array[mask_K_to_fit,0] = np.log10(prm_array[mask_K_to_fit,0])

    prm_values_flatten = prm_array[:,0:6:2].flatten()
    prm_states_flatten = prm_array[:,1:7:2].flatten()
    prm_to_fit_mask, = np.where(prm_states_flatten == 1.0)
    p0 = prm_values_flatten[prm_to_fit_mask]
    
    
    popt ,fopt ,iteration ,funcalls ,warnflag = nelder_mead(target_func,\
                                                              p0 ,\
                                                              args=(hv,\
                                                                    prm_array,\
                                                                    iph_exp_complex,\
                                                                    phi_N,
                                                                    weights,
                                                                    Ki_log_flag),\
                                                              maxiter = maxiter,\
                                                              maxfun = maxfun,\
                                                              xtol = xtol,\
                                                              ftol = ftol,\
                                                              full_output=full_output,\
                                                              retall=retall,\
                                                              disp=disp,\
                                                              callback=callback)

    prm_values_flatten[prm_to_fit_mask] = popt
    row, col = prm_array.shape
    prm_array[:,0:6:2] = prm_values_flatten.reshape(row,3)

    if Ki_log_flag == True:
        prm_array[mask_K_to_fit, 0] = 10**prm_array[mask_K_to_fit,0]

    prm_array = sort_prm_Eg(prm_array)

    return (prm_array, fopt)


def _callback_fit(run, nb_run, fit, nb_minimization,\
              distance, valid,\
              LCC_results, prm_array, additional_messages=[]):

    os.system('cls' if os.name == 'nt' else 'clear')

    sys.stdout.write('***** Run = %d/%d ***** \n'  % (run+1, nb_run))
    sys.stdout.write('Minimizing ...\n')
    sys.stdout.write('Fit {0:03d}/{4:03d}-log10(D)={1:+09.4f}-Valid={2:b}-LCC={5:.6f},{6:.6f},{7:.6f},{8:.6f}\n'.format(fit+1, np.log10(distance), valid, run, nb_minimization,LCC_results[0], LCC_results[1],LCC_results[2],LCC_results[3]))
    sys.stdout.write(str(prm_array[:,0:6:2])+'\n')
    for i in additional_messages:
        sys.stdout.write(i+'\n')
    sys.stdout.flush()


def run_fit(nb_run, nb_fit, init_types, hv_limits, datafilepath, prmfilepath,\
            root = './', alloy='Unspecified_Alloy', alloy_id='Unspecified_ID',\
            fit_type='Iph*', weights_type='1', average_error=1.0, K_bound = (10**-12, 10**-1), theta_bound = (-180.0, 180.0), Eg_bound=(0.0, 6.2),\
            Ki_log_flag=True, random_loops=200, console_output=False,\
            maxiter=None, maxfun=None, xtol=1e-11, ftol=1e-23,\
            full_output=True, retall=False, disp=False, fmin_callback=None, process_id=1, callback=None):

    r"""

    Run ``nb_run`` runs of ``nb_fit`` calls to the function :func:`minimize`.

    Return two lists containing the minimal and end values f the triplets :math:`(K_i, \theta_i, Eg_i)` for each run.

    Parameters
    ----------
    nb_run: int
        Number of run to be performed.

    nb_fit: int
        Number of calls of the function :func:`minimize`.

    init_types: tuple
        Initialization procedure of the triplets :math:`(K_i, \theta_i, Eg_i)` for each run.
        The tuple contains 3 types of initializations  (First run , Propagation, validation).
        
    datafilepath: str
       Filepath of the experimental data stored in .dot or .data files.

    prmdatafilepath: str
        Filepath of the parameter file stored in .PrmInit, .PrmMin, .PrmEnd files.

    fit_type: str, optional
        Flag for choosing if the fit is performed on the as-measured photo-current (:math:`Iph`) or
        the true photo-current (:math:`Iph^{\ast}`).

    K_bound:tuple, optional
        Contains the lower and upper limits for the :math:`K_i` values.

    theta_bound: tuple, optional
        Contains the lower and upper limits for the :math:`\theta _i` values.

    Eg_bound:tuple, optional
        Contains the lower and upper limits for the :math:`Eg_i` values.

    Ki_log_flag: bool, optional
        Indicates if the :math:`K_i` values are in logarithmic space.

    random_loops: int, optional
        Number of random combinations to test before starting the minimization.

    console_output: bool, optional
        Indicates if the console ouput must be displayed for each call of the function :func:`minimize`

    maxiter : int, optional
        Maximum number of iterations to perform.

    maxfun : number, optional
        Maximum number of function evaluations to make.

    xtol : float, optional
        Relative error in xopt acceptable for convergence.
        
    ftol : number, optional
        Relative error in func(xopt) acceptable for convergence.

    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.

    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        
    disp : bool, optional
        Set to True to print convergence messages.

    fmin_callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    callback: callable, optional
        Called after each minimization, as
        callback(run, nb_run_per_process, fit, nb_minimization, distance, valid, LCC_results, prm_end_run, additional_messages=[]).
        

    Returns
    -------
    prm_min_list: list
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)` that gave the minimal distance for each run.
        
    prm_end_list : list
        Represents the last values and states of the triplets :math:`(K_i, \theta_i, Eg_i)` for each run.
        
    """
    #import data
    datafilepath = os.path.abspath(datafilepath)
    datafilename, ext = os.path.basename(datafilepath).split('.')
    hv, iph, iphN, theta, iphB, flag_iphB = get_exp_data(datafilepath)
    iph_true_exp_complex = iph*np.exp(1j*np.deg2rad(theta))
    iph_trueN_exp_complex = iphN*np.exp(1j*np.deg2rad(theta))

    phi_N = np.ones(shape=hv.shape, dtype=_FLOAT)
    w = np.ones(shape=(hv.shape), dtype=_FLOAT)

    hv_start, hv_end = hv_limits
    mask, = np.where( (hv>=hv_start) & (hv<=hv_end) )
    n = mask.size

    if flag_iphB is False:
        fit_type = 'Iph*'

    if fit_type == 'Iph*':
        iph_exp_complex = iph*np.exp(1j*np.deg2rad(theta))
    elif fit_type == 'Iph':
        iph_exp_complex = iphB*np.exp(1j*np.deg2rad(theta))
        phi_N = iphB/iph
    else:
        iph_exp_complex = iph*np.exp(1j*np.deg2rad(theta))

    if weights_type == u'1/|Iph(*)|^2':
        w = 1.0/iph_exp_complex
    elif weight_type == u'Abs.Err.':
        if average_error == 0.0:
            average_error = 1.0
        new_max = np.max(iph[mask])
        new_iph = iph/new_max
        w = new_iph*1.0/average_error

    hv_start, hv_end = hv_limits
    mask, = np.where( (hv>=hv_start) & (hv<=hv_end) )
    n = mask.size

    #import parameters
    prmfilepath = os.path.abspath(prmfilepath)
    user_prm = import_prm_file(prmfilepath)
    nb_SC, col = user_prm.shape
    mask_to_fit, = np.where(user_prm[:,1:7:2].flatten() == 1)
    nb_params = mask_to_fit.size

    #initialize the parameter that will be used for fitting: prm_array
    prm_array = np.zeros(shape=user_prm.shape, dtype=_FLOAT)
    prm_array[:] = user_prm[:]
    
    prm_min_run = np.zeros(shape=prm_array.shape, dtype=_FLOAT)
    prm_end_run = np.zeros(shape=prm_array.shape, dtype=_FLOAT)

    prm_min_run[:] = prm_array[:]
    prm_end_run[:] = prm_array[:]

    init_type_0, init_type_validation, init_type_N = init_types

    alloy = alloy.replace(' ','_')
    alloy_id = alloy_id.replace(' ','_')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')

    if len(alloy) == 0:
        alloy = 'Unspecified_Alloy'
    if len(alloy_id) == 0:
        alloy_id = 'Unspecified_ID'

    
    suffix = ''
    for i in range(nb_SC):
        bin_str = ''.join(prm_array[i, 1:7:2].astype(np.int32).astype(np.str).tolist())
        suffix = suffix + str(int(bin_str,2))

    alloy_folder = root + '/' + alloy + '-' + alloy_id
    fit_folder = alloy_folder + '/' + timestamp + '-' + alloy + '-' + alloy_id + '-' + suffix + '-' +\
                          init_type_0[0].capitalize() + init_type_validation[0].capitalize() + init_type_N[0].capitalize()+\
                          '-' + '{0:.2f}eV_{1:.2f}eV'.format(hv_start, hv_end)
                          
    if not os.path.exists(alloy_folder):
        os.mkdir(alloy_folder)       
    os.mkdir(fit_folder)

    filepath = fit_folder + '/' + datafilename + '.' + ext
    shutil.copy(datafilepath, filepath)

    filepath = fit_folder + '/' + datafilename + '-' + suffix+ '.' + PRM_INIT_EXT
    np.savetxt(filepath, user_prm, fmt=['%+.6e', '%d']*4+['%+.1e']*3, delimiter='\t', newline='\n')

    header_minimization_elements = ['Nb of Run',\
              'Valid',\
              'np.log10(D)',\
              'LCC Module',\
              'LCC Phase',\
              'LCC Re',\
              'LCC Im',\
              'slope Module',\
              'slope Phase',\
              'slope Re',\
              'slope Im',\
              'intercept Module',\
              'intercept Phase',\
              'intercept Re',\
              'intercept Im']
    header_minimization_results = '\t'.join(header_minimization_elements) + '\t'
    for i in range(nb_SC):
        header_minimization_results += 'K_'+str(i+1) + '\t'
        header_minimization_results += 'Phi_'+str(i+1) + '\t'
        header_minimization_results += 'Eg_'+str(i+1) + '\t'

    col = len(header_minimization_elements) + 3*nb_SC
    minimization_results = np.zeros(shape=(nb_fit, col),dtype=_FLOAT)

    Text=''
    #Text = Text+'FIT SETTINGS\n\n'
    Text = Text + 'Experimental Data File=' + datafilepath + '\n'
    Text = Text + 'Energy Range (eV)={0:s},{1:s}\n'.format(str(hv_start), str(hv_end))
    Text = Text + 'Fit type={0:s}\n'.format(fit_type)
    Text = Text + 'Weights={0:s}\n'.format(weights_type)

    Text=Text + 'No of SC Contributions='+str(nb_SC) + '\n'
    Text=Text + 'Parameter File=' + prmfilepath + '\n'
    
    Text=Text + 'No of Processes=' + str(1) + '\n'
    Text=Text + 'No of Runs per Process=' + str(nb_run) + '\n'
    Text=Text + 'No of Runs=' + str(nb_run) + '\n'
    Text=Text + 'No of Fits per Run=' + str(nb_fit) + '\n'
    Text=Text + 'Suffix='+suffix + '\n'

    Text = Text + 'Initialization Run 1={0:s} \n'.format(init_type_0)
    Text = Text + 'Random Loops={0:d} \n'.format(random_loops)
    Text = Text + 'Propagation Run > 1={0:s} \n'.format(init_type_N)
    Text = Text + 'Initialize after non valid parameters={0:s} \n'.format(init_type_validation)

    Text = Text + 'log10 xtol={0:.0f} \n'.format(np.log10(xtol))
    Text = Text + 'log10 ftol={0:.0f} \n'.format(np.log10(ftol))
    Text = Text + 'Iterations/Parameter={0:s} \n'.format(str(maxiter))
    Text = Text + 'fcalls/Parameter={0:s} \n'.format(str(maxfun))

    Text = Text + 'K min, K max={0:.2e},{1:.2e} \n'.format(*K_bound)
    Text = Text + 'Log scan of K={0:b} \n'.format(bool(Ki_log_flag))
    Text = Text + 'theta min, theta max={0:.0f},{1:.0f} \n'.format(*theta_bound)
    Text = Text + 'Eg min, Eg max={0:.2f},{1:.2f} \n'.format(*Eg_bound)
    Text=Text+'Result Folder=' + os.path.abspath(fit_folder)

    filepath = fit_folder + '/' + datafilename + '-' + suffix + '.' + FIT_SETTING_EXT
    fobj = open(filepath,'w')
    fobj.write(Text)
    fobj.close()

                                                  

    for run in range(nb_run):

        if run == 0:

            if init_type_0 == 'random':
                prm_array = _random_scan(hv=hv[mask],\
                                         prm_array=prm_array,\
                                         iph_exp_complex=iph_exp_complex[mask],\
                                         phi_N=phi_N[mask],
                                         weights=w[mask],
                                         loops=random_loops,\
                                         K_bound=K_bound,\
                                         theta_bound=theta_bound,\
                                         Eg_bound=Eg_bound,\
                                         phase_flag = True)
            elif init_type_0 == 'user':
                prm_array[:] = user_prm[:]
        else:
            if init_type_N == 'random':
                prm_array = get_random_prm_values(prm_array, K_bound=K_bound, theta_bound=theta_bound, Eg_bound=Eg_bound, phase_flag=True)
            elif init_type_N == 'user':
                prm_array[:] = user_prm[:]
            elif init_type_N == 'min':
                prm_array[:] = prm_min_run[:]
            elif init_type_N == 'end':
                prm_array[:] = prm_end_run[:]
            
        prm_array = sort_prm_Eg(prm_array)
        iph_calc_complex = get_Iph_calc(hv, prm_array, phi_N)
        distance = get_distance(iph_exp_complex[mask], iph_calc_complex[mask])
        #distance = _get_chi2(prm_array[:,0:6:2].flatten(), hv[mask], prm_array, iph_exp_complex[mask], phi_N[mask], w[mask])
        distance_min_run = distance
        distance_end_run = distance

        for fit in range(nb_fit):
            
            prm_array, distance = minimize(hv=hv[mask],
                                           iph_exp_complex=iph_exp_complex[mask],
                                           phi_N=phi_N[mask],
                                           weights=w[mask],
                                           prm_array=prm_array,
                                           Ki_log_flag=Ki_log_flag,
                                           maxiter=maxiter*nb_params, maxfun=maxfun*nb_params, xtol=xtol, ftol=ftol,
                                           full_output=full_output, retall=retall, disp=disp, callback=fmin_callback)
        
            iph_calc_complex = get_Iph_calc(hv,prm_array, phi_N)
            LCC_results = get_LCC(iph_exp_complex[mask], iph_calc_complex[mask])

            valid = validate_prm(prm_array, K_bound=K_bound, Eg_bound=Eg_bound)
            prm_array = shift_phase(prm_array, theta_bound=theta_bound)

            minimization_results[fit] = np.hstack((fit+1, int(valid), np.log10(distance), LCC_results, prm_array[:,0:6:2].flatten()))

            prm_end_run[:] = prm_array[:]
            distance_end_run = distance


            if callback is not None:
                 callback(run, nb_run, fit, nb_fit,\
                             distance, valid,\
                             LCC_results, prm_array) 

            else:
                if callback is not None:
                    callback(run, nb_run, fit, nb_fit,\
                             distance, valid,\
                             LCC_results, prm_array) 
            if valid is False:
                if init_type_validation == 'random':
                    prm_array = get_random_prm_values(prm_array, K_bound=K_bound, theta_bound=theta_bound, Eg_bound=Eg_bound, phase_flag=False)
                elif init_type_validation == 'user':
                    prm_array[:] = user_prm[:]

            elif valid:
                if distance_min_run > distance:
                    distance_min_run = distance
                    prm_min_run[:] = prm_array[:]


        if callback is not None:
            callback(run, nb_run, fit, nb_fit,\
                     distance, valid,\
                     LCC_results, prm_end_run, ['Computing Covariance Matrix...'])
        
        args = (hv[mask], prm_end_run, iph_exp_complex[mask], phi_N[mask], w[mask])
        prm_end_run[:,8:11] = _get_prm_error(_get_residuals, _EPSILON, *args)
        args = (hv[mask], prm_min_run, iph_exp_complex[mask], phi_N[mask], w[mask])
        prm_min_run[:,8:11] = _get_prm_error(_get_residuals, _EPSILON, *args)

        if callback is not None:
            callback(run, nb_run, fit, nb_fit,\
                     distance, valid,\
                     LCC_results, prm_end_run, ['Saving Results...'])

        
        save_results(run, process_id,\
                     fit_folder, datafilepath, suffix, hv, mask, iph_true_exp_complex, phi_N,\
                     prm_min_run, prm_end_run, distance_min_run, distance_end_run,\
                     minimization_results, header_minimization_results)

    if callback is not None:
        callback(run, nb_run, fit, nb_fit,\
                 distance, valid,\
                 LCC_results, prm_end_run, ['Computing Summary...'])

    get_summary(fit_folder)
    plot_summary(fit_folder)


def _round_errors(errors):

    r"""DocString"""
    log_errors = np.log10(errors)
    log_errors = np.floor(log_errors)
    errors = np.ceil(errors*10**(-log_errors))*10**log_errors

    return errors


def _get_prm_error(func, epsilon, *args):

    row, col = args[1].shape
    p = args[1][:,0:6:2].flatten()
    
    n = args[0].size
    nb_param = p.size
    dof = n-nb_param-1
    tvp = t.isf(0.05/2.0, dof)

    try:
        jac = approx_jacobian(p, func, epsilon, *args)
        cov = np.dual.inv(np.dot(jac.T, jac))
        g = _get_chi2(p, *args)/dof
        dp = _round_errors(np.sqrt(cov.diagonal()*g)*tvp)
    except np.linalg.LinAlgError as error:
        print(error.message)
        dp = np.ones(shape=p.shape, dtype=_FLOAT)*-1.0

    return dp.reshape(row, 3)    
            


def get_LCC(iph_exp_complex, iph_calc_complex):
    r"""


    """

    mod_exp = np.absolute(iph_exp_complex)
    phase_exp = np.angle(iph_exp_complex, deg=True)
    Re_exp = np.real(iph_exp_complex)
    Im_exp = np.real(iph_exp_complex)

    mod_calc = np.absolute(iph_calc_complex)
    phase_calc = np.angle(iph_calc_complex, deg=True)
    Re_calc = np.real(iph_calc_complex)
    Im_calc = np.real(iph_calc_complex)

    slope_mod, intercept_mod, r_mod, p_mod, std_mod = linregress(mod_exp, mod_calc)
    slope_phase, intercept_phase, r_phase, p_phase, std_phase = linregress(phase_exp, phase_calc)
    slope_Re, intercept_Re, r_Re, p_Re, std_Re = linregress(Re_exp, Re_calc)
    slope_Im, intercept_Im, r_Im, p_Im, std_Im = linregress(Im_exp, Im_calc)

    return (r_mod, r_phase, r_Re, r_Im,\
            slope_mod, slope_phase, slope_Re, slope_Im,\
            intercept_mod, intercept_phase, intercept_Re, intercept_Im)

def get_results_array(hv, iph_exp_complex, iph_calc_complex):

    r"""

    Build the data array of the experimental and calculated data: :math:`h\nu`,
    :math:`\left| Iph_{exp} \right|`, :math:`\theta _{exp}`, :math:`\left| Iph_{calc} \right|` and :math:`\theta _{calc}`

            

    
    Parameters
    ----------
    hv: 1d numpy array
        Contains the energy vector.

    iph_exp: 1d array
            Contains the complex values of :math:`Iph_{exp}`.

    iph_calc: 1d array
            Contains the complex values of :math:`Iph_{calc}`.

    Returns
    -------
    data_array: 2d array
            Array containing the .

    """
    header = '\t'.join([r'hv /eV',\
                        r'|Iph_exp| /A', r'Phase_exp /deg', r'Re Iph_exp /A' , r'Im Iph_exp /A',\
                        r'|Iph_calc| /A', r'Phase_calc /deg', r'Re Iph_calc /A' , r'Im Iph_calc /A',\
                        r'Res |Iph| /A', r'Res Phase /deg', r'Res Re Iph /A', r'Res Im Iph /A'])
    
    mod_iph_exp = np.absolute(iph_exp_complex)	
    phase_iph_exp = np.angle(iph_exp_complex, deg=True)
    Re_iph_exp = np.real(iph_exp_complex)
    Im_iph_exp = np.imag(iph_exp_complex)


    mod_iph_calc = np.absolute(iph_calc_complex)	
    phase_iph_calc = np.angle(iph_calc_complex, deg=True)
    Re_iph_calc = np.real(iph_calc_complex)
    Im_iph_calc = np.imag(iph_calc_complex)

    Res_mod = mod_iph_exp - mod_iph_calc
    Res_phase = phase_iph_exp - phase_iph_calc
    Res_Re = Re_iph_exp - Re_iph_calc
    Res_Im = Im_iph_exp - Im_iph_calc
    
    data_array = np.transpose(np.vstack((hv,\
                                         mod_iph_exp, phase_iph_exp, Re_iph_exp, Im_iph_exp,\
                                         mod_iph_calc, phase_iph_calc, Re_iph_calc, Im_iph_calc,\
                                         Res_mod, Res_phase, Res_Re, Res_Im)))
    
    return (header, data_array)


def save_results(run, process_id, fit_folder, datafilepath, suffix, hv, mask, iph_exp_complex, phi_N,\
                  prm_min_run, prm_end_run, distance_min_run, distance_end_run,\
                  minimization_results, header_minimization_results):


    name, ext = os.path.basename(datafilepath).split('.')
    nb_SC, col = prm_end_run.shape

    #save minimization results
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '.' + PRM_ALL_RUN_EXT
    filepath = fit_folder + '/' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '.' + PRM_ALL_RUN_EXT
    np.savetxt(filepath, minimization_results,\
               fmt=['%d','%d','%+.4f'] + ['%+.16e']*12 +['%+.16e']*3*nb_SC,\
               delimiter='\t',\
               newline='\n',\
               header=header_minimization_results)

    #save minimum
    distance_str = '{0:.4f}'.format(np.log10(distance_min_run))
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + PRM_MIN_EXT
    filepath = fit_folder + '/' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + PRM_MIN_EXT
    np.savetxt(filepath, prm_min_run, fmt=['%+.16e', '%d']*3 + ['%d']*2 +['%+.1e']*3 ,delimiter='\t' ,newline='\n')

    iph_calc_complex = get_Iph_calc(hv, prm_min_run, phi_N)
    header, data = get_results_array(hv[mask], iph_exp_complex[mask], iph_calc_complex[mask])
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + DATA_MIN_EXT
    filepath = fit_folder + '/' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + DATA_MIN_EXT
    np.savetxt(filepath, data, fmt='%+.16e' ,delimiter='\t' ,newline='\n' ,header=header)

    ext='pdf'
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id) + '-' + str(run+1) + '-' + 'min' + '-' + 'd' + distance_str + '.' + ext
    filepath = fit_folder + '/' + suffix + '-' + str(process_id) + '-' + str(run+1) + '-' + 'min' + '-' + 'd' + distance_str + '.' + ext
    save_pdf(filepath,\
             hv, iph_exp_complex, iph_calc_complex,\
             mask, minimization_results)


    #save end
    distance_str = '{0:.4f}'.format(np.log10(distance_end_run))
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + PRM_END_EXT
    filepath = fit_folder + '/' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + PRM_END_EXT
    np.savetxt(filepath, prm_end_run, fmt=['%+.16e', '%d']*3 + ['%d']*2 +['%+.1e']*3,delimiter='\t' ,newline='\n')

    iph_calc_complex = get_Iph_calc(hv, prm_end_run, phi_N)
    header, data = get_results_array(hv[mask], iph_exp_complex[mask], iph_calc_complex[mask])
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + DATA_END_EXT
    filepath = fit_folder + '/' + suffix + '-' + str(process_id)+ '-' + str(run+1) + '-d' + distance_str + '.' + DATA_END_EXT
    np.savetxt(filepath, data, fmt='%+.16e' ,delimiter='\t' ,newline='\n' ,header=header)

    ext='pdf'
    #filepath = fit_folder + '/' + name + '-' + suffix + '-' + str(process_id) + '-' + str(run+1) + '-' + 'End' + '-' + 'd' + distance_str + '.' + ext
    filepath = fit_folder + '/' + suffix + '-' + str(process_id) + '-' + str(run+1) + '-' + 'End' + '-' + 'd' + distance_str + '.' + ext
    save_pdf(filepath,\
             hv, iph_exp_complex, iph_calc_complex,\
             mask, minimization_results)


def save_pdf(filepath,\
             hv, iph_exp_complex, iph_calc_complex,\
             mask, all_results):

        
        pdf = PdfPages(filepath)
        scilimits = (1e-6,1e6)

        plt.figure(figsize=(8,6))
        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.ticklabel_format(scilimits=scilimits)
        plt.plot(hv[mask], np.abs(iph_exp_complex[mask]),'bo' , markersize=4, markeredgewidth=1, mfc='w', mec='b', label='exp')
        plt.plot(hv[mask], np.abs(iph_calc_complex[mask]),'r' , linewidth=1, label='fit')
        plt.title(r'Iph vs $h\nu$')
        plt.xlabel(r'$h\nu$ /eV')
        plt.ylabel('Iph /A')
        pdf.savefig()
        plt.close()


        plt.figure(figsize=(8,6))
        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.ticklabel_format(scilimits=scilimits)
        plt.plot(hv[mask], np.angle(iph_exp_complex[mask], deg=True),'bo' , markersize=4, markeredgewidth=1, mfc='w', mec='b', label='exp')
        plt.plot(hv[mask], np.angle(iph_calc_complex[mask], deg=True),'r' , linewidth=1, label='fit')
        plt.title(r'$\theta$ vs $h\nu$')
        plt.xlabel(r'$h\nu$ /eV')
        plt.ylabel(r'$\theta$ /$^{\circ}$')
        pdf.savefig()
        plt.close()


        plt.figure()
        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.ticklabel_format(scilimits=scilimits)
        plt.plot(hv[mask], np.real(iph_exp_complex[mask]),'bo',label='Re(Iph) exp' ,markersize=4, markeredgewidth=1, mfc='w', mec='b')
        plt.plot(hv[mask], np.imag(iph_exp_complex[mask]),'go',label='Im(Iph) exp' ,markersize=4, markeredgewidth=1, mfc='w', mec='g')
        plt.plot(hv[mask], np.real(iph_calc_complex[mask]),'r',label='Re(Iph) fit' ,linewidth=1)
        plt.plot(hv[mask], np.imag(iph_calc_complex[mask]),'r',label='Im(Iph) fit' ,linewidth=1)
        plt.title(r'Re(Iph) and Im(Iph) vs $h\nu$')
        plt.xlabel(r'$h\nu$ /eV')
        plt.ylabel('Iph /A')
        plt.legend(loc='upper left')
        pdf.savefig()
        plt.close()


        plt.figure(figsize=(8,6))
        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.ticklabel_format(scilimits=scilimits)
        plt.plot(hv, np.abs(iph_exp_complex), 'bo' ,markersize=4, markeredgewidth=1, mfc='w', mec='b', label='exp')
        plt.plot(hv, np.abs(iph_calc_complex), 'r' ,linewidth=1, label='fit')
        plt.title(r'Iph vs $h\nu$')
        plt.xlabel(r'$h\nu$ /eV')
        plt.ylabel('Iph /A')
        pdf.savefig()
        plt.close()


        plt.figure(figsize=(8,6))
        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.ticklabel_format(scilimits=scilimits)
        mask_, = np.where(all_results[:,0] != 0)
        mask_valid = np.where(all_results[mask_,1] == 1)
        plt.plot(all_results[mask_,0], all_results[mask_,2], color='k', marker='o', linestyle='-', linewidth=1, mfc='w', mec='k', markeredgewidth=1)
        plt.plot(all_results[mask_valid,0], all_results[mask_valid,2], color='g', marker='o', linestyle='None', linewidth=1, mfc='w', mec='g', markeredgewidth=1)
        plt.title('log10(D) vs no fit')
        plt.xlabel('No of minimization')
        plt.ylabel('log(D)')
        pdf.savefig()
        plt.close()
        pdf.close()


def get_summary(fit_folder):
    r"""

    List result files for the triplets :math:`(K_i, \theta_i, Eg_i)` at the end and the minimum of each run.

    Compute the distance, the LCCs for the energy interval that was used for minimizing the the triplets :math:`(K_i, \theta_i, Eg_i)`.

    The results are saved in 4 files: .SumEnd, .SumEndEg, *.SumMin, *.SumMinEg.


    Parameters
    -----------
    fit_folder: string
        Path of the fit folder.


    """
    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)
    PRM_end = []
    PRM_min = []
    fitsettings_filepath = ''

    run = 0
    for i in listfiles:
        ext = i.split('.')[-1]
        if ext == PRM_END_EXT:
            run+=1
            PRM_end.append(os.path.abspath(dirpath + '/' + i))
        elif ext == PRM_MIN_EXT:
            PRM_min.append(os.path.abspath(dirpath + '/' + i))
        elif ext in DATA_FILE_EXTS:
            datafilepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    fitsettings_fobj = open(fitsettings_filepath,'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n','')
        
    nb_run_per_process = int(fitsettings_dict['No of Runs per Process'])
    fit_type = fitsettings_dict['Fit type']
    hv_start, hv_end = fitsettings_dict['Energy Range (eV)'].split(',')
    hv_start, hv_end = float(hv_start), float(hv_end)

    K_min, K_max = fitsettings_dict['K min, K max'].split(',')
    K_bound = (float(K_min), float(K_max))
    Eg_min, Eg_max = fitsettings_dict['Eg min, Eg max'].split(',')
    Eg_bound = (float(Eg_min), float(Eg_max))

    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    suffix = fitsettings_dict['Suffix']

    hv, iph, iphN, phase, iphB, flag_iphB = get_exp_data(datafilepath)
    mask, = np.where( (hv >= hv_start) & (hv <= hv_end) )

    if fit_type == 'Iph*':
        iph_exp_comp = iph*np.exp(1j*phase*np.pi/180.0)
        phi_N = np.ones(shape=hv.shape, dtype=np.float64)
        weights = np.ones(shape=hv.shape, dtype=np.float64)

    elif fit_type == 'Iph':
        iph_exp_comp = iphB*np.exp(1j*phase*np.pi/180.0)
        phi_N = iphB/iph
        weights = 1/iphB

    prm_filepath = os.path.abspath(PRM_end[0])
    prm_i = import_prm_file(prm_filepath)
    nb_SC, col = prm_i.shape

    header_end_elements = ['Nb of Run',\
                  'Valid',\
                  'log10(D)',\
                  'LCC Module',\
                  'LCC Phase',\
                  'LCC Re',\
                  'LCC Im',\
                  'slope Module',\
                  'slope Phase',\
                  'slope Re',\
                  'slope Im',\
                  'intercept Module',\
                  'intercept Phase',\
                  'intercept Re',\
                  'intercept Im']
    header_end = '\t'.join(header_end_elements) + '\t'
    for i in range(nb_SC):
        header_end = header_end + 'K_'+str(i+1) + '\t'
        header_end = header_end + 'Phi_'+str(i+1) + '\t'
        header_end = header_end + 'Eg_'+str(i+1) + '\t'

    header_end_dtypes = np.dtype({'names':header_end.split('\t')[:-1],
                                  'formats':[(np.str_, 128)]+[_FLOAT]*(len(header_end.split('\t')[:-1])-1)})

    header_end_Eg_elements = ['Nb of Run',\
                  'Valid',\
                  'log10(D)',\
                  'LCC Module',\
                  'LCC Phase',\
                  'LCC Re',\
                  'LCC Im']
    header_end_Eg = '\t'.join(header_end_Eg_elements) + '\t'
    for i in range(nb_SC):
        header_end_Eg = header_end_Eg + 'Eg_'+str(i+1) + '\t'

    header_end_Eg_dtypes = np.dtype({'names':header_end_Eg.split('\t')[:-1],
                                  'formats':[(np.str_, 128)]+[_FLOAT]*(len(header_end.split('\t')[:-1])-1)})

    col = len(header_end_elements) + 3*nb_SC
    col_Eg = len(header_end_Eg_elements) + nb_SC
    
    summary_end = np.zeros(shape = (run,), dtype=header_end_dtypes)
    summary_Eg_end = np.zeros(shape = (run,), dtype=header_end_Eg_dtypes)
    summary_min = np.zeros(shape = (run,), dtype=header_end_dtypes)
    summary_Eg_min = np.zeros(shape = (run,), dtype=header_end_Eg_dtypes)

    
    for ind,i in enumerate(PRM_end):
        run_i, minimization_i = os.path.basename(i).split('-d')[0].split('-')[-2:]

        prm = import_prm_file(os.path.abspath(i))
        valid = validate_prm(prm, K_bound, Eg_bound)
        
        iph_calc_comp = get_Iph_calc(hv, prm, phi_N)
        distance = np.log10(get_distance(iph_exp_comp[mask], iph_calc_comp[mask]))
        #distance = np.log10(_get_chi2(prm[:,0:6:2].flatten(), hv[mask], prm, iph_exp_comp[mask], phi_N[mask],
        #weights[mask] ))

        LCC_results = get_LCC(iph_exp_comp[mask], iph_calc_comp[mask])

        summary_end[ind] = (run_i+'-'+minimization_i, valid,
        distance)+tuple(LCC_results)+tuple(prm[:,0:6:2].flatten().tolist())
        summary_Eg_end[ind] = (run_i+'-'+minimization_i, valid,
        distance)+tuple(LCC_results[0:4])+tuple(prm[:,4].flatten().tolist())
 
    #sort over the log10 (D)
    mask_end = np.argsort(summary_end['log10(D)'])
    mask_Eg_end = np.argsort(summary_Eg_end['log10(D)'])
    
    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '.' + SUMMARY_END_EXT)
    filepath = os.path.abspath(result_folder + '/' + suffix + '.' + SUMMARY_END_EXT)
    np.savetxt(filepath, X=summary_end[mask_end], fmt=['%s','%d','%+.4f'] + ['%+.16e']*(col-3), delimiter='\t', header=header_end)
    
    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '.' + SUMMARY_END_Eg_EXT)
    filepath = os.path.abspath(result_folder + '/' + suffix + '.' + SUMMARY_END_Eg_EXT)
    np.savetxt(filepath, X= summary_Eg_end[mask_Eg_end], fmt=['%s','%d','%+.4f'] + ['%+.16e']*(col_Eg-3), delimiter='\t', header=header_end_Eg)


    for ind,i in enumerate(PRM_min):

        run_i, minimization_i = os.path.basename(i).split('-d')[0].split('-')[-2:]
       
        prm = import_prm_file(os.path.abspath(i))
        valid = validate_prm(prm, K_bound, Eg_bound)
        
        iph_calc_comp = get_Iph_calc(hv, prm, phi_N)
        distance = np.log10(get_distance(iph_exp_comp[mask],iph_calc_comp[mask]))
        #distance = np.log10(_get_chi2(prm[:,0:6:2].flatten(), hv[mask], prm, iph_exp_comp[mask], phi_N[mask],
        #weights[mask] ))

        LCC_results = get_LCC(iph_exp_comp[mask], iph_calc_comp[mask])

        summary_min[ind] = (run_i+'-'+minimization_i, valid,
        distance)+tuple(LCC_results)+tuple(prm[:,0:6:2].flatten().tolist())
        summary_Eg_min[ind] = (run_i+'-'+minimization_i, valid,
        distance)+tuple(LCC_results[0:4])+tuple(prm[:,4].flatten().tolist())

    #sort over the log10 (D)
    mask_min = np.argsort(summary_min['log10(D)'])
    mask_Eg_min = np.argsort(summary_Eg_min['log10(D)'])

    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '.' + SUMMARY_MIN_EXT)
    filepath = os.path.abspath(result_folder + '/' + suffix + '.' + SUMMARY_MIN_EXT)
    np.savetxt(filepath, X=summary_min[mask_min], fmt=['%s','%d','%+.4f'] + ['%+.16e']*(col-3), delimiter='\t', header=header_end)
    
    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '.' + SUMMARY_MIN_Eg_EXT)
    filepath = os.path.abspath(result_folder + '/' + suffix + '.' + SUMMARY_MIN_Eg_EXT)
    np.savetxt(filepath, X= summary_Eg_min[mask_Eg_min], fmt=['%s','%d','%+.4f'] + ['%+.16e']*(col_Eg-3), delimiter='\t', header=header_end_Eg)



def plot_summary(fit_folder):
    r"""

    Plot the result files that were created by the :func:`get_summary`
    for he triplets :math:`(K_i, \theta_i, Eg_i)` at the end and the minimum of each run.

    The results are saved in 2 files: -0-End.pdf, -0-Min.pdf.


    Parameters
    -----------
    fit_folder: string
        Path of the fit folder.


    """

    dirpath = os.path.abspath(fit_folder)
    listfiles = os.listdir(dirpath)
    description_filepath = ''

    summary_end_filepath = ''
    summary_min_filepath = ''

    for i in listfiles:
        cuts = i.split('.')
        ext = cuts[-1]
        if ext == SUMMARY_END_EXT:
            summary_end_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == SUMMARY_MIN_EXT:
            summary_min_filepath = os.path.abspath(dirpath + '/' + i)
        elif ext == FIT_SETTING_EXT:
            fitsettings_filepath = os.path.abspath(dirpath + '/' + i)

    #read the number of SC contributions
    fitsettings_fobj = open(fitsettings_filepath,'r')
    fitsettings_lines = fitsettings_fobj.readlines()
    fitsettings_fobj.close()
    fitsettings_dict = {}
    for line in fitsettings_lines:
        key, value = line.split('=')
        fitsettings_dict[key] = value.replace('\n','')
        
    nb_SC = int(fitsettings_dict['No of SC Contributions'])
    datafilepath = fitsettings_dict['Experimental Data File']
    basename, ext = os.path.basename(datafilepath).split('.')
    result_folder = os.path.abspath(fitsettings_dict['Result Folder'])
    suffix = fitsettings_dict['Suffix']

    #read header
    with open(summary_end_filepath,'r') as f:
        header_end = f.readline().replace('# ','').replace('\n','').split('\t')[:-1]

    Eg_start_col = header_end.index('Eg_1')
    K_start_col = header_end.index('K_1')
    Phi_start_col = header_end.index('Phi_1')

    col = len(header_end)


    summary_end = np.loadtxt(summary_end_filepath, comments='#',\
                             delimiter='\t',\
                             skiprows=0,\
                             unpack=False,
                             dtype=np.dtype({'names':header_end,
                             'formats':[(np.str_,128)]+[_FLOAT]*(col-1)}),
                             ndmin=1)

    summary_min = np.loadtxt(summary_min_filepath, comments='#',\
                             delimiter='\t',\
                             skiprows=0,\
                             unpack=False,
                             dtype=np.dtype({'names':header_end,
                             'formats':[(np.str_,128)]+[_FLOAT]*(col-1)}),
                             ndmin=1)
    
    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '-' + '0' + '-' + 'End' + '.' + 'pdf')
    filepath = os.path.abspath(result_folder + '/' + suffix + '-' + '0' + '-' + 'End' + '.' + 'pdf')
    pdf_end = PdfPages(filepath)
    #filepath = os.path.abspath(result_folder + '/' + basename + '-' + suffix + '-' + '0' + '-' + 'Min' + '.' + 'pdf')
    filepath = os.path.abspath(result_folder + '/' + suffix + '-' + '0' + '-' + 'Min' + '.' + 'pdf')
    pdf_min = PdfPages(filepath)   

    colors = ['b', 'r', 'g', 'y', 'm', 'c', 'darkblue', 'darkred', 'darkgreen', 'lightblue', 'orange', 'lightgreen','k', 'gray']
    #Eg vs No Runs for END prm values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('$Eg_i$ vs No Run')
    ax.set_xlabel('No Run')
    ax.set_ylabel(r'$Eg_i$ /eV')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    Egi = summary_end[header_end[Eg_start_col:Eg_start_col+3*nb_SC+1:3]]
    row = summary_end.size
    no_run = range(0, row+1)
    for i in range(nb_SC):
        indc = int(i - np.floor(i*1.0/len(colors))*len(colors))
        ax.plot(no_run[1:], Egi['Eg_'+str(i+1)], color=colors[indc], marker='o', mfc='w', mec=colors[indc], ls='-', lw=1, ms=4, mew=1 )
    ax.set_xticks(no_run)
    if summary_end['Nb of Run'].size == 1:
        labels = [summary_end['Nb of Run'].tolist()]
    else:
        labels = summary_end['Nb of Run'].tolist()
    labels.insert(0, '')    
    ax.set_xticklabels(labels, fontsize=6, rotation=45)
    pdf_end.savefig(fig)
    plt.close(fig)
    
    if summary_end.size > 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Distribution of $Eg_i$')
        ax.set_xlabel(r'$Eg_i$ /eV')
        ax.set_ylabel(u'% of runs ({0:d} runs)'.format(int(no_run[-1])))
        ax.set_ylim(0, no_run[-1])
        ax.set_yticklabels(np.round(ax.get_yticks()/float(no_run[-1])*100.0,0))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        for i in range(nb_SC):
            indc = int(i - np.floor(i*1.0/len(colors))*len(colors))
            ax.hist(Egi['Eg_'+str(i+1)], bins=25, color=colors[indc])
        pdf_end.savefig(fig)
        plt.close(fig)

    
    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    K = summary_end[header_end[K_start_col:K_start_col+3*nb_SC+1:3]].view(_FLOAT)
    phase = summary_end[header_end[Phi_start_col:Phi_start_col+3*nb_SC+1:3]].view(_FLOAT)
    phase = np.deg2rad(phase)
    ax.set_title(r'$K_i$ vs $\theta _i$')
    #ax.scatter(phase, K, c=phase, s = 100)
    ax = scatter_logpolar(ax, phase, K, c=phase, s = 100)
    pdf_end.savefig(fig)
    plt.close(fig)
    
    pdf_end.close()


    #Eg vs No Runs for MIN prm values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('$Eg_i$ vs No Run')
    ax.set_xlabel('No Run')
    ax.set_ylabel(r'$Eg_i$ /eV')
    Egi = summary_min[header_end[Eg_start_col:Eg_start_col+3*nb_SC+1:3]]
    row = summary_min.size
    no_run = range(0, row+1)
    for i in range(nb_SC):
        indc = int(i - np.floor(i*1.0/len(colors))*len(colors))
        ax.plot(no_run[1:], Egi['Eg_'+str(i+1)], color=colors[indc], marker='o', mfc='w', mec=colors[indc], ls='-', lw=1, ms=4, mew=1 )
    ax.set_xticks(no_run)
    if summary_end['Nb of Run'].size == 1:
        labels = [summary_min['Nb of Run'].tolist()]
    else:
        labels = summary_min['Nb of Run'].tolist()
    labels.insert(0, '')    
    ax.set_xticklabels(labels, fontsize=6, rotation=45)
    pdf_min.savefig(fig)
    plt.close(fig)
    
    if summary_min.size > 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Distribution of $Eg_i$')
        ax.set_xlabel(r'$Eg_i$ /eV')
        ax.set_ylabel(u'% of runs ({0:d} runs)'.format(int(no_run[-1])))
        ax.set_ylim(0,no_run[-1])
        ax.set_yticklabels(np.round(ax.get_yticks()/no_run[-1]*100.0, 0))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        for i in range(nb_SC):
            indc = int(i - np.floor(i*1.0/len(colors))*len(colors))
            ax.hist(Egi['Eg_'+str(i+1)], bins=25, color=colors[indc])
        pdf_min.savefig(fig)
        plt.close(fig)


    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    ax.set_title(r'$K_i$ vs $\theta _i$')
    K = summary_min[header_end[K_start_col:K_start_col+3*nb_SC+1:3]].view(_FLOAT)
    phase = summary_min[header_end[Phi_start_col:Phi_start_col+3*nb_SC+1:3]].view(_FLOAT)
    phase = np.deg2rad(phase)
    #ax.scatter(phase, K, c=phase, s = 100)
    ax = scatter_logpolar(ax, phase, K, c=phase, s = 100)
    pdf_min.savefig(fig)
    plt.close(fig)


    pdf_min.close()


def scatter_logpolar(ax, theta, r_, ticks=5, bullseye=0.0, **kwargs):
    min10 = np.floor(np.log10(np.min(r_)))
    max10 = np.ceil(np.log10(np.max(r_)))
    r = np.log10(r_) - min10 + bullseye
    ax.scatter(theta, r, **kwargs)
    l = np.linspace(min10, max10, ticks)
    ax.set_rticks(l - min10 + bullseye) 
    ax.set_yticklabels(["$10^{" + '%.2f' % x +'}$' for x in l])
    ax.set_rlim(0, max10 - min10 + bullseye)
    return ax


if __name__ == '__main__':


    print('TEST 1')

    root = 'C:/Users/mskocic/Desktop/Zy2-S8/'
    folder = '2015_12_18-093519-Zy2-S8-777-RRR-2.13eV_5.64eV/'
    get_summary(root+folder)
    plot_summary(root+folder)

    




