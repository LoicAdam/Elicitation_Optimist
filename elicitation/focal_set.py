# -*- coding: utf-8 -*-
"""This module gives tools for focal sets (compute empr notably)."""

from copy import deepcopy
import numpy as np

def _compute_levels(possibility_list):
    """
    Determine the levels (for the focal sets)

    Parameters
    ----------
    possibility_list : list
        Possibility for each polytope.

    Returns
    -------
    levels : array_like
        Levels.

    """
    ind_sort = np.asarray(possibility_list).argsort()
    sorted_possibility_list = np.asarray(possibility_list)[ind_sort[::-1]]
    levels = np.unique(sorted_possibility_list)
    if np.min(levels) != 0:
        levels = np.append(0,levels)
    levels = np.sort(levels)[::-1]
    return levels

### Minmax regret ###

def compute_epmr_emr(pmr_list, possibility_list, inconsistency_type = 'ignorance'):
    """
    Compute the EMPR and EMR

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'ignorance'.

    Returns
    -------
    epmr : float
        epmr.
    emr : float
        emr.

    """
    new_pmr_list, mr_list, new_possibility_list = _update_pmr_mr(pmr_list, possibility_list, inconsistency_type)
    levels = _compute_levels(new_possibility_list)
    epmr = _epmr_compute(new_pmr_list, new_possibility_list, levels)
    emr = _emr_compute(mr_list, new_possibility_list, levels)
    return epmr, emr

def _update_pmr_mr(pmr_list, possibility_list, inconsistency_type = 'ignorance'):
    """
    Update the PMR and possibility list to handle uncertainty.

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'ignorance'.

    Returns
    -------
    new_pmr_list : list
        Updated pmr list.
    mr_list : list
        mr list.
    new_possibility_list : list
        Updated possibility list.

    """
    new_pmr_list = deepcopy(pmr_list)
    new_possibility_list = deepcopy(possibility_list)
    if np.max(possibility_list) != 1:
        new_possibility_list.append(1)
        if inconsistency_type == 'ignorance':
            new_pmr_list.append(np.max(pmr_list, axis = 0))
        elif inconsistency_type == 'zero':
            #Equivalent to Guillot min model max(0, regret)
            new_pmr_list.append(np.zeros((new_pmr_list[0].shape[0], new_pmr_list[0].shape[0])))
        else:
            raise NotImplementedError(inconsistency_type, 'is an unknown rule.')
    mr_list = np.max(new_pmr_list, axis = 2)
    return new_pmr_list, mr_list, new_possibility_list

def _epmr_compute(pmr_list, possibility_list, levels):
    """
    Compute the EMPR

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.

    Returns
    -------
    float
        epmr.

    """
    new_pmr_list = np.zeros((len(levels) - 1, pmr_list[0].shape[0],pmr_list[0].shape[0]))

    for i in range(len(levels) - 1):
        idx_in_focal_sets = np.where(possibility_list >= levels[i])[0]
        new_pmr_list[i] = np.max(np.asarray([pmr_list[j] for j in idx_in_focal_sets]), axis = 0)
    res = np.sum(new_pmr_list * (levels[0:-1] - levels[1:])[:,None,None], axis = 0)
    return res

def _emr_compute(mr_list, possibility_list, levels):
    """
    Compute the EMR

    Parameters
    ----------
    mr_list : list
        MR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.
        
    Returns
    -------
    float
        emr.

    """

    new_mr_list = np.zeros((len(levels) - 1, mr_list[0].shape[0]))

    for i in range(len(levels) - 1):
        idx_in_focal_sets = np.where(possibility_list >= levels[i])[0]
        new_mr_list[i] = np.max(np.asarray([mr_list[j] for j in idx_in_focal_sets]), axis = 0)
    res = np.sum(new_mr_list * (levels[0:-1] - levels[1:])[:,None], axis = 0)
    return res

### Maximax or Maximin ###

def compute_emax_emin(max_list, possibility_list, criterion = 'maximax',
                      inconsistency_type = 'ignorance'):
    """
    Compute the emax (or emin)

    Parameters
    ----------
    max_list : list
        Max (or min) for each polytope.
    possibility_list : list
        Possibility for each polytope.
    criterion : string, optional
        Maximax or Maximin. The default is 'maximax'.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'ignorance'.

    Returns
    -------
    emax : float
        emax (or emin).
    """
    new_max_list, new_possibility_list = _update_max_min(max_list, possibility_list,
                                                         criterion, inconsistency_type)
    levels = _compute_levels(new_possibility_list)
    emax = _emax_emin_compute(new_max_list, new_possibility_list, levels, criterion)
    return emax

def _update_max_min(max_list, possibility_list, criterion = 'maximax',
                    inconsistency_type = 'ignorance'):
    """
    Update the max (or emin) and possibility list to handle uncertainty.

    Parameters
    ----------
    max_list : list
        Max (or min) for each polytope.
    possibility_list : list
        Possibility for each polytope.
    criterion : string, optional
        Maximax or Maximin. The default is 'maximax'.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'ignorance'.

    Returns
    -------
    new_pmr_list : list
        Updated max (or min) list.
    new_possibility_list : list
        Updated possibility list.

    """
    new_max_list = deepcopy(max_list)
    new_possibility_list = deepcopy(possibility_list)
    if np.max(possibility_list) != 1:
        new_possibility_list.append(1)
        if inconsistency_type == 'ignorance':
            if criterion == 'maximax':
                new_max_list.append(np.max(max_list, axis = 0))
            elif criterion == 'maximin':
                new_max_list.append(np.min(max_list, axis = 0))
            else:
                raise NotImplementedError("I didn't do that")
        elif inconsistency_type == 'zero':
            new_max_list.append(np.zeros(len(max_list[0])))
        else:
            raise NotImplementedError(inconsistency_type, 'is an unknown rule.')
    return new_max_list, new_possibility_list

def _emax_emin_compute(max_list, possibility_list, levels, criterion = 'maximax'):
    """
    Compute the emax (or emin)

    Parameters
    ----------
    max_list : list
        max (or min) for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.
    criterion : string, optional
        Maximax or Maximin. The default is 'maximax'.

    Returns
    -------
    float
        emax (or emin).

    """

    new_max_list = np.zeros((len(levels) - 1, max_list[0].shape[0]))

    for i in range(len(levels) - 1):
        idx_in_focal_sets = np.where(possibility_list >= levels[i])[0]
        if criterion == 'maximax':
            new_max_list[i] = np.max(np.asarray([max_list[j] for j in idx_in_focal_sets]), axis = 0)
        elif criterion == "maximin":
            new_max_list[i] = np.min(np.asarray([max_list[j] for j in idx_in_focal_sets]), axis = 0)
    res = np.sum(new_max_list * (levels[0:-1] - levels[1:])[:,None], axis = 0)
    return res
