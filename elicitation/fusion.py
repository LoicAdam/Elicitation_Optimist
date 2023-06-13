# -*- coding: utf-8 -*-
"""This module manipulated T-norms and T-conorms."""

import numpy as np

def tnorm(information, fusion_rule = "product"):
    """T-norm of pieces of information.

    Parameters
    ----------
    information : float
        Pieces of information to fuse.
    fusion_rule : array_like
        The T-norm used for merging information.
        
    Returns
    -------
    float
        The T-norm of the pieces.
        
    Raises
    ------
    NotImplementedError
        If the rule given during initialisation is not known.
    """
    if np.min(information) == 0: #If one zero: stop.
        return 0
    if fusion_rule == 'minimum':
        return np.min(information)
    if fusion_rule == 'product':
        return np.product(information)
    if fusion_rule == 'lukasiewicz':
        luk = 1
        for info in information:
            luk = np.maximum(0, luk + info - 1)
        return luk
    raise NotImplementedError(fusion_rule, 'is an unknown rule.')

def tconorm(information, fusion_rule = "probabilistic"):
    """T-Conorm of pieces of information.

    Parameters
    ----------
    information : float
        Pieces of information to fuse.
    fusion_rule : array_like
        The T-conorm used for merging information.
        
    Returns
    -------
    float
        The T-Conorm of the pieces.
        
    Raises
    ------
    NotImplementedError
        If the rule given during initialisation is not known.
    """
    if np.max(information) == 1:  #If one one: stop.
        return 1
    if fusion_rule == 'maximum':
        return np.max(information)
    if fusion_rule == 'probabilistic':
        prob = 0
        for info in information:
            prob = prob + info - prob * info
        return prob
    if fusion_rule == 'bounded':
        bound = 0
        for info in information:
            bound = np.minimum(1, bound + info)
        return bound
    raise NotImplementedError(fusion_rule, 'is an unknown rule.')
    