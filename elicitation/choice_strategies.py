# -*- coding: utf-8 -*-
"""Strategies to determine the final recommendation."""

import numpy as np

def minimax_regret_choice(alternatives, mr):
    """
    Get the best alternative according to the Minimax regret.

    Parameters
    ----------
    mr : array_like
        MR.

    Returns
    -------
    best_alt : array_like
        The best alternative.
    best_alt_id : integrer
        Its indice.
    regret_best : float
        The regret.

    """
    best_alt_id = np.argmin(mr)
    best_alt = alternatives[best_alt_id]
    regret_best = mr[best_alt_id]
    return best_alt, best_alt_id, regret_best

def maximax_choice(alternatives, max_list):
    """
    Get the best alternative according to the Maximax.

    Parameters
    ----------
    max_list : array_like
        The maximal value for each alternative.

    Returns
    -------
    best_alt : array_like
        The best alternative.
    best_alt_id : integrer
        Its indice.
    max_best : float
        The max max.

    """
    best_alt_id = np.argmax(max_list)
    best_alt = alternatives[best_alt_id]
    max_best = max_list[best_alt_id]
    return best_alt, best_alt_id, max_best

def maximin_choice(alternatives, min_list):
    """
    Get the best alternative according to the Maximin.

    Parameters
    ----------
    min_list : array_like
        The minimal value for each alternative.

    Returns
    -------
    best_alt : array_like
        The best alternative.
    best_alt_id : integrer
        Its indice.
    min_best : float
        The max min.

    """
    best_alt_id = np.argmax(min_list)
    best_alt = alternatives[best_alt_id]
    min_best = min_list[best_alt_id]
    return best_alt, best_alt_id, min_best
