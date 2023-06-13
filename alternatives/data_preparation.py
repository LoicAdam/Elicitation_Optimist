# -*- coding: utf-8 -*-
"""This module generates multi-criteria alternatives."""

import numpy as np

def generate_alternatives_score(nb_alternatives, nb_parameters, value, delta = 0.05, multiplicator = 100):
    """Generate alternatives according to an uniform distribution with parameters having a common sum value.

    Parameters
    ----------
    nb_alternatives : integer
        Number of alternatives you would like to have. Can return less.
    nb_parameters : integer
        Number of paremeters for each alternative.
    value : float
        The sum of parameters each alternative should have.    
    delta : float
        Relaxation of the value: each alternative has a value between [score - delta, score + delta].    
    multiplicator : integer
        Number of alternatives generated: nb_alternatives*multiplicator.     
        
    Returns
    -------
    array_like
        Pereto efficient alternatives.
    """
    alternatives = np.empty((0, nb_parameters))
    new_alternatives = np.random.uniform(0, 1, (multiplicator * nb_alternatives, nb_parameters))
    total_score_alternatives = np.sum(new_alternatives, axis = 1)
    new_alternatives = new_alternatives[(total_score_alternatives >= value - delta) & (total_score_alternatives <= value + delta)]
    new_alternatives_pareto = get_pareto_efficient_alternatives(new_alternatives)
    new_alternatives_pareto = new_alternatives_pareto[0:np.minimum(nb_alternatives,new_alternatives_pareto.shape[0]), :]
    alternatives = np.vstack((alternatives, new_alternatives_pareto))
    return alternatives

def generate_alternatives_random(nb_alternatives, nb_parameters, multiplicator = 100):
    """Generate alternatives according to an uniform distribution.

    Parameters
    ----------
    nb_alternatives : integer
        Number of alternatives you would like to have. Can return less.
    nb_parameters : integer
        Number of paremeters for each alternative.    
    multiplicator : integer
        Number of alternatives generated: nb_alternatives*multiplicator.    
        
    Returns
    -------
    array_like
        Pereto efficient alternatives.
    """
    alternatives = np.empty((0, nb_parameters))
    new_alternatives = np.random.uniform(0, 1, (multiplicator * nb_alternatives, nb_parameters))
    new_alternatives_pareto = get_pareto_efficient_alternatives(new_alternatives)
    new_alternatives_pareto = new_alternatives_pareto[0:np.minimum(nb_alternatives,new_alternatives_pareto.shape[0]), :]
    alternatives = np.vstack((alternatives, new_alternatives_pareto))
    return alternatives

def get_pareto_efficient_alternatives(alternatives):
    """Keep only pareto efficient alternatives among a set of alternatives.

    Parameters
    ----------
    alternatives : array_like
        Number of rows alternatives and number of columns criteria.
        
    Returns
    -------
    array_like
        Pereto efficient alternatives.
    """
    is_efficient = np.ones(alternatives.shape[0], dtype = bool)
    for i, nb_c in enumerate(alternatives):
        if is_efficient[i]:
            worse_alternatives = np.all(alternatives <= nb_c, axis=1)
            worse_alternatives[i] = False
            is_efficient = is_efficient & ~worse_alternatives
    return alternatives[is_efficient]
