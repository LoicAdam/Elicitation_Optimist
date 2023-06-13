# -*- coding: utf-8 -*-
"""This module simulates a decision maker."""

import numpy as np

def get_choice(alternative_1, alternative_2, prob, model):
    """
    The DM is sometimes uncertain, with a parameter modeling the probability 
    to pick the right alternative.
    
    Parameters
    ----------
    alternative_1 : array_like
        An alternative represented by its criteria.
    alternative_2 : array_like
        An alternative represented by its criteria.
    prob : float
        Probability the DM makes a rational choice.
    model : Model
        Aggregation model for the DM preferences.
        
    Returns
    ----------
    dict
        True if first alternative is prefered, False otherwise. Is the DM rational or not.
    """
    res = {}
    score_alt_1 = model.get_model_score(alternative_1)
    score_alt_2 = model.get_model_score(alternative_2)
    incoherence_probability = np.random.uniform()
    rational = incoherence_probability < prob
    if score_alt_1 >= score_alt_2:
        res['accepted'] = rational
    else:
        res['accepted'] = not rational
    res['rational'] = rational
    return res

def get_choice_threshold(alternative_1, alternative_2, confidence, model, threshold = 0.8):
    """
    The DM is always making the good choice if the associated confidence degree
    of an answers is beyond a threshold.
    
    Parameters
    ----------
    alternative_1 : array_like
        An alternative represented by its criteria.
    alternative_2 : array_like
        An alternative represented by its criteria.
    confidence : float
        Fixed confidence of the answer.
    model : Model
        Aggregation model for the DM preferences.
    threshold : float, optional
        Threshold.
        
    Returns
    ----------
    dict
        True if first alternative is prefered, False otherwise. Is the DM rational or not.
    """
    res = {}
    score_alt_1 = model.get_model_score(alternative_1)
    score_alt_2 = model.get_model_score(alternative_2)
    res['rational'] = confidence >= threshold
    if score_alt_1 >= score_alt_2:
        res['accepted'] = confidence >= threshold
    else:
        res['accepted'] = not confidence >= threshold
    return res

def get_choice_fixed(alternative_1, alternative_2, rational, model):
    """
    The DM decision is fixed in advance.
    
    Parameters
    ----------
    alternative_1 : array_like
        An alternative represented by its criteria.
    alternative_2 : array_like
        An alternative represented by its criteria.
    rational : bool
        If the DM makes the right choice or not.
    model : Model
        Aggregation model for the DM preferences.
        
    Returns
    ----------
    dict
        True if first alternative is prefered, False otherwise. Is the DM rational or not.
    """
    res = {}
    score_alt_1 = model.get_model_score(alternative_1)
    score_alt_2 = model.get_model_score(alternative_2)
    res['rational'] = rational
    if score_alt_1 >= score_alt_2:
        res['accepted'] = rational
    else:
        res['accepted'] = not rational
    return res
