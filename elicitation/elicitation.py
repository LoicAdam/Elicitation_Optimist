# -*- coding: utf-8 -*-
"""Elicitation"""

import time
import numpy as np
from alternatives.data_preparation import get_pareto_efficient_alternatives
from elicitation.question_strategies import CSSQuestionStrategy
from elicitation.dm import get_choice_fixed
from elicitation.choice_calculation import pmr_polytope, mr_polytope, min_polytope, max_polytope
from elicitation.choice_strategies import minimax_regret_choice, maximax_choice, maximin_choice
from elicitation.polytope import Polytope, construct_constrainst

def robust_elicitation(alternatives, model, max_iter = -1,
                       rational = None, regret_limit = 10**-8,
                       strategy = "mMr"):
    """
    Robust elicitation classic with CSS.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    regret_limit : float, optional
        If a regret limit. The default is 10**-8.
    strategy : string, optional
        If minimax (mMr) or maximax (MM) or maximin(Mm). The default is mMr.

    Returns
    -------
    dict
        Elicitation information.

    """

    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)

    constraints = model.get_model_constrainsts()
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)

    question_strategy = CSSQuestionStrategy(alternatives)

    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs)
    else:
        max_iter = number_pairs

    rational_list = np.zeros(max_iter)
    if rational is None:
        rational = np.ones(max_iter)

    scores = model.get_model_score(alternatives)
    scores_ordered_idx = np.argsort(scores)[::-1]

    memr_estimated_list = np.zeros(max_iter)
    mmr_real_list = np.zeros(max_iter)
    score_list = np.zeros(max_iter)
    pos_list = np.zeros(max_iter)

    ite = 0
    start_time = time.time()
    
    pmr = pmr_polytope(alternatives, first_polytope, model)
    if strategy == "mMr":
        mr =  mr_polytope(pmr)
    elif strategy == "MM":
        extrema = max_polytope(alternatives, first_polytope, model)
    elif strategy == "Mm":
        extrema = min_polytope(alternatives, first_polytope, model)
    else:
        raise ValueError("Didn't expect that decision rule")
    
    while ite < max_iter:
        if strategy == "mMr":
            candidate_alt, candidate_alt_id = question_strategy.give_candidate(mr)
        else:
            candidate_alt, candidate_alt_id = question_strategy.give_candidate_optimist(extrema)
            
        worst_alt, _ = question_strategy.give_oponent(pmr, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        rational_list[ite] = choice['rational']

        if strategy == "mMr":
            _, best_alt_id, regret = minimax_regret_choice(alternatives, mr)
        elif strategy == "MM":
            _, best_alt_id, regret = maximax_choice(alternatives, extrema)
        elif strategy == "Mm":
            _, best_alt_id, regret = maximin_choice(alternatives, extrema)
            
        score = scores[best_alt_id]
        pos = np.where(scores_ordered_idx == best_alt_id)[0]
        
        mmr_real_list[ite] = np.max(scores) - score
        memr_estimated_list[ite] = regret
        score_list[ite] = score
        pos_list[ite] =  pos

        if np.max(regret) <= regret_limit :# (ite != 0 and best_emr > memr_estimated_list[ite-1]) or :
            break

        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
        first_polytope.add_answer(new_constraint_a, new_constraint_b, 1, "minimum")

        pmr = pmr_polytope(alternatives, first_polytope, model)
        if strategy == "mMr":
            mr =  mr_polytope(pmr)
        elif strategy == "MM":
            extrema = max_polytope(alternatives, first_polytope, model)
        elif strategy == "Mm":
            extrema = min_polytope(alternatives, first_polytope, model)

        ite = ite+1

    memr_estimated_list = memr_estimated_list[0:ite+1]
    mmr_real_list = mmr_real_list[0:ite+1]
    score_list = score_list[0:ite+1]
    pos_list = pos_list[0:ite+1]

    d = {}
    d['time'] = time.time() - start_time
    d['best_alternative'] = best_alt_id
    d['memr_estimated'] = memr_estimated_list
    d['mmr_real'] = mmr_real_list
    d['score'] = score_list
    d['pos'] = pos_list
    d['ite'] = ite
    return d
