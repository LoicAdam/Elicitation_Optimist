# -*- coding: utf-8 -*-
"""This module contains all the info of a poly."""

from copy import deepcopy
import numpy as np
from scipy.optimize import linprog
from elicitation.fusion import tnorm

class Polytope:
    """
    Represent an elementary polytope, a division of the model space, delimited
    by linear constrainsts.
    """

    def __init__(self, constraints_A_ub, constraints_b_ub,
                 constraints_A_eq, constraints_b_eq,
                 bounds):
        """
        Parameters
        ----------
        constraints_A_ub : array_like
            2-D array of values representing A for the constrainst Ax <= b.
        constraints_b_ub : array_like
            1-D array of values representing b for the constrainst Ax <= b.
        constraints_A_eq : array_like
            2-D array of values representing A for the constrainst Ax = b.
        constraints_b_eq : array_like
            1-D array of values representing b for the constrainst Ax = b.
        bounds : sequence
            Minimum and maximum values for each parameters of the model space.
        """
        self._answers = []
        self._possibility = 1
        self._constraints_A_ub = constraints_A_ub
        self._constraints_b_ub = constraints_b_ub
        self._constraints_A_eq = constraints_A_eq
        self._constraints_b_eq = constraints_b_eq
        self._bounds = bounds

    def add_answer(self, constraint_A, constraint_b, confidence, tnorm_rule = 'minimum'):
        """
        Add a new answer properly

        Parameters
        ----------
        constraint_A : array_like
            A (Ax < b).
        constraint_b : array_like
            b (Ax < b).
        confidence : float
            Certainty degree.
        fusion_rule : string, optional
            The T-norm to apply. The default is 'product'.

        Returns
        -------
        None.

        """
        if self._constraints_A_ub is None:
            self._constraints_A_ub = constraint_A
            self._constraints_b_ub = constraint_b
        else:
            self._constraints_A_ub = np.vstack((self._constraints_A_ub, constraint_A))
            self._constraints_b_ub = np.vstack((self._constraints_b_ub, constraint_b))
        self._answers.append(confidence)
        self._possibility = tnorm([self._possibility,confidence],tnorm_rule)

    def delete_answer(self, answer_id, fusion_rule = 'minimum'):
        """
        Removes an answer and update the possibility.

        Parameters
        ----------
        answer_id : integer
            Answer to remove.
        fusion_rule : string, optional
            The T-norm to apply. The default is 'minimum'.

        Returns
        -------
        None.

        """
        self._answers = np.delete(self._answers, answer_id, axis=0)
        possibility = self._answers[0]
        for i in range(1, len(self._answers)):
            possibility = tnorm([possibility, self._answers[i]], fusion_rule)
        self._possibility = possibility

    def get_constrainsts(self):
        """
        Get the constrainsts.
        """
        return self._constraints_A_ub, self._constraints_b_ub, self._constraints_A_eq, self._constraints_b_eq

    def get_bounds(self):
        """
        Get the bounds.
        """
        return self._bounds

    def get_possibility(self):
        """
        Get the possibility.
        """
        return self._possibility

    def get_answers(self):
        """
        Get the confidences of the answers related with the polytope.
        """
        return self._answers

def construct_constrainst(alt_1, alt_2, alt_1_prefered, model):
    """Construct a constrainst according to Current Solution Strategy.

    Parameters
    ----------
    alt_1 : array_like
        1-D array containing the values of the "best" alternative.
    alt_2 : array_like
        1-D array containing the values of the "worst" alternative.
    alt_1_prefered : bool
        True if the DM preferred the "best" alternative, False otherwise.
    model : Model
            The aggregation model.
        
    Returns
    -------
    array_like
        1-D array of values representing A for the constrainst Ax <= b.
    float            
        Value representing b for the constrainst Ax <= b.
    """
    diff_alt = model.get_diff(alt_1, alt_2, alt_1_prefered)
    new_constrainst_a = -diff_alt
    new_constraints_b = [0]
    if new_constrainst_a.ndim == 1:
        new_constrainst_a = new_constrainst_a[np.newaxis,:]
    return np.asarray(new_constrainst_a), np.asarray(new_constraints_b)

def is_polytope_not_empty(A_ub, b_ub, A_eq, b_eq, bounds):
    """
    Check if a polytope is not empty

    Parameters
    ----------
    A_ub : array_like
        A_ub.
    b_ub : array_like
        b_ub.
    A_eq : array_like
        A_eq.
    b_eq : array_like
        b_eq.
    bounds : tuple
        bounds.

    Returns
    -------
    bool
        If it is empty.

    """
    if A_ub.ndim == 1:
        A_ub = A_ub[np.newaxis,:]
    if A_eq.ndim == 1:
        A_eq = A_eq[np.newaxis,:]
    _, p = A_ub.shape
    if b_ub.ndim == 2:
        b_ub = b_ub[:,0]
    if b_eq.ndim == 2:
        b_eq = b_eq[:,0]
    c = np.ones((p,1))
    linprog_res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method = 'highs')
    return linprog_res.fun is not None

def intersection_checker(polytope, constrainst_a, constrainst_b):
    """Check if a constrainst Ax < b intersects with a polytope.

    Parameters
    ----------
    polytope : ElementaryPolytope
        A polytope of interest.
    constrainst_a : array_like
        1-D array of values representing A for the constrainst Ax <= b.
    constrainst_b : float
        Value representing b for the constrainst Ax <= b.
        
    Returns
    -------
    float
        0 if the constrainst intersects. 
        -1 if b < min(Ax) given the constrainsts and bounds of the polytope.
        1 if b > max(Ax) given the constrainsts and bounds of the polytope.
    """
    A_ub, b_ub, A_eq, b_eq = polytope.get_constrainsts()
    polytope_bounds = polytope.get_bounds()
    Aplus = constrainst_a
    Amoins = -constrainst_a
    bplus = constrainst_b
    bmoins = -constrainst_b
    if A_ub is not None:
        Aplus = np.vstack((Aplus, A_ub))
        Amoins = np.vstack((Amoins, A_ub))
        bplus = np.vstack((bplus, b_ub))
        bmoins = np.vstack((bmoins, b_ub))
    first_side = is_polytope_not_empty(Aplus, bplus, A_eq, b_eq, polytope_bounds)
    second_side = is_polytope_not_empty(Amoins, bmoins, A_eq, b_eq, polytope_bounds)
    if first_side and second_side :
        return 0
    if first_side:
        return 1
    if second_side:
        return -1
    
    #Case it does not work: add some noise.
    Aplus = Aplus + np.random.normal(0.0, 10**-8, size = Aplus.shape)
    Amoins = Amoins + np.random.normal(0.0, 10**-8, size = Amoins.shape)
    bplus = bplus + np.random.normal(0.0, 10**-8, size = bplus.shape)
    bmoins = bmoins + np.random.normal(0.0, 10**-8, size = bmoins.shape)
    first_side = is_polytope_not_empty(Aplus, bplus, A_eq, b_eq, polytope_bounds)
    second_side = is_polytope_not_empty(Amoins, bmoins, A_eq, b_eq, polytope_bounds)
    if first_side and second_side :
        return 0
    if first_side:
        return 1
    if second_side:
        return -1
    
    return None #If fails.

def cut_polytope(polytope, constrainst_a, constrainst_b, confidence = 1, fusion_rule = 'minimum'):
    """Seperate a polytope into two polytopes according to a constrainst Ax < b.

    Parameters
    ----------
    polytope : ElementaryPolytope
        A polytope of interest.
    constrainst_a : array_like
        1-D array of values representing A for the constrainst Ax <= b.
    constrainst_b : float
        Value representing b for the constrainst Ax <= b.
    confidence : float, optional
        The confidence level given by a DM, which will determine the possibility 
        of the second polytope.
    fusion_rule : string
        The T-norm used for merging information.
        
    Returns
    -------
    ElementaryPolytope
        The first polytope, with a possibility of 1 or lower.
    ElementaryPolytope
        The second polytope, with a possibility of (1-confidence) or lower.
        
    Raises
    ------
    ValueError
        If the confidence is not in the interval [0,1].
    """
    if confidence < 0 or confidence > 1:
        raise ValueError('The confidence has to be in the interval [0,1].')
    polytope_1 = deepcopy(polytope)
    polytope_2 = deepcopy(polytope)
    polytope_1.add_answer(constrainst_a, constrainst_b, 1, fusion_rule)
    polytope_2.add_answer(-constrainst_a, -constrainst_b, 1-confidence, fusion_rule)
    return polytope_1, polytope_2