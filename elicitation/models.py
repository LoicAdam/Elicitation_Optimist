# -*- coding: utf-8 -*-
"""This module represents the preference aggregation model."""

import numpy as np

class ModelWeightedSum():
    """
    Weighted Sum model.
    """
    def __init__(self, model_parameters):
        """
        Parameters
        ----------
        model_parameters : array_like
            Parameters of the model.
        """
        self._model_parameters = model_parameters

    def get_opti_alternative(self, alternative):
        """
        The correct form of the alternative for optimisation.
        
        Parameters
        ----------
        alternative : array_like
            An alternative represented by its criteria.
            
        Returns
        ----------
        array_like
            The correct form.
        """
        return alternative

    def get_diff(self, alternative_1, alternative_2, best_prefered = True):
        """
        Difference between two alternatives.
        
        Parameters
        ----------
        alternative_1 : array_like
            An alternative represented by its criteria.
        alternative_2 : array_like
            An alternative represented by its criteria.
        best_prefered : boolean
            Flag to know which is the best alternative.
            
        Returns
        ----------
        array_like
            The difference, criterion by criterion.
        """
        return (alternative_1-alternative_2) if best_prefered else (alternative_2-alternative_1)

    def get_model_score(self, alternatives):
        """
        Score of all alternatives according to the WS.
        
        Parameters
        ----------
        alternatives : array_like
            Number of rows alternatives, and number of columns criteria.
            
        Returns
        ----------
        array_like
            The score of each alternative.
        """
        if alternatives.ndim == 1:
            alternatives = alternatives[np.newaxis,:]
        nb_alternatives = alternatives.shape[0]
        alternative_score = np.multiply(np.repeat(np.asarray(self._model_parameters)[np.newaxis,:],
                                                  nb_alternatives, axis = 0), alternatives)
        alternative_score = np.sum(alternative_score, axis = 1)
        return alternative_score

    def get_model_constrainsts(self):
        """
        Get the initial constrainsts acording to a model.
        
        Returns
        ----------
        A_ub : 2-D array
            The inequality constraint matrix.
        b_ub : 1-D array
            The inequality constraint vector.
        A_eq : 2-D array
            The equality constraint matrix.
        b_eq : 1-D array
            The equality constraint vector.
        bounds : sequence
            The bounds of each variable.
        """
        nb_parameters = len(self._model_parameters)
        res = {}
        res['A_ub'] = []
        res['b_ub'] = []
        res['A_eq'] = np.ones((1,nb_parameters))
        res['b_eq'] = np.ones(1)
        res['bounds'] = tuple((0, 1) for _ in range(nb_parameters))
        return res

    def get_model_bounds(self):
        """
        Get the initial bounds acording to a model.
        
        Returns
        ----------
        sequence
            The bounds.
        """
        nb_parameters = len(self._model_parameters)
        return [(0,1)] * nb_parameters

class ModelOWA():
    """
    Ordered Weighted Averaging model.
    """

    def __init__(self, model_parameters):
        """
        Parameters
        ----------
        model_parameters : array_like
            Parameters of the model.
        """
        self._model_parameters = model_parameters

    def get_opti_alternative(self, alternative):
        """
        The correct form of the alternative for optimisation.
        
        Parameters
        ----------
        alternative : array_like
            An alternative represented by its criteria.
            
        Returns
        ----------
        array_like
            The correct form.
        """
        return np.sort(alternative)[::-1]

    def get_diff(self, alternative_1, alternative_2, best_prefered = True):
        """
        Difference between two alternatives.
        
        Parameters
        ----------
        alternative_1 : array_like
            An alternative represented by its criteria.
        alternative_2 : array_like
            An alternative represented by its criteria.
        best_prefered : boolean
            Flag to know which is the best alternative.
            
        Returns
        ----------
        array_like
            The difference, criterion by criterion.
        """
        alternative_1 = np.sort(alternative_1)[::-1]
        alternative_2 = np.sort(alternative_2)[::-1]
        return (alternative_1-alternative_2) if best_prefered else (alternative_2-alternative_1)

    def get_model_score(self, alternatives):
        """
        Score of all alternatives according to the OWA.
        
        Parameters
        ----------
        alternatives : array_like
            Number of rows alternatives, and number of columns criteria.
            
        Returns
        ----------
        array_like
            The score of each alternative.
        """
        if alternatives.ndim == 1:
            alternatives = alternatives[np.newaxis,:]
        alternatives_sorted = -np.sort(-alternatives, axis = 1)
        nb_alternatives = alternatives_sorted.shape[0]
        alternative_score = np.multiply(np.repeat(np.asarray(self._model_parameters)[np.newaxis,:],
                                                  nb_alternatives, axis = 0), alternatives_sorted)
        alternative_score = np.sum(alternative_score, axis = 1)
        return alternative_score

    def get_model_constrainsts(self):
        """
        Get the initial constrainsts acording to a model.
        
        Returns
        ----------
        A_ub : 2-D array
            The inequality constraint matrix.
        b_ub : 1-D array
            The inequality constraint vector.
        A_eq : 2-D array
            The equality constraint matrix.
        b_eq : 1-D array
            The equality constraint vector.
        bounds : sequence
            The bounds of each variable.
        """
        nb_parameters = len(self._model_parameters)
        res = {}
        res['A_ub'] = []
        res['b_ub'] = []
        res['A_eq'] = np.ones((1,nb_parameters))
        res['b_eq'] = np.ones(1)
        res['bounds'] = tuple((0, 1) for _ in range(nb_parameters))
        return res

    def get_model_bounds(self):
        """
        Get the initial bounds acording to a model.
        
        Returns
        ----------
        sequence
            The bounds.
        """
        nb_parameters = len(self._model_parameters)
        return [(0,1)] * nb_parameters
