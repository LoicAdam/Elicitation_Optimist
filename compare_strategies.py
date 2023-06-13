# -*- coding: utf-8 -*-

import os
import sys
import time
import multiprocessing
import pickle
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum, ModelOWA
from elicitation.elicitation import robust_elicitation
import matplotlib.pyplot as plt
import tikzplotlib
import scipy.stats as st

t_norm = "product"
nb_repetitions = 5 #Number of experiments
nb_alternatives = 5 #Number of alternatives
nb_parameters = 8 #Criteria                        
nb_questions = 11+1 #Nb iterations
path = 'results/'

def init_globals(counter):
    global cnt
    cnt = counter

def evaluation(alternatives, model, rational, strategy):
    
    res = robust_elicitation(alternatives, model, rational = rational, strategy = strategy,
                             max_iter = nb_questions)
    pos = res['pos']
    score = res['score']
    len_score = len(score)
    if len_score < nb_questions:
        elements_to_add_pos = np.ones(nb_questions - len_score) * pos[-1]
        elements_to_add_score = np.ones(nb_questions - len_score) * score[-1]
        pos = np.hstack((pos, elements_to_add_pos))
        score = np.hstack((score, elements_to_add_score))
    return pos, score

if __name__ == '__main__':
    
    def tikzplotlib_fix_ncols(obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
    
    alternatives_good_all = np.zeros((nb_repetitions, nb_alternatives, nb_parameters))
    for i in range(0, nb_repetitions):
        alternatives_good_all[i,:,:] = generate_alternatives_score(nb_alternatives,
                                                                   nb_parameters = nb_parameters,
                                                                   value = nb_parameters/2)
    alternatives_bad_all = np.random.dirichlet(np.ones(nb_parameters),
                                               size = (nb_repetitions, nb_alternatives))

    model_parameters_WS_balanced = np.random.dirichlet(np.ones(nb_parameters)/nb_parameters*1000, nb_repetitions)
    model_parameters_WS_biased = np.random.dirichlet(np.ones(nb_parameters)/nb_parameters, nb_repetitions)
    lin = np.linspace(1, nb_parameters, nb_parameters)
    model_parameters_OWA_min = np.random.dirichlet(50*lin/np.sum(lin), nb_repetitions)
    model_parameters_OWA_max = np.random.dirichlet(50*lin[::-1]/np.sum(lin), nb_repetitions)
    
    model_WS_balanced = [ModelWeightedSum(i) for i in model_parameters_WS_balanced]
    model_WS_biased  = [ModelWeightedSum(i) for i in model_parameters_WS_biased]
    model_OWA_min = [ModelOWA(i) for i in model_parameters_OWA_min]
    model_OWA_max = [ModelOWA(i) for i in model_parameters_OWA_max]

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    rational_all = np.ones((nb_repetitions, nb_questions))
    
    alternatives_type = [alternatives_good_all, alternatives_bad_all]
    alternatives_names = ['good_alternatives', 'bad_alternatives']
    model_type = [model_WS_balanced, model_WS_biased, model_OWA_min, model_OWA_max]
    model_names = ['balanced','biased','min','max']
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'data.pk','wb') as f:
        d = {}
        d['alternatives_good'] = alternatives_good_all
        d['alternatives_bad'] = alternatives_bad_all
        d['model_WS_balanced'] = model_WS_balanced
        d['model_WS_biased'] = model_WS_biased
        d['model_OWA_min'] = model_OWA_min
        d['model_OWA_max'] = model_OWA_max
        pickle.dump(d,f)
                
    for a in  range(0, len(alternatives_type)):
        for m in range(0,len(model_type)) :
    
            start_time = time.time()
            cnt = Value('i', 0)
            with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
                elicitation_mMR = pool.starmap(evaluation, zip(alternatives_type[a], 
                                                               model_type[m],
                                                               rational_all,
                                                               np.repeat("mMr", nb_repetitions)))
                elicitation_MM = pool.starmap(evaluation, zip(alternatives_type[a], 
                                                              model_type[m],
                                                              rational_all,
                                                              np.repeat("MM", nb_repetitions)))
            sys.stdout.flush()
            pool.close()
            pool.join()
            print("Done", time.time() - start_time)
            
            if not os.path.exists(path + alternatives_names[a]):
                os.makedirs(path + alternatives_names[a])
            with open(path + alternatives_names[a] + '/score_' + model_names[m] + '.pk','wb') as f:
                d = {}
                d['elicitation_mMR'] = elicitation_mMR
                d['elicitation_MM'] = elicitation_MM
                pickle.dump(d,f)
            
            score_ic = np.zeros((nb_questions,3,2))
            pos_stats = np.zeros((nb_questions,5,2))
            
            k = 0
            for eli in [elicitation_mMR, elicitation_MM]:
                pos, score = list(zip(*eli))
        
                pos = np.asarray(pos)+1
                pos_stats[:,0,k] = np.min(pos, axis = 0)
                pos_stats[:,1,k] = np.quantile(pos, 0.25, axis = 0)
                pos_stats[:,2,k] = np.median(pos, axis = 0)
                pos_stats[:,3,k] = np.quantile(pos, 0.75, axis = 0)
                pos_stats[:,4,k] = np.max(pos, axis = 0)
        
                score = np.asarray(score)
                score_ic[:,1,k] = np.mean(score, axis = 0)
                for i in range(0, nb_questions):
                    score_ic[i,[0,2],k] = st.t.interval(0.95, len(score)-1, 
                                                    loc=score_ic[i,1,k], 
                                                    scale=st.sem(score[:,i]))
                k = k+1

            fig, ax = plt.subplots(1,1)
            ax.plot(np.arange(0, nb_questions), score_ic[:,1,0], '-rX')
            ax.plot(np.arange(0, nb_questions), score_ic[:,0,0], '-r')
            ax.plot(np.arange(0, nb_questions), score_ic[:,2,0], '-r')
            ax.plot(np.arange(0, nb_questions), score_ic[:,1,1], '-gv')
            ax.plot(np.arange(0, nb_questions), score_ic[:,0,1], '-g')
            ax.plot(np.arange(0, nb_questions), score_ic[:,2,1], '-g')
            ax.set_xlabel("Number of questions")    
            ax.set_ylabel("Score")
            ax.xaxis.set_ticks(np.arange(0, nb_questions, 5))
            plt.savefig(path + alternatives_names[a] + '/score_' + model_names[m] + '.png')
            tikzplotlib.save(path + alternatives_names[a] + '/score_' + model_names[m] + '.tex')
            
            fig, ax = plt.subplots(1,1)
            ax.plot(np.arange(0, nb_questions), pos_stats[:,2,0], '-rX')
            ax.plot(np.arange(0, nb_questions), pos_stats[:,1,0], '-r')
            ax.plot(np.arange(0, nb_questions), pos_stats[:,3,0], '-r')
            ax.plot(np.arange(0, nb_questions), pos_stats[:,2,1], '-gv')
            ax.plot(np.arange(0, nb_questions), pos_stats[:,1,1], '-g')
            ax.plot(np.arange(0, nb_questions), pos_stats[:,3,1], '-g')
            ax.set_xlabel("Number of questions")    
            ax.set_ylabel("Position")
            ax.xaxis.set_ticks(np.arange(0, nb_questions, 5))
            plt.savefig(path + alternatives_names[a] + '/pos_' + model_names[m] + '.png')
            tikzplotlib.save(path + alternatives_names[a] + '/pos_' + model_names[m] + '.tex')