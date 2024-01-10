from os import cpu_count
import random as rand

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from timeit import default_timer as timer
from datetime import timedelta

from statistics import stdev

import pandas as pd
import numpy as np

import copy

import sys
import traceback

import warnings

from Individual import Individual

import os
from sklearn.ensemble import RandomForestRegressor


class BO_PU:
    def __init__(
        self,
        population_size=101,
        generation_count=50,
        mutation_prob=0.1,
        crossover_prob=0.9,
        gene_crossover_prob=0.5,
        tournament_size=2,
        fitness_threshold=0.2,
        internal_fold_count=5,
        spies=False,
        log_directory=None,
        random_state=None,
        n_jobs=1,
        try_next=False
    ):
        """Initialise the Auto_PU algorithm.

        Parameters
        ----------
        population_size: int, optional (default: 101)
            Number of indiviiduals in the population for each generation.
        generation_count: int, optional (default: 50)
            Number of iterations to run the optimisation algorithm.
        mutation_prob: float, optional (default: 0.1)
            The probability of gene of an individual undergoing mutation.
        crossover_prob: float, optional (default: 0.9)
            The probability of two individuals undergoing crossover.
        gene_crossover_prob: float, optional (default: 0.5)
            The probability of the values of a gene being swapped
            between two individuals.
        tournament_size: int, optional (default: 2)
            The number of individuals randomly sampled for tournament
            selection.
        fitness_threshold: float, optional (default: 0.2)
            The difference by which two values are considered
            significantly different by way of Cohen's d.
        internal_fold_count: int, optional (default: 5)
            The number of folds for internal cross validation.
        log_directory: string, optional (default: None)
            The directory to store log files.
        random_state: int, optional (default: None)
            The random number generator seed. Use this parameter
            for reproducibility.
        n_jobs: int, optional (default: 1)
            Number of CPUs for evaluating individuals in parallel.
        try_next: bool, optional (default: False)
            Indicates whether to use the next fittest individual
            in the population if an error occurs with the fittest.
            Only recommended for debugging.
            If errors occur with fittest individual it is
            recommended to use a higher generation count or
            number of individuals.

        Returns
        -------
        None

        """
        self.population_size = population_size
        self.generation_count = generation_count
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.fitness_threshold = fitness_threshold
        self.internal_fold_count = internal_fold_count
        self.log_directory = log_directory
        self.random_state = random_state
        self.try_next = try_next
        self.spies = spies

        if n_jobs == 0:
            raise ValueError("The value of n_jobs cannot be 0.")
        elif n_jobs < 0:
            self.n_jobs = cpu_count() + 1 + n_jobs
        else:
            self.n_jobs = n_jobs

        self.best_config = None

    def generate_individual(self):
        """Randomly generate an indvidual.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        individual: Individual
            Returns a randomly generated individual.
        """
        # instantiate the individual
        individual = Individual()

        # generate values for phase 1 a
        individual.it_count_1a = \
            individual.generate_iteration_count()
        individual.rn_thresh_1a = \
            individual.generate_rn_threshold()
        individual.classifier_1a = \
            individual.generate_classifier()

        # generate the values for phase 1 b
        individual.it_count_1b = \
            individual.generate_iteration_count()
        individual.rn_thresh_1b = \
            individual.generate_rn_threshold()
        individual.classifier_1b = \
            individual.generate_classifier()
        individual.flag_1b = \
            individual.generate_bool()
        
        # generate spy values
        if self.spies:
            individual.spies = \
                individual.generate_bool() 
            individual.spy_rate = \
                individual.generate_spy_rate()
            individual.spy_tolerance = \
                individual.generate_spy_tolerances()

        # generate the classifier for phase 2
        individual.classifier_2 = individual.generate_classifier()

        return individual

    def generate_population(self):
        """Randomly generate the population.
        A population of self.population_size will be created and the values
        for the individual genes will be randomly generated with their
        respective methods.

        Parameters
        ----------
        None

        Returns
        -------
        population: array-like {self.population_size, n_genes}
            Returns a randomly generated population.
        """

        # initialise an empty population list
        population = [self.generate_individual() for _ in range(self.population_size)]

        # return completed population
        return population

    def predict_probability(self, classifier, X_train, y_train, X_test):
        """Use the specified classifier to predict the probability
        of the instances in the test set belonging to the positive class.

        Parameters
        ----------
        classifier: Object
            The classifier to train and test.
        X_train: array-like {n_samples, n_features}
            Feature matrix.
        y_train: array-like {n_samples}
            Class values for feature matrix.
        X_test: array-like {n_samples, n_features}
            Feature matrix.

        Returns
        -------
        y_prob: array-like {n_samples}
            The predicted class values for X_test.

        """

        # clone the classifier
        clf = clone(classifier)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fit the classifier to the training set
            clf.fit(X_train, y_train)

        # predict the instances in the test set
        y_prob = clf.predict_proba(X_test)[:, 1]

        # return the predicted probability
        return y_prob

    def create_rn_x_y(self, rn_set):
        """Create the X and y vectors of the RN set.
        
        Parameters
        ----------
        rn_set: array-like {n_samples, n_features} 
            The rn set feature matrix.
        
        Returns
        -------
        X_rn: array-like {n_samples, n_features}
            Feature matrix
        y_rn: array-like {n_samples}
            Class labels.
        
        """

        # create the rn class array
        y_rn = np.array([0 for _ in range(len(rn_set))])

        return rn_set, y_rn

    def get_rn_unlabelled(self, y_prob, y_actual, rn_threshold, X):
        """For each negative instance, check if the predicted probability
        is lower than the rn threshold. If so, add to the rn set, otherwise,
        add to the unlabelled set.

        Parameters
        ----------
        y_prob: array-like {n_samples}
            Class probability predictions.
        y_actual: array-like {n_samples}
            Actual labelled class values.
        rn_threshold: float
            The threshold under which an instance is
            considered reliably negative.
        X: array-like {n_samples, n_features}
            Feature matrix.

        Returns
        -------
        rn_set: array-like {n_samples, n_features}
            The identified reliable negative instance set.
        new_unlabelled: array-like {n_samples, n_features}
            The remaining unlabelled instances.

        """

        rn_set = [X[i] for i in range(len(X)) if y_actual[i] == 0 and y_prob[i] < rn_threshold]
        new_unlabelled = [X[i] for i in range(len(X)) if y_actual[i] == 0 and y_prob[i] >= rn_threshold]

        return rn_set, new_unlabelled

    def phase_1a(self, individual, positive_set, unlabelled_set):
        """Perform phase 1 a of the PU learning procedure.
        If the boolean Spy is True, spies will be used to determine 
        threshold 1A. 

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        positive_set: DataFrame
            The set of positive instances.
        unlabelled_set: DataFrame
            The set of unlabelled instances.

        Returns
        -------
        rn_set: array-like {n_samples, n_features}
            The set of reliable negative instances.
        new_unlabelled: array-like {n_samples, n_features}
            The set of unlabelled instances with the reliable negatives
            removed.

        """

        # unlabelled features
        X_unlabelled = unlabelled_set[:, :-1]
        # unlabelled class
        y_unlabelled = unlabelled_set[:, -1]

        # positive features
        X_positive = positive_set[:, :-1]
        # positive class
        y_positive = positive_set[:, -1]

        # initialise the rn set and unlabelled set to be returned,
        # as empty sets
        rn_set = []
        new_unlabelled = []

        # if the individual does not use spies, use the threshold defined by 
        # threshold_1a
        if not individual.spies:

            # if the iteration count is greater than 1, split the unlabelled set
            # into it_count_1a subsets to handle the class imbalance
            # then perform classification with the smaller unlabelled set
            # and the positive set
            if individual.it_count_1a > 1:
                # stratified k fold the data
                skf = StratifiedKFold(n_splits=individual.it_count_1a,
                                    random_state=self.random_state, shuffle=True)
                skf.get_n_splits(X_unlabelled, y_unlabelled)
                c = 0

                # split the unlabelled set
                # switch test and train so that we're training with the smaller set
                for _, train in skf.split(X_unlabelled, y_unlabelled):
                    # create training set
                    X_train, y_train = X_unlabelled[train], y_unlabelled[train]
                    X_train = np.concatenate([X_train, X_positive])
                    y_train = np.concatenate([y_train, y_positive])

                    # get the phase 1 a classifier, fit to the training set
                    # and predict the probability of each instance in training set
                    # being positive

                    y_prob = \
                        self.predict_probability(individual.classifier_1a,
                                                X_train,
                                                y_train,
                                                X_train)

                    # get the rn set and the unlabelled set from this iteration
                    rn, unlabelled = \
                        self.get_rn_unlabelled(y_prob,
                                            y_train,
                                            individual.rn_thresh_1a,
                                            X_train)
                    rn_set.append(rn)
                    new_unlabelled.append(unlabelled)
                    c+=1

            # if the iteration count is 1, get the rn set without
            # using undersampling
            else:
                # create the training set
                X_train = np.concatenate([X_positive, X_unlabelled])
                y_train = np.concatenate([y_positive, y_unlabelled])

                # get the phase 1 a classifier, fit to the training set
                # and predict the probability of each instance in training set
                # being positive
                y_prob = self.predict_probability(individual.classifier_1a,
                                                X_train,
                                                y_train,
                                                X_train)

                # get the rn set and the unlabelled set
                rn, unlabelled = \
                    self.get_rn_unlabelled(y_prob,
                                        y_train,
                                        individual.rn_thresh_1a,
                                        X_train)
                rn_set.append(rn)
                new_unlabelled.append(unlabelled)
        
        else:
            
            if individual.it_count_1a > 1:
                # stratified k fold the data
                skf = StratifiedKFold(n_splits=individual.it_count_1a,
                                    random_state=self.random_state, shuffle=True)
                skf.get_n_splits(X_unlabelled, y_unlabelled)
                c = 0

                # split the unlabelled set
                # switch test and train so that we're training with the smaller set
                for _, train in skf.split(X_unlabelled, y_unlabelled):
                    # create training set
                    X_train, y_train = X_unlabelled[train], y_unlabelled[train]
                    X_train = np.concatenate([X_train, X_positive])
                    y_train = np.concatenate([y_train, y_positive])

                    OP = copy.deepcopy(positive_set[:, :-1])
                    P = copy.deepcopy(positive_set[:, :-1])
                    OU = copy.deepcopy(unlabelled_set[:, :-1])
                    U = copy.deepcopy(unlabelled_set[:, :-1])
                    S = []
                    OSlen = 0

                    to_del = []
                    while len(S) < len(OP)*individual.spy_rate:
                        index = rand.choice(range(len(P)))
                        to_del.append(index)
                        S.append(P[index])
                        OSlen += 1
                    P = np.delete(P, to_del, axis=0)
                    
                    US = np.vstack([U, S])
                    MS_spies = np.hstack([np.zeros(len(U)), np.ones(len(S))])
                    USy = np.zeros(len(US))
                    Px = P
                    USP = np.vstack([US, Px])
                    Py = np.ones(len(Px))
                    USPy = np.hstack([USy, Py])
                    # Fit first model

                    y_prob = self.predict_probability(individual.classifier_1a, USP, USPy, US)

                    # Find optimal t
                    t = 0.001
                    while MS_spies[y_prob <= t].sum()/MS_spies.sum() <= individual.spy_tolerance:
                        t += 0.001

                    # get the rn set and the unlabelled set
                    rn, unlabelled = \
                        self.get_rn_unlabelled(y_prob,
                                            USy,
                                            t,
                                            US)
                    rn_set.append(rn)
                    new_unlabelled.append(unlabelled)

            else:
                OP = copy.deepcopy(positive_set[:, :-1])
                P = copy.deepcopy(positive_set[:, :-1])
                U = copy.deepcopy(unlabelled_set[:, :-1])
                S = []
                OSlen = 0

                to_del = []
                while len(S) < len(OP)*individual.spy_rate:
                    index = rand.choice(range(len(P)))
                    to_del.append(index)
                    S.append(P[index])
                    OSlen += 1
                P = np.delete(P, to_del, axis=0)
                
                US = np.vstack([U, S])
                MS_spies = np.hstack([np.zeros(len(U)), np.ones(len(S))])
                USy = np.zeros(len(US))
                Px = P
                USP = np.vstack([US, Px])
                Py = np.ones(len(Px))
                USPy = np.hstack([USy, Py])
                # Fit first model

                y_prob = self.predict_probability(individual.classifier_1a, USP, USPy, US)

                # Find optimal t
                t = 0.001
                while MS_spies[y_prob <= t].sum()/MS_spies.sum() <= individual.spy_tolerance:
                    t += 0.001

                # get the rn set and the unlabelled set
                rn, unlabelled = \
                    self.get_rn_unlabelled(y_prob,
                                        USy,
                                        t,
                                        US)
                rn_set.append(rn)
                new_unlabelled.append(unlabelled)

        # flatten the rn and unlabelled lists
        rn_set = [instances for rn in rn_set for instances in rn]
        new_unlabelled = [instances for unlabelled in new_unlabelled
                        for instances in unlabelled]

        if len(rn_set) < 1:
            raise Exception("No RN set identified.")

        # return the rn set and the new unlabelled set
        return rn_set, new_unlabelled

    def phase_1_b(self, individual, positive_set, rn_set, unlabelled_set):
        """Perform phase 1 b of the PU learning procedure.

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        positive_set: array-like {n_samples, n_features}
            The set of instances labelled as positive.
        rn_set: array-like {n_samples, n_features}
            The set of instances identified as reliably negative.
        unlabelled_set: array-like {n_samples, n_features}
            The set of unlabelled instances.

        Returns
        -------
        rn_set: array-like {n_samples, n_features}
            The set of instances identified as reliably negative.
        new_unlabelled: array-like {n_samples, n_features}
            The set of remaining unlabelled instances.

        """

        # unlabelled features
        X_unlabelled = np.array(unlabelled_set)

        y_unlabelled = [0 for _ in range(len(X_unlabelled))]

        # positive features
        X_positive = positive_set[:, :-1]
        # positive class
        y_positive = positive_set[:, -1]

        # create rn x and y vectors
        X_rn, y_rn = self.create_rn_x_y(rn_set)

        # create the training set
        X_train = np.concatenate([X_positive, X_rn])
        y_train = np.concatenate([y_positive, y_rn])

        # get the phase 1 b classifier, fit to the training set
        # and predict the probability of each instance in unlabelled set
        # being positive
        y_prob = self.predict_probability(individual.classifier_1b,
                                          X_train,
                                          y_train,
                                          X_unlabelled)

        # get the rn set and the unlabelled set
        new_rn_set, new_unlabelled = \
            self.get_rn_unlabelled(y_prob,
                                   y_unlabelled,
                                   individual.rn_thresh_1b,
                                   X_unlabelled)

        rn = [rn_set, new_rn_set]
        rn_set = [instances for rns in rn for instances in rns]

        # return the new reliable negatives and unlabelled instances
        return rn_set, new_unlabelled

    def pred(self, classifier, X_train, y_train, X_test):
        """Predict whether the instances in the test set are
        positive or negative.

        Parameters
        ----------
        classifier: Object
            The classifier to build and evaluate.
        X_train: array-like {n_samples, n_features}
            Training set feature matrix.
        y_train: array-like {n_samples}
            Training set class labels.
        X_test: array-like {n_samples, n_features}
            Test set feature matrix.

        Returns
        -------
        y_pred: array-like {n_samples}
            Class label predictions.

        """

        # clone the classifier
        clf = clone(classifier)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fit the classifier to the training set
            clf.fit(X_train, y_train)

        # predict the instances in the test set
        y_pred = clf.predict(X_test)

        # return the predictions
        return y_pred

    def perf_measure(self, y_actual, y_hat):
        """Get the true positive, false positive, true negative,
        and false negative counts of predictions (y_hat).

        Parameters
        ----------
        y_actual: array-like {n_samples}
            The actual class labels.
        y_hat: array-like {n_samples}
            Predicted class labels.

        Returns
        -------
        TP: int
            The true positive count.
        FP: int
            The false positive count.
        TN: int
            The true negative count.
        FN: int
            The false negative count.

        """

        TP = len([1 for i in range(len(y_actual)) if y_actual[i] == y_hat[i] == 1])
        FP = len([1 for i in range(len(y_actual)) if y_actual[i] != y_hat[i] == 1])
        TN = len([1 for i in range(len(y_actual)) if y_actual[i] == y_hat[i] == 0])
        FN = len([1 for i in range(len(y_actual)) if y_actual[i] != y_hat[i] == 0])

        return TP, FP, TN, FN


    def internal_CV(self, individual, X, y):
        """Perform internal cross validation to get a list of
        recalls and precisions.

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        X: array-like {n_samples, n_features}
            Feature matrix.
        y: array-like {n_samples}
            Class values for feature matrix.

        Returns
        -------
        """

        # stratified 5-fold
        skf = StratifiedKFold(n_splits=self.internal_fold_count,
                              random_state=42, shuffle=True)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # perform 5 fold CV
        for train, test in skf.split(X, y):
            
            try:
                X_train, y_train = np.array(X[train]), np.array(y[train])
                X_test, y_test = X[test], y[test]
                training_set = np.c_[X_train, y_train]

                unlabelled_set = np.array([x for x in training_set if x[-1] == 0])
                positive_set = np.array([x for x in training_set if x[-1] == 1])

                # perform phase one a
                rn_set, unlabelled_set = \
                    self.phase_1a(individual, positive_set, unlabelled_set)

            

                # ensure that phase 1 a returned a reliable negative set
                # before proceeding
                if len(rn_set) > 0:

                    # perform phase 1 b if the flag is set and if there are
                    # still instances left in the unlabelled set
                    if individual.flag_1b and len(unlabelled_set) > 0:
                        rn_set, unlabelled_set = \
                            self.phase_1_b(individual, positive_set, rn_set,
                                        unlabelled_set)

                    # positive features
                    X_positive = positive_set[:, :-1]
                    # positive class
                    y_positive = positive_set[:, -1]

                    X_rn, y_rn = self.create_rn_x_y(rn_set)

                    # create the training set
                    X_train = np.concatenate([X_positive, X_rn])
                    y_train = np.concatenate([y_positive, y_rn])
                else:
                    raise Exception("No RN set identified.")

                # predict the class of the instances in the test set
                # with the phase 2 classifier trained on the positive
                # and the reliable negative data
                y_pred = self.pred(individual.classifier_2,
                                X_train,
                                y_train,
                                X_test)

                # get the true positives, false positives, true negatives,
                # and false negatives
                TP, FP, TN, FN = self.perf_measure(y_test, y_pred)
                
                tp += TP
                fp += FP
                tn += TN
                fn += FN

            except Exception as e:
                tp += 0
                fp += 0
                tn += 0
                fn += 0

                print(e)
                print(traceback.format_exc())

        individual.tp = tp
        individual.fp = fp
        individual.tn = tn
        individual.fn = fn

        try:
            recall = (tp/(tp+fn))
        except:
            recall = 0

        try:        
            precision = (tp/(tp+fp))
        except:
            precision = 0
        
        try:
            f_measure = (2*((precision * recall)/(precision+recall)))
        except:
            f_measure = 0

        return f_measure

    def assess_fitness(self, individual, features, target, assessed):
        """Assess the fitness of an individual on the current training set.
        Individual will be checked against the list of already assessed
        configurations. If present, their recall and precision will be set
        to the values previously calculated for the configuration on this
        training set.

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            Class labels.
        assessed: array-like {n_assessed_configs}
            The previously assessed configurations.

        Returns
        -------
        new_assessed_configs: array-like {n_assessed_configs}
            The new list of assessed configurations.

        """
        
        # copy all values and add to config list
        it_count_1a = copy.deepcopy(individual.it_count_1a)
        rn_thresh_1a = copy.deepcopy(individual.rn_thresh_1a)
        classifier_1a = copy.deepcopy(individual.classifier_1a)
        rn_thresh_1b = copy.deepcopy(individual.rn_thresh_1b)
        classifier_1b = copy.deepcopy(individual.classifier_1b)
        flag_1b = copy.deepcopy(individual.flag_1b)
        spies = copy.deepcopy(individual.spies)
        spy_rate = copy.deepcopy(individual.spy_rate)
        spy_tolerance = copy.deepcopy(individual.spy_tolerance)
        classifier_2 = copy.deepcopy(individual.classifier_2)
        config = [it_count_1a, rn_thresh_1a, classifier_1a,
                  rn_thresh_1b, classifier_1b, flag_1b, spies, spy_rate,
                  spy_tolerance, classifier_2]
                
        # initialise found as false
        found = False

        # search for current config in list of assessed configs
        # if found, set fitness to the previously calculated values
        for i in range(len(assessed)):
            if str(config) == assessed[i][0]:
                found = True
                individual.fitness = assessed[i][5]

        # if the configuration is not found, assess it
        if not found:
            try:
                individual.fitness = self.internal_CV(individual, features, target)
            except Exception as e:
                # if individual produces an error, fitness
                # of the individual is set to 0
                print(traceback.format_exc())
                individual.fitness = 0

        return individual

    def get_avg_prec_rec(self, population):
        """Get the average recall, precision, and average
        standard deviation for both.

        Parameters
        ----------
        population: array_like {population_size}
            The list of individuals.

        Returns
        -------
        avg_precision: float
            The average precision of the population.
        avg_recall: float
            The average recall of the population.
        avg_f_measure: float
            The average f-measure of the population.
        avg_std_precision: float
            The average standard deviation of precision of the population.
        avg_std_recall: float
            The average standard deviation of recall of the population.
        avg_std_f_measure: float
            The average standard deviation of f-measure of the population.

        """

        # initialise all values as 0
        avg_f_measure = 0

        pop_len = len(population)

        # for every individual in the population
        # add values to total
        for individual in population:
            avg_f_measure += individual.fitness
        # get the average of the values
        avg_f_measure = (avg_f_measure / pop_len)

        return avg_f_measure


    def get_total_difference(self, list, avg):
        """Get the sum of the difference between all values
        in list and the mean value.

        Parameters
        ----------
        list:
            List of values of which to calculate difference.
        avg:
            Mean.

        Returns
        -------
        diff_total: float
            Total difference.

        """

        # initialise total difference as 0
        diff_total = 0

        # get the sum of all values minus the mean
        for value in list:
            diff_total += (value - avg)

        return diff_total

    def get_pooled_std_dev(self, new_indiv_diff, current_indiv_diff):
        """Get the pooled standard deviation.

        Parameters
        ----------
        new_indiv_diff: float
            Difference between value of the new individual and the mean.
        current_indivi_diff: float
            Difference between value of the current individual and the mean.

        Returns
        -------
        std_pooled: float
            The pooled standard deviation.

        """

        # numerator of equation
        numerator = ((new_indiv_diff ** 2) + (current_indiv_diff ** 2))

        # denominator of equation
        denominator = (self.internal_fold_count + self.internal_fold_count - 2)

        # calculate pooled standard deviation
        std_pooled = numerator / denominator

        return std_pooled

    def get_fittest(self, population):
        """Get the fittest individual in a population.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        fittest_individual: Individual
            The individual with the best fitness in the population.
        f: int
            The number of individuals deemed fitter than the comparison
            by way of f-measure.
        rec: int
            The number of individuals deemed fitter than the comparison
            by way of recall.

        """

        # initialise fittest individual as the first in the population
        fittest_individual = population[0]

        # counters for measure that determines decision
        f = 0
        rec = 0

        # compare every individual in population
        for i in range(len(population)):

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if population[i].fitness > \
                    fittest_individual.fitness:
                fittest_individual = population[i]

        # return the fittest individual and the counters
        return fittest_individual, f, rec

    def get_fittest_estimate(self, population):
        """Get the fittest individual in a population.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        fittest_individual: Individual
            The individual with the best fitness in the population.
        f: int
            The number of individuals deemed fitter than the comparison
            by way of f-measure.
        rec: int
            The number of individuals deemed fitter than the comparison
            by way of recall.

        """

        # initialise fittest individual as the first in the population
        fittest_individual = population[0]

        # counters for measure that determines decision
        f = 0
        rec = 0

        # compare every individual in population
        for i in range(len(population)):

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if population[i].est_f_measure > \
                    fittest_individual.est_f_measure:
                fittest_individual = population[i]

        # return the fittest individual and the counters
        return fittest_individual, f, rec

    def log_individuals(self, population, current_generation):
        """Save the details of all individuals in population to csv.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.
        current_generation: int
            The current generation number.

        Returns
        -------
        None

        """

        # initialise list for storing the details of all individuals
        # in the population
        individual_details = []

        # for every individual in the population, convert the values to strings
        # and save to individual details list
        for individual in population:
            it_count_1a = str(individual.it_count_1a)
            rn_thresh_1a = str(individual.rn_thresh_1a)
            classifier_1a = str(individual.classifier_1a)
            rn_thresh_1b = str(individual.rn_thresh_1b)
            classifier_1b = str(individual.classifier_1b)
            flag_1b = str(individual.flag_1b)
            classifier_2 = str(individual.classifier_2)
            f_measure = str(individual.fitness)
            tp = str(individual.tp)
            fp = str(individual.fp)
            tn = str(individual.tn)
            fn = str(individual.fn)

            indiv_detail = [it_count_1a, rn_thresh_1a,
                            classifier_1a, rn_thresh_1b,
                            classifier_1b, flag_1b, classifier_2,
                            tp, fp, tn, fn, f_measure]

            individual_details.append(indiv_detail)

        # column names
        col = ["Iteration Count P1A", "RN Threshold P1A", "Classifier P1A",
               "RN Threshold P1B", "Classifier P1B", "Flag P1B",
               "Classifier P2", "TP",
               "FP", "TN", "FN", "F measure"]

        # create dataframe
        individuals_df = pd.DataFrame(individual_details, columns=[col])

        try:
            # save to csv
            individuals_df.to_csv(self.log_directory +
                                  " Generation " +
                                  str(current_generation) +
                                  " individual details BO.csv",
                                  index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory +
                  " Generation " + str(current_generation) +
                  " individual details BO OG.csv")

    def tournament_selection(self, population):
        """Tournament selection.
        Randomly select individuals from the population and
        return the fittest.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        fittest_individual: Individual
            The individual with the best fitness in the population.
        f: int
            The number of individuals deemed fitter than
            the comparison by way of f-measure.
        rec: int
            The number of individuals deemed fitter than
            the comparison by way of recall.
        """

        # initialise tournament population as empty list
        tournament = [rand.choice(population) for _ in range(self.tournament_size)]

        # return the fittest individual from the population
        return self.get_fittest(tournament)

    def swap_genes(self, new_indiv1, new_indiv2):
        """Swap each gene of two individuals with a given probability.

        Parameters
        ----------
        new_indiv1: Individal
            First individual for swapping genes.
        new_indiv2: Individual
            Second individual for swapping genes.

        Returns
        -------
        new_indiv1: Individual
            First individual after swapping genes.
        new_indiv2: Individual
            Second individual after swapping genes.

        """

        # swap phase 1 a iteration count
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.it_count_1a = \
                copy.deepcopy(new_indiv2.it_count_1a)
            new_indiv2.it_count_1a = \
                copy.deepcopy(new_indiv1.it_count_1a)

        # swap phase 1 a rn threshold
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.rn_thresh_1a = \
                copy.deepcopy(new_indiv2.rn_thresh_1a)
            new_indiv2.rn_thresh_1a = \
                copy.deepcopy(new_indiv1.rn_thresh_1a)

        # swap phase 1 a classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_1a = \
                copy.deepcopy(new_indiv2.classifier_1a)
            new_indiv2.classifier_1a = \
                copy.deepcopy(new_indiv1.classifier_1a)

        # swap phase 1 b iteration count
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.iteration_count_1_b = \
                copy.deepcopy(new_indiv2.iteration_count_1_b)
            new_indiv2.iteration_count_1_b = \
                copy.deepcopy(new_indiv1.iteration_count_1_b)

        # swap phase 1 b rn threshold
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.rn_thresh_1b = \
                copy.deepcopy(new_indiv2.rn_thresh_1b)
            new_indiv2.rn_thresh_1b = \
                copy.deepcopy(new_indiv1.rn_thresh_1b)

        # swap phase 1 b classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_1b = \
                copy.deepcopy(new_indiv2.classifier_1b)
            new_indiv2.classifier_1b = \
                copy.deepcopy(new_indiv1.classifier_1b)

        # swap phase 1 b flag
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.flag_1b = \
                copy.deepcopy(new_indiv2.flag_1b)
            new_indiv2.flag_1b = \
                copy.deepcopy(new_indiv1.flag_1b)

        # swap phase 2 classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_2 = \
                copy.deepcopy(new_indiv2.classifier_2)
            new_indiv2.classifier_2 = \
                copy.deepcopy(new_indiv1.classifier_2)

        # return the new individuals
        return new_indiv1, new_indiv2

    def get_best_configuration(self, fittest_individual):
        """Get the details of the fittest individual configuration.
        Convert to string so all are the same data type.

        Parameters
        ----------
        fittest_individual: Individual
            The fittest individual from the population.

        Returns
        -------
        best_configuration: array-like {n_genes}
            List of string values representing individual configuration.

        """
        it_count_1a = str(fittest_individual.it_count_1a)
        rn_thresh_1a = str(fittest_individual.rn_thresh_1a)
        classifier_1a = str(clone(fittest_individual.classifier_1a))
        rn_thresh_1b = str(fittest_individual.rn_thresh_1b)
        classifier_1b = str(clone(fittest_individual.classifier_1b))
        flag_1b = str(fittest_individual.flag_1b)
        classifier_2 = str(clone(fittest_individual.classifier_2))

        # add the values to a list showing the best configuration
        best_configuration = [it_count_1a, rn_thresh_1a,
                              classifier_1a, rn_thresh_1b, classifier_1b,
                              flag_1b, classifier_2]

        return best_configuration

    def log_generation_stats(self, fittest_individual,
                             avg_recall, avg_precision,
                             avg_f_measure, avg_std_dev_recall,
                             avg_std_dev_precision, avg_std_dev_f_measure,
                             current_generation):
        """Save all statistics from a generation to csv file.

        Parameters
        ----------
        fittest_individual: Individual
            The fittest individual from the population.
        avg_recall: float
            Average recall of all individuals in the population.
        avg_precision: float
            Average precision of all individuals in the population.
        avg_f_measure: float
            Average f-measure of all individuals in the population.
        avg_std_dev_recall: float
            Average standard deviation of recall of all
            individuals in the population.
        avg_std_dev_precision: float
            Average standard deviation of precision of all
            individuals in the population.
        avg_std_dec_f_measure: float
            Average standard deviation of f-measure of all
            individuals in the population.
        current_generation: int
            The current generation.

        Returns
        -------
        None

        """

        # first column of the combined stats file
        # multiple blank values are needed as all columns
        # must have the same number of values
        stat_names = ["Recall", "Precision", "F measure",
                      "tp",
                      "fp",
                      "tn",
                      "fn", "", "", ""]

        # second column of combined stats file denotes the
        # recall and precision of the fittest individual
        f_measure = str(fittest_individual.fitness)
        tp = str(fittest_individual.tp)
        fp = str(fittest_individual.fp)
        tn = str(fittest_individual.tn)
        fn = str(fittest_individual.fn)
        best_stats = ["", "", f_measure, tp, fp, tn, fn, "", "", ""]

        # third columns denotes the average population recall and precision
        avg_stats = [str(avg_recall), str(avg_precision), str(avg_f_measure),
                     str(avg_std_dev_recall), str(avg_std_dev_precision),
                     str(avg_std_dev_f_measure), "", "", "", ""]

        # the final two columns give details of the best configuration
        # these are the configuration component names
        config_names = ["Iteration Count P1A", "RN Threshold P1A",
                        "Classifier P1A", "RN Threshold P1B", "Classifier P1B",
                        "Flag P1B", "Classifier P2"]

        # get the details of the best configuration
        best_configuration = self.get_best_configuration(fittest_individual)

        # combine all columns into a single list
        combined_stats = [stat_names, best_stats, avg_stats,
                          config_names, best_configuration]

        # conver to np array so can be transposed in dataframe
        stats = np.array(combined_stats)

        # create dataframe
        generation_stats = pd.DataFrame(
            stats.T, columns=["Stat", "Best individual",
                              "Population average", "Configuration", "Best"])

        try:
            # save dataframe as csv
            generation_stats.to_csv(self.log_directory + " Generation " +
                                    str(current_generation) +
                                    " stats.csv", index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory +
                  " Generation " + str(current_generation) +
                  " stats.csv")

    def log_fitness_decisions(self, name, f_decs, recall_decs):
        """Store the details of the fitness decisions.

        Parameters
        ----------
        name: String
            Filename to save log file as.
        f_decs: int
            The number of individuals deemed fitter than their comparison
            by way of f-measure.
        recall_decs: int
            The number of individuals deemed fitter than their comparison
            by way of recall.

        Returns
        -------
        None

        """

        # add all values to dataframe
        fitness_decisions = pd.DataFrame(np.array([f_decs, recall_decs]).T,
                                         columns=["F_measure", "Recall"])

        try:
            # save dataframe to csv
            fitness_decisions.to_csv(self.log_directory +
                                     str(name) + ".csv")
        except Exception as e:
            print("Could not save file:", self.log_directory +
                  str(name) + ".csv")
            print(e)
            print(traceback.format_exc())
    
    def get_X_indiv(self, individual):

        X_indiv = []

        X_indiv.append(individual.it_count_1a)
        X_indiv.append(individual.rn_thresh_1a)
        classifier_1a = str(individual.classifier_1a)
        if classifier_1a == "LinearDiscriminantAnalysis()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "CascadeForestClassifier(backend='sklearn', random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "KNeighborsClassifier()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "BernoulliNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "GaussianNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "LogisticRegression(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "ExtraTreesClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "ExtraTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "HistGradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "AdaBoostClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('sgdclassifier', SGDClassifier(loss='log', random_state=42))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "RandomForestClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "BaggingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "DecisionTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "MLPClassifier(random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "GaussianProcessClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "GradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1a == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('svc',\n                 SVC(gamma='auto', probability=True, random_state=42,\n                     verbose=0))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if individual.spies:
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        X_indiv.append(individual.spy_rate)
        X_indiv.append(individual.spy_tolerance)
        X_indiv.append(individual.rn_thresh_1b)
        classifier_1b = str(individual.classifier_1b)
        if classifier_1b == "LinearDiscriminantAnalysis()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "CascadeForestClassifier(backend='sklearn', random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "KNeighborsClassifier()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "BernoulliNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "GaussianNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "LogisticRegression(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "ExtraTreesClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "ExtraTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "HistGradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "AdaBoostClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('sgdclassifier', SGDClassifier(loss='log', random_state=42))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "RandomForestClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "BaggingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "DecisionTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "MLPClassifier(random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "GaussianProcessClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "GradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_1b == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('svc',\n                 SVC(gamma='auto', probability=True, random_state=42,\n                     verbose=0))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)

        if individual.flag_1b == "True":
            X_indiv.append(1)
        else:
            X_indiv.append(0)

        classifier_2 = str(individual.classifier_2)
        if classifier_2 == "LinearDiscriminantAnalysis()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "CascadeForestClassifier(backend='sklearn', random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "KNeighborsClassifier()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "BernoulliNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "GaussianNB()":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "LogisticRegression(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "ExtraTreesClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "ExtraTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "HistGradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "AdaBoostClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('sgdclassifier', SGDClassifier(loss='log', random_state=42))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "RandomForestClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "BaggingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "DecisionTreeClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "MLPClassifier(random_state=42, verbose=0)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "GaussianProcessClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "GradientBoostingClassifier(random_state=42)":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        if classifier_2 == "Pipeline(steps=[('standardscaler', StandardScaler()),\n                ('svc',\n                 SVC(gamma='auto', probability=True, random_state=42,\n                     verbose=0))])":
            X_indiv.append(1)
        else:
            X_indiv.append(0)

        return X_indiv

    def fit(self, features, target):
        """Fit an optimised PU learning
        algorithm to the input data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            List of class labels for prediction.

        Returns
        -------
        self: object
            Returns a fitted Auto_PU object.
        """

        # self.seed_everything()

        features = np.array(features)
        target = np.array(target)

        if self.random_state is not None:
            rand.seed(self.random_state)
            np.random.seed(self.random_state)

        # randomly initialise the population
        population = self.generate_population()
        # initialise current generation to 0
        current_generation = 0

        # list to store all assessed configurations
        # saves computation time by allowing us to not have to
        # assess configurations that have already been assessed
        assessed_configs = []

        # lists to store how fittest individual was defined
        # e.g., the number of time recall was the most important factor
        recall_decs = []
        f_decs = []

        # lists to store how fittest individual was defined in crossover
        # e.g., the number of time recall was the most important factor
        cross_f_decs = []
        cross_recall_decs = []

        # assess the fitness of individuals concurrently
        # the ray.get function calls the assess_fitness function with
        # the parameters specified in remote()
        # this will call the function for all individuals in population

        population = \
            Parallel(n_jobs=self.n_jobs,
                        verbose=0)(delayed(self.assess_fitness)
                                    (individual=population[i],
                                    features=features,
                                    target=target, assessed=assessed_configs)
                                    for i in range(len(population)))
        
        #population = [self.assess_fitness(population[i], features, target, assessed_configs) for i in range(len(population))]

        for i in range(len(population)):
            description = str([population[i].it_count_1a,
            population[i].rn_thresh_1a,
            population[i].classifier_1a,
            population[i].spies,
            population[i].spy_rate,
            population[i].spy_tolerance,
            population[i].rn_thresh_1b,
            population[i].classifier_1b,
            population[i].flag_1b,
            population[i].classifier_2])

            if description not in assessed_configs:
                # add config to assessed list
                assessed_configs.append([description, 
                                        population[i].tp, 
                                        population[i].fp, 
                                        population[i].tn,
                                        population[i].fn,
                                        population[i].fitness])

        # calculate average precision and recall
        avg_f_measure = self.get_avg_prec_rec(population)

        X_individuals = []
        y_individuals = []

        for i in range(len(population)):
            X_individuals.append(self.get_X_indiv(population[i]))
            y_individuals.append([population[i].fitness])       

                                        
        clf = RandomForestRegressor(random_state=42)
        clf.fit(X_individuals, y_individuals)

        # start the optimisation process
        while current_generation < self.generation_count:

            start_time = timer()

            new_pop = self.generate_population()


            X_new_indivs = []
            for i in range(len(new_pop)):
                X_new_indivs.append(self.get_X_indiv(new_pop[i]))
            
            y_pred = clf.predict(X_new_indivs)

            ix = np.argmax(y_pred)
            X_individuals.append(self.get_X_indiv(new_pop[ix]))

            # assess the fitness of individuals concurrently
            # the ray.get function calls the assess_fitness function with
            # the parameters specified in remote()
            # this will call the function for all individuals in population

            new_pop = self.assess_fitness(individual=new_pop[ix],features=features,target=target, assessed=assessed_configs)
            
            population.append(new_pop)
            
            #population = [self.assess_fitness(population[i], features, target, assessed_configs) for i in range(len(population))]

            description = str([population[i].it_count_1a,
            population[i].rn_thresh_1a,
            population[i].classifier_1a,
            population[i].spies,
            population[i].spy_rate,
            population[i].spy_tolerance,
            population[i].rn_thresh_1b,
            population[i].classifier_1b,
            population[i].flag_1b,
            population[i].classifier_2])

            if description not in assessed_configs:
                # add config to assessed list
                assessed_configs.append([description, 
                                        population[i].tp, 
                                        population[i].fp, 
                                        population[i].tn,
                                        population[i].fn,
                                        population[i].fitness])

            y_individuals.append([new_pop.fitness])

            # # calculate average precision and recall
            # avg_precision, avg_recall, avg_f_measure, avg_std_dev_precision, \
            #     avg_std_dev_recall, avg_std_dev_f_measure \
            #     = self.get_avg_prec_rec(new_pop)   


            if self.log_directory is not None:
                # save the details of all individuals in population
                # for this generation
                self.log_individuals(population, current_generation)

            clf = RandomForestRegressor(random_state=42)
            clf.fit(X_individuals, np.array(y_individuals).ravel())
            
            # if self.log_directory is not None:
            #     # save the statistics of this generation to a csv file
            #     self.log_generation_stats(fittest_individual,
            #                               0,
            #                               0,
            #                               avg_f_measure,
            #                               0,
            #                               0,
            #                               0,
            #                               current_generation)    

            end_time = timer()

            # increment generation count
            current_generation += 1

        if self.log_directory is not None:
            # save details of the fitness decisions to a csv file

            self.log_fitness_decisions("Fitness decisions",
                                       f_decs,
                                       recall_decs)

            # save details of the fitness decisions to a csv file
            self.log_fitness_decisions("Crossover fitness decisions",
                                       cross_f_decs,
                                       cross_recall_decs)

        target = np.array(target, subok=True, ndmin=2).T

        training_set = np.concatenate([features, target], axis=1)

        # get positive set and initial unlabelled set
        unlabelled_set = training_set[training_set[:, -1] == 0]
        positive_set = training_set[training_set[:, -1] == 1]

        for _ in range(count):

            try:
                fittest_individual, _, _ = self.get_fittest(population)

                self.best_config = fittest_individual

                self.best_config.classifier_1a = clone(fittest_individual.classifier_1a)
                self.best_config.classifier_1b = clone(fittest_individual.classifier_1b)
                self.best_config.classifier_2 = clone(fittest_individual.classifier_2)

                self.best_config.print_details()

                population.remove(fittest_individual)

                rn_set = []
                # perform phase one a
                rn_set, unlabelled_set = \
                    self.phase_1a(self.best_config,
                                positive_set,
                                unlabelled_set)


                # ensure that phase 1 a returned a reliable negative set
                # before proceeding
                if len(rn_set) > 0:
                    # perform phase 1 b if the flag is set and if there are
                    # still instances left in the unlabelled set
                    if self.best_config.flag_1b and len(unlabelled_set) > 0:
                        rn_set, unlabelled_set = \
                            self.phase_1_b(self.best_config,
                                           positive_set,
                                           rn_set,
                                           unlabelled_set)

                    # positive features
                    X_positive = positive_set[:, :-1]
                    # positive class
                    y_positive = positive_set[:, -1]

                    X_rn, y_rn = self.create_rn_x_y(rn_set)

                    # create the training set
                    X_train = np.concatenate([X_positive, X_rn])
                    y_train = np.concatenate([y_positive, y_rn])

                    # predict the class of the instances in the test set
                    # with the phase 2 classifier trained on the positive
                    # and the reliable negative data
                    self.best_config.classifier_2 = clone(self.best_config.classifier_2)
                    self.best_config.classifier_2.fit(X_train, y_train)

                    break

            except Exception as e:

                print("Optimised individual was unable to be trained on full \
                      training set.")
                print("It is likely that the individual was overfit during \
                      the optimisation process.")
                if not self.try_next:
                    print("Try again with different parameters, such as a \
                          higher number of individuals or generations.")
                else:
                    print("Trying next individual.")
                print("For debugging, the exception is printed below.")
                print(e)
                traceback.print_exc()

        return self

    def predict(self, features):

        features = np.array(features)

        if not self.best_config:
            raise RuntimeError(
                "Auto_PU has not yet been fitted to the data. \
                Please call fit() first."
            )

        try:
            return self.best_config.classifier_2.predict(features)
        except Exception as e:
            print("Error for individual with following configuration")
            self.best_config.print_details()
            print(e)
