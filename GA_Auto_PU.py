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
from Classifiers import Classifiers


class Auto_PU:
    def __init__(
        self,
        population_size=101,
        generation_count=50,
        mutation_prob=0.1,
        crossover_prob=0.9,
        gene_crossover_prob=0.5,
        tournament_size=2,
        internal_fold_count=5,
        spies=False,
        log_directory=None,
        random_state=None,
        n_jobs=1):

        """Initialise the Auto_PU genetic algorithm.

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
        internal_fold_count: int, optional (default: 5)
            The number of folds for internal cross validation.
        spies: boolean, optional (default: False)
            Whether to allow spy-based individuals.
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
        self.internal_fold_count = internal_fold_count
        self.spies = spies
        self.log_directory = log_directory
        self.random_state = random_state

        if n_jobs == 0:
            raise ValueError("The value of n_jobs cannot be 0.")
        elif n_jobs < 0:
            self.n_jobs = cpu_count() + 1 + n_jobs
        else:
            self.n_jobs = n_jobs

        self.best_config = None

        classifiers = Classifiers()
        self.classifiers = classifiers.classifiers

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

    def get_rn_unlabelled(self, y_prob, y_actual, rn_thresh, X):
        """For each negative instance, check if the predicted probability
        is lower than the rn threshold. If so, add to the rn set, otherwise,
        add to the unlabelled set.

        Parameters
        ----------
        y_prob: array-like {n_samples}
            Class probability predictions.
        y_actual: array-like {n_samples}
            Actual labelled class values.
        rn_thresh: float
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

        rn_set = [X[i] for i in range(len(X)) if y_actual[i] == 0 and y_prob[i] < rn_thresh]
        new_unlabelled = [X[i] for i in range(len(X)) if y_actual[i] == 0 and y_prob[i] >= rn_thresh]

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

    def phase_1b(self, individual, positive_set, rn_set, unlabelled_set):
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
        # unlabelled class
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
        f_measure: float
            The F-measure value achieved by the individual
        """

        # stratified 5-fold
        skf = StratifiedKFold(n_splits=self.internal_fold_count,
                              random_state=self.random_state, shuffle=True)

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
                            self.phase_1b(individual, positive_set, rn_set,
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

            except:
                # if the individual generates an error, fitness for
                # individual is 0
                return 0

        individual.tp = tp
        individual.fp = fp
        individual.tn = tn
        individual.fn = fn

        try:
            recall = (tp/(tp+fn))
            precision = (tp/(tp+fp))
            return (2*((precision * recall)/(precision+recall)))
        except:
            return 0

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

        # for every individual in the population
        # add values to total
        for individual in population:
            avg_f_measure += individual.fitness
        # get the average of the values
        avg_f_measure = (avg_f_measure / len(population))

        return avg_f_measure

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

        # compare every individual in population
        for i in range(len(population)):

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if population[i].fitness > \
                    fittest_individual.fitness:
                fittest_individual = population[i]

        # return the fittest individual and the counters
        return fittest_individual

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
            spies = str(individual.spies)
            spy_rate = str(individual.spy_rate)
            spy_tolerance = str(individual.spy_tolerance)
            f_measure = str(individual.fitness)
            tp = str(individual.tp)
            fp = str(individual.fp)
            tn = str(individual.tn)
            fn = str(individual.fn)

            indiv_detail = [it_count_1a, rn_thresh_1a,
                            classifier_1a, rn_thresh_1b,
                            classifier_1b, flag_1b, classifier_2,
                            spies, spy_rate, spy_tolerance,
                            tp, fp, tn, fn, f_measure]

            individual_details.append(indiv_detail)

        # column names
        col = ["Iteration Count P1A", "RN Threshold P1A", "Classifier P1A",
               "RN Threshold P1B", "Classifier P1B", "Flag P1B",
               "Classifier P2", "Spies", "Spy_rate", "Spy_tolerance", "TP",
               "FP", "TN", "FN", "F measure"]

        # create dataframe
        individuals_df = pd.DataFrame(individual_details, columns=[col])

        try:
            # save to csv
            individuals_df.to_csv(self.log_directory +
                                  " Generation " +
                                  str(current_generation) +
                                  " individual details.csv",
                                  index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory +
                  " Generation " + str(current_generation) +
                  " individual details.csv")

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
            new_indiv1.it_count_1b = \
                copy.deepcopy(new_indiv2.it_count_1b)
            new_indiv2.it_count_1b = \
                copy.deepcopy(new_indiv1.it_count_1b)

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

        # swap spy bool
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.spies = \
                copy.deepcopy(new_indiv2.spies)
            new_indiv2.spies = \
                copy.deepcopy(new_indiv1.spies)

        # swap spy rate
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.spy_rate = \
                copy.deepcopy(new_indiv2.spy_rate)
            new_indiv2.spy_rate = \
                copy.deepcopy(new_indiv1.spy_rate)

        # swap spy tolerance
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.spy_tolerance = \
                copy.deepcopy(new_indiv2.spy_tolerance)
            new_indiv2.spy_tolerance = \
                copy.deepcopy(new_indiv1.spy_tolerance)

        # swap phase 2 classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_2 = \
                copy.deepcopy(new_indiv2.classifier_2)
            new_indiv2.classifier_2 = \
                copy.deepcopy(new_indiv1.classifier_2)

        # return the new individuals
        return new_indiv1, new_indiv2

    def crossover(self, population):
        """Perform crossover on the population.
        Individuals are selected through tournament selection
        and their genes are swapped with a given probability.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        new_population: array-like {population_size}
            The list of individuals after undergoing crossover.
        f: int
            The number of individuals deemed to be fitter
            than the comparison by f-measure.
        rec: int
            The number of individuals deemed to be fitter
            than the comparison by recall.

        """

        # empty list to store the modified population
        new_population = []

        # keep performing crossover until the new population
        # is the correct size
        while len(new_population) < self.population_size - 1:

            # select two individuals with tournament selection
            indiv1 = \
                self.tournament_selection(population)
            indiv2 = \
                self.tournament_selection(population)

            # initialise the new individuals
            new_indiv1 = indiv1
            new_indiv2 = indiv2

            # if random number is less than crossover probability,
            # the individuals undergo crossover
            if rand.uniform(0, 1) < self.crossover_prob:
                new_indiv1, new_indiv2 = \
                    self.swap_genes(new_indiv1, new_indiv2)

            # add the new individuals to the population
            new_population.append(new_indiv1)
            new_population.append(new_indiv2)

        # return the new population and the counters
        return new_population

    def mutate_it_count(self, it_count):
        """Mutate phase 1 a/b iteration count.

        Parameters
        ----------
        it_count: int
            The iteration count to be mutated.

        Returns
        -------
        it_count: int
            The iteration count after mutation.

        """

        # if it_count is between 1 and 10, either add
        # or subtract
        if it_count > 1 and it_count < 10:
            if rand.uniform(0, 1) < 0.5:
                it_count += 1
            else:
                it_count -= 1

        # if iteration count is 1, add 1
        elif it_count == 1:
            it_count += 1

        # if iteration count is 10, minus 1
        elif it_count == 10:
            it_count -= 1

        # return modified iteration count
        return it_count

    def mutate_rn_thresh(self, rn_thresh):
        """Mutate phase 1 a/b rn threshold.

        Parameters
        ----------
        rn_thresh: float
            The RN threshold to mutate.

        Returns
        -------
        rn_thresh: float
            The RN threshold after mutation.

        """

        # if threhsold is between 0.05 and 0.5, either add
        # or subtract
        if rn_thresh > 0.05 and rn_thresh < 0.5:
            if rand.uniform(0, 1) < 0.5:
                rn_thresh = round(rn_thresh + 0.05, 3)
            else:
                rn_thresh = round(rn_thresh - 0.05, 3)

        # if threshold is 0.05, add 0.05
        elif rn_thresh == 0.05:
            rn_thresh = round(rn_thresh + 0.05, 3)

        # if threshold is 0.5, subtract 0.05
        elif rn_thresh == 0.5:
            rn_thresh = round(rn_thresh - 0.05, 3)

        # return modified rn threshold
        return rn_thresh

    def mutate_spy_rate(self, spy_rate):
        """Mutate spy rate.

        Parameters
        ----------
        rn_thresh: float
            The spy rate to mutate.

        Returns
        -------
        rn_thresh: float
            The spy rate after mutation.

        """

        # if spy_rate is between 0.05 and 0.35, either add
        # or subtract
        if spy_rate > 0.05 and spy_rate < 0.35:
            if rand.uniform(0, 1) < 0.5:
                spy_rate = round(spy_rate + 0.05, 3)
            else:
                spy_rate = round(spy_rate - 0.05, 3)

        # if spy_rate is 0.05, add 0.05
        elif spy_rate == 0.05:
            spy_rate = round(spy_rate + 0.05, 3)

        # if spy_rate is 0.5, subtract 0.05
        elif spy_rate == 0.35:
            spy_rate = round(spy_rate - 0.05, 3)

        # return modified spy_rate
        return spy_rate

    def mutate_spy_tolerance(self, spy_tolernace):
        """Mutate spy tolerance.

        Parameters
        ----------
        rn_thresh: float
            The spy tolerance to mutate.

        Returns
        -------
        rn_thresh: float
            The spy tolerance after mutation.

        """

        # if spy_tolernace is between 0 and 0.1, either add
        # or subtract
        if spy_tolernace > 0 and spy_tolernace < 0.1:
            if rand.uniform(0, 1) < 0.5:
                spy_tolernace = round(spy_tolernace + 0.01, 3)
            else:
                spy_tolernace = round(spy_tolernace - 0.01, 3)

        # if spy_tolernace is 0, add 0.01
        elif spy_tolernace == 0:
            spy_tolernace = round(spy_tolernace + 0.01, 3)

        # if spy_tolernace is 0.1, subtract 0.01
        elif spy_tolernace == 0.1:
            spy_tolernace = round(spy_tolernace - 0.01, 3)

        # return modified spy_tolernace
        return spy_tolernace

    def mutate(self, population):
        """Perform mutation on the population.
        Each gene is slightly altered with a given probability.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        population: array-like {population_size}
            The list of individuals after undergoing mutation.

        """

        # perform on every individual in population
        for i, indiv in enumerate(population):

            # mutate phase 1 a iteration count
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_it_count_1a()

            # mutate phase 1 a rn threshold
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_rn_thresh_1a()

            # mutate phase 1 a classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i].classifier_1a = rand.choice(self.classifiers)

            # mutate phase 1 b iteration count
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_it_count_1b()

            # mutate phase 1 b rn threshold
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_rn_thresh_1b()

            # mutate phase 1 b classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i].classifier_1b = rand.choice(self.classifiers)

            # mutate phase 1 b flag
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i].flag_1b = rand.choice([True, False])

            # mutate spy bool
            if self.spies:
                if rand.uniform(0, 1) < self.mutation_prob:
                    population[i].spies = rand.choice([True, False])

            # mutate spy_rate
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_spy_rate()

            # mutate spy_tolerance
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i] = indiv.mutate_spy_tolerance()

            # mutate phase 2 classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                population[i].classifier_2 = rand.choice(self.classifiers)
        
        # return the altered population
        return population

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
        spies = str(fittest_individual.spies)
        spy_rate = str(fittest_individual.spy_rate)
        spy_tolerance = str(fittest_individual.spy_tolerance)

        # add the values to a list showing the best configuration
        best_configuration = [it_count_1a, rn_thresh_1a,
                              classifier_1a, rn_thresh_1b, classifier_1b,
                              flag_1b, classifier_2, spies, spy_rate, spy_tolerance]

        return best_configuration

    def log_generation_stats(self, fittest_individual,
                             avg_f_measure, current_generation):
        """Save all statistics from a generation to csv file.

        Parameters
        ----------
        fittest_individual: Individual
            The fittest individual from the population.
        avg_f_measure: float
            Average f-measure of all individuals in the population.
        current_generation: int
            The current generation.

        Returns
        -------
        None

        """

        # first column of the combined stats file
        # multiple blank values are needed as all columns
        # must have the same number of values
        stat_names = ["F measure",
                      "tp",
                      "fp",
                      "tn",
                      "fn", "", "", "", "", "",]

        # second column of combined stats file denotes the
        # recall and precision of the fittest individual
        f_measure = str(fittest_individual.fitness)
        tp = str(fittest_individual.tp)
        fp = str(fittest_individual.fp)
        tn = str(fittest_individual.tn)
        fn = str(fittest_individual.fn)
        best_stats = [f_measure, tp, fp, tn, fn, "", "", "", "", ""]

        avg_stats = [str(avg_f_measure),"", "", "", "", "", "", "", "", ""]

        # the final two columns give details of the best configuration
        # these are the configuration component names
        config_names = ["Iteration Count P1A", "RN Threshold P1A",
                        "Classifier P1A", "RN Threshold P1B", "Classifier P1B",
                        "Flag P1B", "Classifier P2", "Spies", "Spy rate", "Spy tolerance"]

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
        except:
            print("Could not save file:", self.log_directory +
                  " Generation " + str(current_generation) +
                  " stats.csv")

    def fit(self, features, target):
        """Use a genetic algorithm to fit an optimised PU learning
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

        # start the evolution process
        while current_generation < self.generation_count:
            # assess the fitness of individuals concurrently
            # the ray.get function calls the assess_fitness function with
            # the parameters specified in remote()
            # this will call the function for all individuals in population

            start_time = timer()

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
                population[i].rn_thresh_1b,
                population[i].classifier_1b,
                population[i].flag_1b,
                population[i].classifier_2,
                population[i].spies,
                population[i].spy_rate,
                population[i].spy_tolerance])

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

            # get the fittest individual in the population so that it can
            # be preserved without modification
            # also get the number of decisions made by f-measure and recall
            fittest_individual = self.get_fittest(population)

            if self.log_directory is not None:
                # save the details of all individuals in population
                # for this generation
                self.log_individuals(population, current_generation)

            if current_generation < self.generation_count-1:

                # remove the fittest individual from the population
                population.remove(fittest_individual)

                # perform crossover
                population = self.crossover(population)

                # perform mutation
                population = self.mutate(population)

                # add the fittest individual back into the population
                population.append(fittest_individual)

                if self.log_directory is not None:
                    # save the statistics of this generation to a csv file
                    self.log_generation_stats(fittest_individual,
                                              avg_f_measure,
                                              current_generation)                     

            end_time = timer()

            print("Generation", current_generation, "complete, time taken:", timedelta(seconds=end_time-start_time))


            # display any output from this generation
            sys.stdout.flush()

            # increment generation count
            current_generation += 1

        target = np.array(target, subok=True, ndmin=2).T

        training_set = np.concatenate([features, target], axis=1)

        # get positive set and initial unlabelled set
        unlabelled_set = training_set[training_set[:, -1] == 0]
        positive_set = training_set[training_set[:, -1] == 1]

        try:
            fittest_individual = self.get_fittest(population)

            self.best_config = fittest_individual

            print("Best configuration")
            self.best_config.print_details()

            population.remove(fittest_individual)

            # perform phase one a
            rn_set, new_unlabelled_set = \
                self.phase_1a(self.best_config,
                            positive_set,
                            unlabelled_set)

            # perform phase 1 b if the flag is set and if there are
            # still instances left in the unlabelled set
            if self.best_config.flag_1b and len(new_unlabelled_set) > 0:
                rn_set, unlabelled_set = \
                    self.phase_1b(self.best_config,
                                    positive_set,
                                    rn_set,
                                    new_unlabelled_set)

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

        except Exception as e:

            print("Evolved individual was unable to be trained on full \
                    training set.")
            print("It is likely that the individual was overfit during \
                    the evolution process.")
            print("Try again with different parameters, such as a \
                    higher number of individuals or generations.")
            print("For debugging, the exception is printed below.")
            print(e)
            print("Traceback:", traceback.format_exc())

        return self

    def predict(self, features):
        """Perform mutation on the population.
        Each gene is slightly altered with a given probability.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            The features of the dataset of which to predict the class.

        Returns
        -------
        predictions: ndarray {n_samples}
            The array of predicted classes for all samples in features.

        """

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
            print(traceback.format_exc())