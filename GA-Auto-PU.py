from os import cpu_count
import random as rand
from numpy.lib.function_base import average

from joblib import Parallel, delayed

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from deepforest import CascadeForestClassifier

from statistics import stdev

import pandas as pd
import numpy as np

import copy

import sys
import traceback

import warnings


RANDOM_STATE = 42

rand.seed(RANDOM_STATE)

iteration_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rn_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
classifiers = [GaussianNB(),
               RandomForestClassifier(random_state=RANDOM_STATE),
               DecisionTreeClassifier(random_state=RANDOM_STATE),
               MLPClassifier(random_state=RANDOM_STATE, verbose=0),
               make_pipeline(StandardScaler(),
                             SVC(gamma='auto', probability=True,
                                 random_state=RANDOM_STATE, verbose=0)),
               make_pipeline(StandardScaler(),
                             SGDClassifier(random_state=RANDOM_STATE,
                                           verbose=0, loss="log")),
               KNeighborsClassifier(),
               LogisticRegression(random_state=RANDOM_STATE, verbose=0),
               CascadeForestClassifier(random_state=RANDOM_STATE, verbose=0,
                                       backend="sklearn"),
               AdaBoostClassifier(random_state=RANDOM_STATE),
               GradientBoostingClassifier(random_state=RANDOM_STATE,
                                          verbose=0),
               LinearDiscriminantAnalysis(),
               ExtraTreesClassifier(random_state=RANDOM_STATE, verbose=0),
               BaggingClassifier(random_state=RANDOM_STATE, verbose=0),
               BernoulliNB(),
               ExtraTreeClassifier(random_state=RANDOM_STATE),
               GaussianProcessClassifier(random_state=RANDOM_STATE),
               HistGradientBoostingClassifier(verbose=0,
                                              random_state=RANDOM_STATE)
               ]


class Individual:
    """Class to store information of an individual."""

    def __init__(self):
            
        # initialise all values as 0 or empty
        self.avg_recall = self.avg_precision = self.std_dev_recall = self.std_dev_precision = self.avg_f_measure = self.std_dev_f_measure = self.iteration_count_1_a = self.rn_threshold_1_a = self.iteration_count_1_b = self.rn_threshold_1_b = self.fitness = 0
        self.classifier_1_a = self.classifier_1_b = self.classifier_2 = None
        self.flag_1_b = True
        self.recall_list = self.precision_list = self.f_measure_list = []

    def generate_iteration_count(self):
        """Return a random value from the list of iteration counts.

        Parameters
        ----------
        None

        Returns
        -------
        iteration_count: int
            Number of times to iterate a phase.

        """
        return rand.choice(iteration_counts)

    def generate_rn_threshold(self):
        """Return a random value from the list of rn thresholds.

        Parameters
        ----------
        None

        Returns
        -------
        rn_threshold: float
            The threshold under which an instance is considered reliably negative.

        """
        return rand.choice(rn_thresholds)

    def generate_classifier(self):
        """Return a random value from the list of classifiers.

        Parameters
        ----------
        None

        Returns
        -------
        classifier: Object
            The classifier to use in a given phase.

        """
        return rand.choice(classifiers)

    def generate_flag(self):
        """Randomly return either true or false.

        Parameters
        ----------
        None

        Returns
        -------
        flag: bool
            Whether to use phase 1 B.

        """
        return rand.choice([True, False])

    def print_details(self):
        """Print the details of the individual configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("Configuration details for individual")
        print("Phase 1 A iteration count:", self.iteration_count_1_a)
        print("Phase 1 A RN threshold:", self.rn_threshold_1_a)
        print("Phase 1 A classifier:", self.classifier_1_a)
        print("Phase 1 B flag:", self.flag_1_b)
        print("Phase 1 B iteration count:", self.iteration_count_1_b)
        print("Phase 1 B RN threshold:", self.rn_threshold_1_b)
        print("Phase 1 B classifier:", self.classifier_1_b)
        print("Phase 2 classifier:", self.classifier_2)


class Auto_PU:
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
        log_directory=None,
        random_state=None,
        n_jobs=1,
        try_next=False
    ):
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
        individual.iteration_count_1_a = \
            individual.generate_iteration_count()
        individual.rn_threshold_1_a = \
            individual.generate_rn_threshold()
        individual.classifier_1_a = \
            individual.generate_classifier()

        # generate the values for phase 1 b
        individual.iteration_count_1_b = \
            individual.generate_iteration_count()
        individual.rn_threshold_1_b = \
            individual.generate_rn_threshold()
        individual.classifier_1_b = \
            individual.generate_classifier()
        individual.flag_1_b = \
            individual.generate_flag()

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

    def phase_1_a(self, individual, X_positive, X_unlabelled):
        """Perform phase 1 a of the PU learning procedure.

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

        # initialise the rn set and unlabelled set to be returned
        # as empty sets
        rn_set = []
        new_unlabelled = []

        # if the iteration count is greater than 1, split the unlabelled set
        # into iteration_count_1_a subsets to handle the class imbalance
        # then perform classification with the smaller unlabelled set
        # and the positive set
        if individual.iteration_count_1_a > 1:

            y_unlabelled = np.zeros(X_unlabelled.shape[0])

            # stratified k fold the data
            skf = StratifiedKFold(n_splits=individual.iteration_count_1_a,
                                  random_state=RANDOM_STATE, shuffle=True)
            skf.get_n_splits(X_unlabelled, y_unlabelled)
            c = 0

            # split the unlabelled set
            # switch test and train so that we're training with the smaller set
            for _, train in skf.split(X_unlabelled, y_unlabelled):
                # create training set
                X_train, y_train = X_unlabelled[train], y_unlabelled[train]
                X_train = np.vstack([X_train, X_positive])
                y_train = np.hstack([y_train, np.ones(X_positive.shape[0])])

                # get the phase 1 a classifier, fit to the training set
                # and predict the probability of each instance in training set
                # being positive

                y_prob = \
                    self.predict_probability(individual.classifier_1_a,
                                             X_train,
                                             y_train,
                                             X_train)

                # get the rn set and the unlabelled set from this iteration
                rn, unlabelled = \
                    self.get_rn_unlabelled(y_prob,
                                           y_train,
                                           individual.rn_threshold_1_a,
                                           X_train)
                rn_set.append(rn)
                new_unlabelled.append(unlabelled)
                c+=1

        # if the iteration count is 1, get the rn set without
        # using undersampling
        else:
            # create the training set
            X_train = np.vstack([X_positive, X_unlabelled])
            y_train = np.hstack([np.ones(X_positive), y_unlabelled])

            # get the phase 1 a classifier, fit to the training set
            # and predict the probability of each instance in training set
            # being positive
            y_prob = self.predict_probability(individual.classifier_1_a,
                                              X_train,
                                              y_train,
                                              X_train)

            # get the rn set and the unlabelled set
            rn, unlabelled = \
                self.get_rn_unlabelled(y_prob,
                                       y_train,
                                       individual.rn_threshold_1_a,
                                       X_train)
            rn_set.append(rn)
            new_unlabelled.append(unlabelled)

        # flatten the rn and unlabelled lists
        rn_set = [instances for rn in rn_set for instances in rn]
        new_unlabelled = [instances for unlabelled in new_unlabelled
                          for instances in unlabelled]

        # print("RN size:", len(rn_set))

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
        y_prob = self.predict_probability(individual.classifier_1_b,
                                          X_train,
                                          y_train,
                                          X_unlabelled)

        # get the rn set and the unlabelled set
        new_rn_set, new_unlabelled = \
            self.get_rn_unlabelled(y_prob,
                                   y_unlabelled,
                                   individual.rn_threshold_1_b,
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

        # lists to store the recalls and precisions
        recall_list = []
        precision_list = []
        f_measure_list = []

        # stratified 5-fold
        skf = StratifiedKFold(n_splits=self.internal_fold_count,
                              random_state=RANDOM_STATE, shuffle=True)

        # perform 5 fold CV
        for train, test in skf.split(X, y):
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]

            unlabelled_set = [X_train[y_train == 0]]
            positive_set = [X_train[y_train == 1]]

            # perform phase one a
            rn_set, unlabelled_set = \
                self.phase_1_a(individual, positive_set, unlabelled_set)

            # ensure that phase 1 a returned a reliable negative set
            # before proceeding
            if len(rn_set) > 0:

                # perform phase 1 b if the flag is set and if there are
                # still instances left in the unlabelled set
                if individual.flag_1_b and len(unlabelled_set) > 0:
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

                # predict the class of the instances in the test set
                # with the phase 2 classifier trained on the positive
                # and the reliable negative data
                try:
                    y_pred = self.pred(individual.classifier_2,
                                       X_train,
                                       y_train,
                                       X_test)

                    # get the true positives, false positives, true negatives,
                    # and false negatives
                    TP, FP, _, FN = self.perf_measure(y_test, y_pred)

                    # calculate the precision and recall
                    recall = (TP / (TP + FN))
                    precision = (TP / (TP + FP))
                    f_measure = \
                        (2*((precision * recall) / (precision + recall)))

                except Exception as e:
                    # print(e)
                    # print(traceback.format_exc())
                    # sys.stdout.flush()
                    # print("Size of rn:", len(X_rn))
                    # print("It count 1 a:", individual.iteration_count_1_a)
                    # print("RN 1 a", individual.rn_threshold_1_a)
                    # print("Classifier 1 a:", individual.classifier_1_a)
                    # print("RN 1 b:", individual.rn_threshold_1_b)
                    # print("Classifier 1 b:", individual.classifier_1_b)
                    # print("Flag 1 b", individual.flag_1_b)
                    # print("Classifier 2:", individual.classifier_2)
                    recall = 0
                    precision = 0
                    f_measure = 0

                # print("Recall:", recall)
                # print("Precision:", precision)
                # print("F measure:", f_measure)

                # add precision and recall to the respective lists
                recall_list.append(recall)
                precision_list.append(precision)
                f_measure_list.append(f_measure)

            # if no reliable negative set was received,
            # set the precision and recall to 0
            else:
                # print("No reliable negative set found.")
                recall_list.append(0)
                precision_list.append(0)
                f_measure_list.append(0)

        # print(recall_list)
        # print(precision_list)
        # print(f_measure_list)

        return recall_list, precision_list, f_measure_list

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

        # initialise recall and precision as 0
        avg_recall = 0
        avg_precision = 0
        avg_f_measure = 0

        # copy all values and add to config list
        iteration_count_1_a = copy.deepcopy(individual.iteration_count_1_a)
        rn_threshold_1_a = copy.deepcopy(individual.rn_threshold_1_a)
        classifier_1_a = copy.deepcopy(individual.classifier_1_a)
        rn_threshold_1_b = copy.deepcopy(individual.rn_threshold_1_b)
        classifier_1_b = copy.deepcopy(individual.classifier_1_b)
        flag_1_b = copy.deepcopy(individual.flag_1_b)
        classifier_2 = copy.deepcopy(individual.classifier_2)
        config = [iteration_count_1_a, rn_threshold_1_a, classifier_1_a,
                  rn_threshold_1_b, classifier_1_b, flag_1_b, classifier_2]

        # initialise found as false
        found = False

        # search for current config in list of assessed configs
        # if found, set the recall and precision to the previously
        # calculated values
        for i in range(len(assessed)):
            if str(config) == assessed[i][0]:
                found = True
                individual.avg_recall = assessed[i][1]
                individual.avg_precision = assessed[i][2]
                individual.avg_f_measure = assessed[i][3]
                individual.std_dev_recall = assessed[i][4]
                individual.std_dev_precision = assessed[i][5]
                individual.std_dev_f_measure = assessed[i][6]

        # if the configuration is not found, assess it
        if not found:
            try:
                # get the recalls and precisions from the
                # internal cross validation
                recall_list, precision_list, f_measure_list \
                    = self.internal_CV(individual, features, target)
                avg_recall = average(recall_list)
                avg_precision = average(precision_list)
                avg_f_measure = average(f_measure_list)
                std_dev_recall = stdev(recall_list)
                std_dev_precision = stdev(precision_list)
                std_dev_f_measure = stdev(f_measure_list)

                individual.avg_recall = avg_recall
                individual.avg_precision = avg_precision
                individual.avg_f_measure = avg_f_measure
                individual.std_dev_recall = std_dev_recall
                individual.std_dev_precision = std_dev_precision
                individual.std_dev_f_measure = std_dev_f_measure
                individual.recall_list = recall_list
                individual.precision_list = precision_list
                individual.f_measure_list = f_measure_list

                #print("Recall: ", individual.avg_recall)

                # print(avg_recall)
                # print(avg_precision)
                # print(avg_f_measure)
                # print(std_dev_recall)
                # print(std_dev_precision)
                # print(std_dev_f_measure)
                # print(recall_list)
                # print(precision_list)
                # print(f_measure_list)

            except Exception as e:
                print("Error for individual: ",
                      individual.iteration_count_1_a,
                      individual.rn_threshold_1_a,
                      individual.classifier_1_a,
                      individual.rn_threshold_1_b,
                      individual.classifier_1_b,
                      individual.flag_1_b,
                      individual.classifier_2)
                print(e)
                print(traceback.format_exc())
                sys.stdout.flush()
                avg_recall = 0
                avg_precision = 0
                avg_f_measure = 0
                std_dev_recall = 0
                std_dev_precision = 0
                std_dev_f_measure = 0

            # new_assessed_configs.append([str(config),
            #                              avg_recall,
            #                              avg_precision,
            #                              avg_f_measure,
            #                              std_dev_recall,
            #                              std_dev_precision,
            #                              std_dev_f_measure])

            

        sys.stdout.flush()

        # previously returned new_assessed_configs, but should just be able to return the individual bc new_assessed_configs did not contain previous
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
        avg_precision = 0
        avg_recall = 0
        avg_f_measure = 0
        avg_std_precision = 0
        avg_std_recall = 0
        avg_std_f_measure = 0

        pop_len = len(population)

        # for every individual in the population
        # add values to total
        for individual in population:
            avg_precision += individual.avg_precision
            avg_recall += individual.avg_recall
            avg_f_measure += individual.avg_f_measure
            avg_std_precision += individual.std_dev_precision
            avg_std_recall += individual.std_dev_recall
            avg_std_f_measure += individual.std_dev_f_measure

        # get the average of the values
        avg_precision = (avg_precision / pop_len)
        avg_recall = (avg_recall / pop_len)
        avg_f_measure = (avg_f_measure / pop_len)
        avg_std_precision = (avg_std_precision / pop_len)
        avg_std_recall = (avg_std_recall / pop_len)
        avg_std_f_measure = (avg_std_f_measure / pop_len)

        return avg_precision, avg_recall, avg_f_measure, \
            avg_std_precision, avg_std_recall, avg_std_f_measure

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

            new_indiv_f_measure_diff = \
                self.get_total_difference(population[i].f_measure_list,
                                          population[i].avg_f_measure)

            # get the f_measure difference for the new individual
            new_indiv_f_measure_diff = \
                self.get_total_difference(population[i].f_measure_list,
                                          population[i].avg_f_measure)

            # get the f_measure difference for the fittest individual
            current_fittest_f_measure_diff = \
                self.get_total_difference(fittest_individual.f_measure_list,
                                          fittest_individual.avg_f_measure)

            # get the pooled standard deviation for the f_measure
            f_measure_std_pooled = \
                self.get_pooled_std_dev(new_indiv_f_measure_diff,
                                        current_fittest_f_measure_diff)

            # calculate the f_measure difference
            try:
                f_measure_difference = abs((population[i].avg_f_measure -
                                            fittest_individual.avg_f_measure) /
                                           f_measure_std_pooled)
            except:
                f_measure_difference = 0

            # if f_measure difference is insignificant, test recall
            if f_measure_difference < self.fitness_threshold:

                # get the recall difference for the new individual
                new_indiv_recall_diff = \
                    self.get_total_difference(population[i].recall_list,
                                              population[i].avg_recall)

                # get the recall difference for the fittest individual
                current_fittest_recall_diff = \
                    self.get_total_difference(fittest_individual.recall_list,
                                              fittest_individual.avg_recall)

                # get the pooled standard deviation for the recall
                recall_std_pooled = \
                    self.get_pooled_std_dev(new_indiv_recall_diff,
                                            current_fittest_recall_diff)

                # calculate the recall difference
                try:
                    recall_difference = \
                        abs((population[i].avg_recall -
                             fittest_individual.avg_recall) /
                            recall_std_pooled)
                except:
                    recall_difference = 0

                # if recall difference is insignificant,
                # individual with highest f_measure is fittest
                if recall_difference < self.fitness_threshold:

                    # if new individuals f_measure is higher than the
                    # fittest individual, new individual becomes fittest
                    if population[i].avg_f_measure > \
                            fittest_individual.avg_f_measure:
                        fittest_individual = population[i]
                        f += 1
                    else:
                        f += 1

                # if new individual recall is higher than fittest individual
                # new individual becomes fittest individual
                elif population[i].avg_recall > \
                        fittest_individual.avg_recall:
                    fittest_individual = population[i]
                    rec += 1
                else:
                    rec += 1

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            elif population[i].avg_f_measure > \
                    fittest_individual.avg_f_measure:
                fittest_individual = population[i]
                f += 1
            else:
                f += 1

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
            iteration_count_1_a = str(individual.iteration_count_1_a)
            rn_threshold_1_a = str(individual.rn_threshold_1_a)
            classifier_1_a = str(individual.classifier_1_a)
            rn_threshold_1_b = str(individual.rn_threshold_1_b)
            classifier_1_b = str(individual.classifier_1_b)
            flag_1_b = str(individual.flag_1_b)
            classifier_2 = str(individual.classifier_2)
            recall = str(individual.avg_recall)
            precision = str(individual.avg_precision)
            f_measure = str(individual.avg_f_measure)

            indiv_detail = [iteration_count_1_a, rn_threshold_1_a,
                            classifier_1_a, rn_threshold_1_b,
                            classifier_1_b, flag_1_b, classifier_2,
                            recall, precision, f_measure]

            individual_details.append(indiv_detail)

        # column names
        col = ["Iteration Count P1A", "RN Threshold P1A", "Classifier P1A",
               "RN Threshold P1B", "Classifier P1B", "Flag P1B",
               "Classifier P2", "Recall", "Precision", "F measure"]

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
            print(e)
            print(traceback.format_exc())

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
            new_indiv1.iteration_count_1_a = \
                copy.deepcopy(new_indiv2.iteration_count_1_a)
            new_indiv2.iteration_count_1_a = \
                copy.deepcopy(new_indiv1.iteration_count_1_a)

        # swap phase 1 a rn threshold
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.rn_threshold_1_a = \
                copy.deepcopy(new_indiv2.rn_threshold_1_a)
            new_indiv2.rn_threshold_1_a = \
                copy.deepcopy(new_indiv1.rn_threshold_1_a)

        # swap phase 1 a classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_1_a = \
                copy.deepcopy(new_indiv2.classifier_1_a)
            new_indiv2.classifier_1_a = \
                copy.deepcopy(new_indiv1.classifier_1_a)

        # swap phase 1 b iteration count
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.iteration_count_1_b = \
                copy.deepcopy(new_indiv2.iteration_count_1_b)
            new_indiv2.iteration_count_1_b = \
                copy.deepcopy(new_indiv1.iteration_count_1_b)

        # swap phase 1 b rn threshold
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.rn_threshold_1_b = \
                copy.deepcopy(new_indiv2.rn_threshold_1_b)
            new_indiv2.rn_threshold_1_b = \
                copy.deepcopy(new_indiv1.rn_threshold_1_b)

        # swap phase 1 b classifier
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.classifier_1_b = \
                copy.deepcopy(new_indiv2.classifier_1_b)
            new_indiv2.classifier_1_b = \
                copy.deepcopy(new_indiv1.classifier_1_b)

        # swap phase 1 b flag
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            new_indiv1.flag_1_b = \
                copy.deepcopy(new_indiv2.flag_1_b)
            new_indiv2.flag_1_b = \
                copy.deepcopy(new_indiv1.flag_1_b)

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

        # counters to store the deciding factors from tournament selection
        f = 0
        rec = 0

        # keep performing crossover until the new population
        # is the correct size
        while len(new_population) < self.population_size - 1:

            # select two individuals with tournament selection
            indiv1, f_1, rec_1 = \
                self.tournament_selection(population)
            indiv2, f_2, rec_2 = \
                self.tournament_selection(population)

            # add values to counters
            f += f_1
            f += f_2
            rec += rec_1
            rec += rec_2

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
        return new_population, f, rec

    def mutate_iteration_count(self, iteration_count):
        """Mutate phase 1 a/b iteration count.

        Parameters
        ----------
        iteration_count: int
            The iteration count to be mutated.

        Returns
        -------
        iteration_count: int
            The iteration count after mutation.

        """

        # if iteration_count is between 1 and 10, either add
        # or subtract
        if iteration_count > 1 and iteration_count < 10:
            if rand.uniform(0, 1) < 0.5:
                iteration_count += 1
            else:
                iteration_count -= 1

        # if iteration count is 1, add 1
        elif iteration_count == 1:
            iteration_count += 1

        # if iteration count is 10, minus 1
        elif iteration_count == 10:
            iteration_count -= 1

        # return modified iteration count
        return iteration_count

    def mutate_rn_threshold(self, rn_threshold):
        """Mutate phase 1 a/b rn threshold.

        Parameters
        ----------
        rn_threshold: float
            The RN threshold to mutate.

        Returns
        -------
        rn_threshold: float
            The RN threshold after mutation.

        """

        # if threhsold is between 0.05 and 0.5, either add
        # or subtract
        if rn_threshold > 0.05 and rn_threshold < 0.5:
            if rand.uniform(0, 1) < 0.5:
                rn_threshold = round(rn_threshold + 0.05, 3)
            else:
                rn_threshold = round(rn_threshold - 0.05, 3)

        # if threshold is 0.05, add 0.05
        elif rn_threshold == 0.05:
            rn_threshold = round(rn_threshold + 0.05, 3)

        # if threshold is 0.5, subtract 0.05
        elif rn_threshold == 0.5:
            rn_threshold = round(rn_threshold - 0.05, 3)

        # return modified rn threshold
        return rn_threshold

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
        for individual in population:

            # mutate phase 1 a iteration count
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.iteration_count_1_a = \
                    self.mutate_iteration_count(individual.iteration_count_1_a)

            # mutate phase 1 a rn threshold
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.rn_threshold_1_b = \
                    self.mutate_rn_threshold(individual.rn_threshold_1_b)

            # mutate phase 1 a classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.classifier_1_a = rand.choice(classifiers)

            # mutate phase 1 b iteration count
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.iteration_count_1_b = \
                    self.mutate_iteration_count(individual.iteration_count_1_b)

            # mutate phase 1 b rn threshold
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.rn_threshold_1_b = \
                    self.mutate_rn_threshold(individual.rn_threshold_1_b)

            # mutate phase 1 b classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.classifier_1_b = rand.choice(classifiers)

            # mutate phase 1 b flag
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.flag_1_b = rand.choice([True, False])

            # mutate phase 2 classifier
            if rand.uniform(0, 1) < self.mutation_prob:
                individual.classifier_2 = rand.choice(classifiers)

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
        iteration_count_1_a = str(fittest_individual.iteration_count_1_a)
        rn_threshold_1_a = str(fittest_individual.rn_threshold_1_a)
        classifier_1_a = str(clone(fittest_individual.classifier_1_a))
        rn_threshold_1_b = str(fittest_individual.rn_threshold_1_b)
        classifier_1_b = str(clone(fittest_individual.classifier_1_b))
        flag_1_b = str(fittest_individual.flag_1_b)
        classifier_2 = str(clone(fittest_individual.classifier_2))

        # add the values to a list showing the best configuration
        best_configuration = [iteration_count_1_a, rn_threshold_1_a,
                              classifier_1_a, rn_threshold_1_b, classifier_1_b,
                              flag_1_b, classifier_2]

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
                      "Standard deviation recall",
                      "Standard deviation precision",
                      "Standard deviation f measure",
                      ""]

        # second column of combined stats file denotes the
        # recall and precision of the fittest individual
        recall = str(fittest_individual.avg_recall)
        precision = str(fittest_individual.avg_precision)
        f_measure = str(fittest_individual.avg_f_measure)
        std_dev_recall = str(fittest_individual.std_dev_recall)
        std_dev_precision = str(fittest_individual.std_dev_precision)
        std_dev_f_measure = str(fittest_individual.std_dev_f_measure)
        best_stats = [recall, precision, f_measure, std_dev_recall,
                      std_dev_precision, std_dev_f_measure, ""]

        # third columns denotes the average population recall and precision
        avg_stats = [str(avg_recall), str(avg_precision), str(avg_f_measure),
                     str(avg_std_dev_recall), str(avg_std_dev_precision),
                     str(avg_std_dev_f_measure), ""]

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
                                    " stats fitness threshold " +
                                    str(self.fitness_threshold) +
                                    ".csv", index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory +
                  " Generation " + str(current_generation) +
                  " stats fitness threshold " +
                  str(self.fitness_threshold) + ".csv")
            print(e)
            print(traceback.format_exc())

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

        # lists to store how fittest individual was defined
        # e.g., the number of time recall was the most important factor
        recall_decs = []
        f_decs = []

        # lists to store how fittest individual was defined in crossover
        # e.g., the number of time recall was the most important factor
        cross_f_decs = []
        cross_recall_decs = []

        # start the evolution process
        while current_generation < self.generation_count:
            # assess the fitness of individuals concurrently
            # the ray.get function calls the assess_fitness function with
            # the parameters specified in remote()
            # this will call the function for all individuals in population

            population = \
                Parallel(n_jobs=self.n_jobs,
                         verbose=10)(delayed(self.assess_fitness)
                                     (individual=population[i],
                                      features=features,
                                      target=target, assessed=assessed_configs)
                                     for i in range(len(population)))
            
            #population = [self.assess_fitness(population[i], features, target, assessed_configs) for i in range(len(population))]

            for i in range(len(population)):
                description = str([population[i].iteration_count_1_a,
                population[i].rn_threshold_1_a,
                population[i].classifier_1_a,
                population[i].rn_threshold_1_b,
                population[i].classifier_1_b,
                population[i].flag_1_b,
                population[i].classifier_2])

                if description not in assessed_configs:
                    # add config to assessed list
                    assessed_configs.append([description, 
                                            population[i].avg_recall, 
                                            population[i].avg_precision, 
                                            population[i].avg_f_measure,
                                            population[i].std_dev_recall,
                                            population[i].std_dev_precision,
                                            population[i].std_dev_f_measure])

            # calculate average precision and recall
            avg_precision, avg_recall, avg_f_measure, avg_std_dev_precision, \
                avg_std_dev_recall, avg_std_dev_f_measure \
                = self.get_avg_prec_rec(population)

            # get the fittest individual in the population so that it can
            # be preserved without modification
            # also get the number of decisions made by f-measure and recall
            fittest_individual, f, rec = self.get_fittest(population)

            if self.log_directory is not None:
                # save the details of all individuals in population
                # for this generation
                self.log_individuals(population, current_generation)

                # save the decision values for logging
                f_decs.append(f)
                recall_decs.append(rec)

            if current_generation < self.generation_count-1:

                # remove the fittest individual from the population
                population.remove(fittest_individual)

                # perform crossover
                population, c_f, c_rec = self.crossover(population)

                # perform mutation
                population = self.mutate(population)

                # add the fittest individual back into the population
                population.append(fittest_individual)

            if self.log_directory is not None:
                # add crossover counters to respective lists
                cross_f_decs.append(c_f)
                cross_recall_decs.append(c_rec)
                # save the statistics of this generation to a csv file
                self.log_generation_stats(fittest_individual,
                                          avg_recall,
                                          avg_precision,
                                          avg_f_measure,
                                          avg_std_dev_recall,
                                          avg_std_dev_precision,
                                          avg_std_dev_f_measure,
                                          current_generation)

            print("Generation", current_generation, "complete.")

            # display any output from this generation
            sys.stdout.flush()

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

        if self.try_next:
            count = len(population)
        else:
            count = 1

        for _ in range(count):

            try:

                fittest_individual, _, _ = self.get_fittest(population)

                self.best_config = fittest_individual

                print("Best configuration")
                self.best_config.print_details()

                population.remove(fittest_individual)

                # perform phase one a
                rn_set, unlabelled_set = \
                    self.phase_1_a(self.best_config,
                                   positive_set,
                                   unlabelled_set)

                # ensure that phase 1 a returned a reliable negative set
                # before proceeding
                if len(rn_set) > 0:

                    # perform phase 1 b if the flag is set and if there are
                    # still instances left in the unlabelled set
                    if self.best_config.flag_1_b and len(unlabelled_set) > 0:
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
                    self.best_config.classifier_2.fit(X_train, y_train)

                    break

            except Exception as e:
                print("Evolved individual was unable to be trained on full \
                      training set.")
                print("It is likely that the individual was overfit during \
                      the evolution process.")
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