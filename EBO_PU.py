import random as rand
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import copy
import sys
import traceback
import warnings

from os import cpu_count
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from deepforest import CascadeForestClassifier

from sklearn.exceptions import NotFittedError

from Individual import Individual
from Classifiers import Classifiers

class EBO_Auto_PU:

    def __init__(
        self,
        it_count=50,
        tourn_size_a=2,
        tourn_size_b=5,
        k_pop_size=9,
        mutation_prob=0.1,
        crossover_prob=0.9,
        comp_crossover_prob=0.5,
        pop_count=101,
        internal_fold_count=5,
        spies=False,
        random_state=42,
        n_jobs=-1,
        log_dir=None):

        # the number of iterations for which to perform optimisation
        self.it_count = it_count
        
        # the number of candidate solutions selected for the first
        # tournament selection
        self.tourn_size_a = tourn_size_a
        
        # the number of candidate solutions selected for the second
        # tournament selection
        self.tourn_size_b = tourn_size_b

        # the number of candidate solutions selected to be added to the 
        # population each iteration
        self.k_pop_size = k_pop_size

        # the probability of a component of a candidate solution
        # undergoing mutation
        self.mutation_prob = mutation_prob

        # the probability of two canddiate solutions undergoing crossover
        self.crossover_prob = crossover_prob

        # the probability of two canddiate solutions undergoing crossover
        self.comp_crossover_prob = comp_crossover_prob

        # the number of candidate solutions in the population
        self.pop_count = pop_count

        # the number of folds to use for internal cross validation
        self.internal_fold_count = internal_fold_count

        # whether to use spies
        self.spies = spies

        # the random state used throughout
        self.random_state = random_state

        # set the random state of rand and np
        rand.seed(self.random_state)
        np.random.seed(self.random_state)

        if n_jobs == 0:
            raise ValueError("The value of n_jobs cannot be 0.")
        elif n_jobs < 0:
            self.n_jobs = cpu_count() + 1 + n_jobs
        else:
            self.n_jobs = n_jobs

        self.best_config = None
        
        self.log_dir = log_dir

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
        if self.spies:
            individual.spies = \
                individual.generate_bool() 
        else:
            individual.spies = False
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
        population = [self.generate_individual() for _ in range(self.pop_count)]

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
            # into iteration_count_1a subsets to handle the class imbalance
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
    
    def internal_CV(self, cand_solution, features, target):
        """Perform internal cross validation to get fitness.

        Parameters
        ----------
        cand_solution: Individual
            The configuration to be assessed.
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            Class values for feature matrix.

        Returns
        -------
        f_measure: float
            The F-measure value achieved by the cand_solution
        """

        # stratified 5-fold
        skf = StratifiedKFold(n_splits=self.internal_fold_count,
                              random_state=self.random_state, shuffle=True)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # perform 5 fold CV
        for train, test in skf.split(features, target):
            
            try:
                X_train, y_train = np.array(features[train]), np.array(target[train])
                X_test, y_test = features[test], target[test]
                training_set = np.c_[X_train, y_train]

                unlabelled_set = np.array([x for x in training_set if x[-1] == 0])
                positive_set = np.array([x for x in training_set if x[-1] == 1])

                # perform phase one a
                rn_set, unlabelled_set = \
                    self.phase_1a(cand_solution, positive_set, unlabelled_set)

                # ensure that phase 1 a returned a reliable negative set
                # before proceeding
                if len(rn_set) > 0:

                    # perform phase 1 b if the flag is set and if there are
                    # still instances left in the unlabelled set
                    if cand_solution.flag_1b and len(unlabelled_set) > 0:
                        rn_set, unlabelled_set = \
                            self.phase_1b(cand_solution, positive_set, rn_set,
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
                y_pred = self.pred(cand_solution.classifier_2,
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

                # print(e)
                # print(traceback.format_exc())
                # sys.stdout.flush()

        cand_solution.tp = tp
        cand_solution.fp = fp
        cand_solution.tn = tn
        cand_solution.fn = fn

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

        cand_solution.fitness = f_measure

        return f_measure
    
    def assess_fitness(self, cand_solution, features, target, assessed):
        """Assess the fitness of a cand_solution on the current training set.
        cand_solution will be checked against the list of already assessed
        configurations. If present, their recall and precision will be set
        to the values previously calculated for the configuration on this
        training set.

        Parameters
        ----------
        cand_solution: Individual
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
        it_count_1a = copy.deepcopy(cand_solution.it_count_1a)
        rn_thresh_1a = copy.deepcopy(cand_solution.rn_thresh_1a)
        classifier_1a = copy.deepcopy(cand_solution.classifier_1a)
        rn_thresh_1b = copy.deepcopy(cand_solution.rn_thresh_1b)
        classifier_1b = copy.deepcopy(cand_solution.classifier_1b)
        flag_1b = copy.deepcopy(cand_solution.flag_1b)
        spies = copy.deepcopy(cand_solution.spies)
        spy_rate = copy.deepcopy(cand_solution.spy_rate)
        spy_tolerance = copy.deepcopy(cand_solution.spy_tolerance)
        classifier_2 = copy.deepcopy(cand_solution.classifier_2)
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
                cand_solution.fitness = assessed[i][5]

        # if the configuration is not found, assess it
        if not found:
            try:
                cand_solution.fitness = self.internal_CV(cand_solution, features, target)
            except Exception as e:
                print("Error for candidate_solution: ", cand_solution.print_details())
                print(e)
                print(traceback.format_exc())
                sys.stdout.flush()
                cand_solution.fitness = 0

        return cand_solution
    
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
        if individual.spies == "True":
            X_indiv.append(1)
        else:
            X_indiv.append(0)
        X_indiv.append(individual.spy_rate)
        X_indiv.append(individual.spy_tolerance)
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

    def tournament_selection(self, population, tourn_size, surrogate=False):
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
        tournament = [rand.choice(population) for _ in range(tourn_size)]

        if surrogate:
            return self.get_fittest_surrogate(tournament)
        else:
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
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.it_count_1a = \
                copy.deepcopy(new_indiv2.it_count_1a)
            new_indiv2.it_count_1a = \
                copy.deepcopy(new_indiv1.it_count_1a)

        # swap phase 1 a rn threshold
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.rn_thresh_1a = \
                copy.deepcopy(new_indiv2.rn_thresh_1a)
            new_indiv2.rn_thresh_1a = \
                copy.deepcopy(new_indiv1.rn_thresh_1a)

        # swap phase 1 a classifier
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.classifier_1a = \
                copy.deepcopy(new_indiv2.classifier_1a)
            new_indiv2.classifier_1a = \
                copy.deepcopy(new_indiv1.classifier_1a)

        # swap phase 1 b iteration count
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.it_count_1b = \
                copy.deepcopy(new_indiv2.it_count_1b)
            new_indiv2.it_count_1b = \
                copy.deepcopy(new_indiv1.it_count_1b)

        # swap phase 1 b rn threshold
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.rn_thresh_1b = \
                copy.deepcopy(new_indiv2.rn_thresh_1b)
            new_indiv2.rn_thresh_1b = \
                copy.deepcopy(new_indiv1.rn_thresh_1b)

        # swap phase 1 b classifier
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.classifier_1b = \
                copy.deepcopy(new_indiv2.classifier_1b)
            new_indiv2.classifier_1b = \
                copy.deepcopy(new_indiv1.classifier_1b)

        # swap phase 1 b flag
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.flag_1b = \
                copy.deepcopy(new_indiv2.flag_1b)
            new_indiv2.flag_1b = \
                copy.deepcopy(new_indiv1.flag_1b)

        # swap spy bool
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.spies = \
                copy.deepcopy(new_indiv2.spies)
            new_indiv2.spies = \
                copy.deepcopy(new_indiv1.spies)

        # swap spy rate
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.spy_rate = \
                copy.deepcopy(new_indiv2.spy_rate)
            new_indiv2.spy_rate = \
                copy.deepcopy(new_indiv1.spy_rate)

        # swap spy tolerance
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.spy_tolerance = \
                copy.deepcopy(new_indiv2.spy_tolerance)
            new_indiv2.spy_tolerance = \
                copy.deepcopy(new_indiv1.spy_tolerance)

        # swap phase 2 classifier
        if rand.uniform(0, 1) < self.comp_crossover_prob:
            new_indiv1.classifier_2 = \
                copy.deepcopy(new_indiv2.classifier_2)
            new_indiv2.classifier_2 = \
                copy.deepcopy(new_indiv1.classifier_2)

        # return the new individuals
        return new_indiv1, new_indiv2

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

    def crossover(self, population, new_pop_size, tourn_size):
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
        while len(new_population) < new_pop_size:

            # select two individuals with tournament selection
            indiv1 = \
                self.tournament_selection(population, tourn_size)
            indiv2 = \
                self.tournament_selection(population, tourn_size)

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
        fittest_individual = copy.deepcopy(population[0])

        # compare every individual in population
        for i in range(len(population)):

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if population[i].fitness > \
                    fittest_individual.fitness:
                fittest_individual = copy.deepcopy(population[i])

        # return the fittest individual and the counters
        return fittest_individual

    def get_fittest_surrogate(self, population):
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
        fittest_individual = copy.deepcopy(population[0])

        # compare every individual in population
        for i in range(len(population)):

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if population[i].surrogate_score > \
                    fittest_individual.surrogate_score:
                fittest_individual = copy.deepcopy(population[i])

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
            surrogate_score = str(individual.surrogate_score)
            f_measure = str(individual.fitness)
            tp = str(individual.tp)
            fp = str(individual.fp)
            tn = str(individual.tn)
            fn = str(individual.fn)

            indiv_detail = [it_count_1a, rn_thresh_1a,
                            classifier_1a, rn_thresh_1b,
                            classifier_1b, flag_1b, classifier_2,
                            tp, fp, tn, fn, f_measure, surrogate_score]

            individual_details.append(indiv_detail)

        # column names
        col = ["Iteration Count P1A", "RN Threshold P1A", "Classifier P1A",
               "RN Threshold P1B", "Classifier P1B", "Flag P1B",
               "Classifier P2", "TP", "FP", "TN", "FN", "F measure", "Surrogate Score"]

        # create dataframe
        individuals_df = pd.DataFrame(individual_details, columns=[col])

        try:
            # save to csv
            individuals_df.to_csv(self.log_dir +
                                  " Generation " +
                                  str(current_generation) +
                                  " individual details EBO.csv",
                                  index=False)
        except Exception as e:
            print("Could not save file:", self.log_dir +
                  " Generation " + str(current_generation) +
                  " individual details.csv")
            print(e)
            print(traceback.format_exc())
            sys.stdout.flush()
    
    def fit(self, features, target):
        """Fit an optimised PU learning algorithm to the given
           input dataset.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            List of class labels for prediction.

        Returns
        -------
        self: object
            Returns a fitted EBO_Auto_PU object.
        """

        # convert features and target to numpy arrays
        features = np.array(features)
        target = np.array(target)

        # initialise the population
        population = self.generate_population()
        # initialise the current iteration to 0
        current_it = 0

        # list to store all assessed configurations
        # saves computation time by allowing us to not have to
        # assess configurations that have already been assessed
        assessed_configs = []

        #######################################################
        ##################### FOR TESTING #####################
        ####################################################### 
        # new_pop = []
        # for indiv in population:
        #     new_pop.append(self.assess_fitness(indiv, features, target, assessed_configs))
        # population = new_pop
        #######################################################
        ##################### FOR TESTING #####################
        ####################################################### 

        # get the objective score (fitness) for all candidate solutions
        # in the population
        population = \
            Parallel(n_jobs=self.n_jobs,
                        verbose=0)(delayed(self.assess_fitness)
                                    (cand_solution=population[i],
                                    features=features,
                                    target=target, assessed=assessed_configs)
                                    for i in range(len(population)))
        
        # for all candidate solutions in the population, check if they
        # have previously been assessed. If not, add then to the list
        # of assessed configurations
        for i in range(len(population)):
            # some issues comparing candidate solutions as Objects, 
            # so have converted the description to a string for easy
            # comparison
            description = str([population[i].it_count_1a,
            population[i].rn_thresh_1a,
            population[i].classifier_1a,
            population[i].rn_thresh_1b,
            population[i].classifier_1b,
            population[i].flag_1b,
            population[i].spies,
            population[i].spy_rate,
            population[i].spy_tolerance,
            population[i].classifier_2])

            if description not in assessed_configs:
                # add config to assessed list
                assessed_configs.append([description, 
                                        population[i].tp, 
                                        population[i].fp, 
                                        population[i].tn,
                                        population[i].fn,
                                        population[i].fitness])
        
        n_population = copy.deepcopy(population)

        while current_it < self.it_count:

            # process the population into a dataset to train a random
            # forest regressor
            X_individuals = []
            y_individuals = []

            for i in range(len(population)):
                X_individuals.append(self.get_X_indiv(population[i]))
                y_individuals.append(population[i].fitness)       
                           
            clf = RandomForestRegressor(random_state=42)
            clf.fit(X_individuals, y_individuals)

            columns = ["It count 1a",
                       "RN threshold 1a",
                       "LinearDiscriminantAnalysis",
                       "CascadeForestClassifier",
                       "KNeighborsClassifier",
                       "BernoulliNB",
                       "GaussianNB",
                       "LogisticRegression",
                       "ExtraTreesClassifier",
                       "ExtraTreeClassifier",
                       "HistGradientBoostingClassifier",
                       "AdaBoostClassifier",
                       "SGD",
                       "RandomForestClassifier",
                       "BaggingClassifier",
                       "DecisionTreeClassifier",
                       "MLPClassifier",
                       "GaussianProcessClassifier",
                       "GradientBoostingClassifier",
                       "SVM",
                       "RN threshold 1b",
                       "LinearDiscriminantAnalysis",
                       "CascadeForestClassifier",
                       "KNeighborsClassifier",
                       "BernoulliNB",
                       "GaussianNB",
                       "LogisticRegression",
                       "ExtraTreesClassifier",
                       "ExtraTreeClassifier",
                       "HistGradientBoostingClassifier",
                       "AdaBoostClassifier",
                       "SGD",
                       "RandomForestClassifier",
                       "BaggingClassifier",
                       "DecisionTreeClassifier",
                       "MLPClassifier",
                       "GaussianProcessClassifier",
                       "GradientBoostingClassifier",
                       "SVM",
                       "Flag 1b",
                       "Spies",
                       "Spy rate",
                       "Spy tolerance",
                       "LinearDiscriminantAnalysis",
                       "CascadeForestClassifier",
                       "KNeighborsClassifier",
                       "BernoulliNB",
                       "GaussianNB",
                       "LogisticRegression",
                       "ExtraTreesClassifier",
                       "ExtraTreeClassifier",
                       "HistGradientBoostingClassifier",
                       "AdaBoostClassifier",
                       "SGD",
                       "RandomForestClassifier",
                       "BaggingClassifier",
                       "DecisionTreeClassifier",
                       "MLPClassifier",
                       "GaussianProcessClassifier",
                       "GradientBoostingClassifier",
                       "SVM",
                       "Fitness"]

            for i in range(len(X_individuals)):
                X_individuals[i].append(y_individuals[i])

            if self.log_dir is not None:
                pd.DataFrame(X_individuals, columns=columns).to_csv(self.log_dir + " individual details dataset format EBO " + str(current_it) + ".csv", index=False)

            # population undergoes crossover using tourn_size_a
            n_population = self.crossover(n_population, len(n_population), self.tourn_size_a)
            # population undergoes mutation
            n_population = self.mutate(n_population)

            # convert evolved population into dataste format to get the surrogate score of each
            X_new_indivs = []
            for i in range(len(n_population)):
                X_new_indivs.append(self.get_X_indiv(n_population[i]))
            
            y_pred = clf.predict(X_new_indivs)

            for i in range(len(y_pred)):
                n_population[i].surrogate_score = y_pred[i]

            X_individuals = []
            y_individuals = []

            for i in range(len(n_population)):
                X_individuals.append(self.get_X_indiv(n_population[i]))
                y_individuals.append(n_population[i].surrogate_score)  

            for i in range(len(X_individuals)):
                X_individuals[i].append(y_individuals[i])

            if self.log_dir is not None:
                pd.DataFrame(X_individuals, columns=columns).to_csv(self.log_dir + " n_population individual details dataset format EBO " + str(current_it) + ".csv", index=False)

            # elistism
            fittest_indiv = self.get_fittest_surrogate(n_population)

            # select k_pop
            k_pop = []
            for _ in range(self.k_pop_size):
                k_pop.append(self.tournament_selection(n_population, self.tourn_size_b, True))
            
            k_pop = self.crossover(k_pop, len(k_pop), self.tourn_size_a)

            k_pop = self.mutate(k_pop)

            X_new_indivs = []
            for i in range(len(k_pop)):
                X_new_indivs.append(self.get_X_indiv(k_pop[i]))
            
            y_pred = clf.predict(X_new_indivs)

            for i in range(len(y_pred)):
                k_pop[i].surrogate_score = y_pred[i]
            
            k_pop.append(fittest_indiv)
            

            # new_k_pop = []
            # for indiv in k_pop:
            #     new_k_pop.append(self.assess_fitness(indiv, features, target, assessed_configs))
            # k_pop = new_k_pop
            #######################################################
            ##################### FOR TESTING #####################
            ####################################################### 

            # get the objective score (fitness) for all candidate solutions
            # in the population
            k_pop = \
                Parallel(n_jobs=self.n_jobs,
                            verbose=0)(delayed(self.assess_fitness)
                                        (cand_solution=k_pop[i],
                                        features=features,
                                        target=target, assessed=assessed_configs)
                                        for i in range(len(k_pop)))
            
            # for all candidate solutions in the population, check if they
            # have previously been assessed. If not, add then to the list
            # of assessed configurations
            for i in range(len(k_pop)):
                # some issues comparing candidate solutions as Objects, 
                # so have converted the description to a string for easy
                # comparison
                description = str([k_pop[i].it_count_1a,
                k_pop[i].rn_thresh_1a,
                k_pop[i].classifier_1a,
                k_pop[i].rn_thresh_1b,
                k_pop[i].classifier_1b,
                k_pop[i].flag_1b,
                k_pop[i].spies,
                k_pop[i].spy_rate,
                k_pop[i].spy_tolerance,
                k_pop[i].classifier_2])

                if description not in assessed_configs:
                    # add config to assessed list
                    assessed_configs.append([description, 
                                            k_pop[i].tp, 
                                            k_pop[i].fp, 
                                            k_pop[i].tn,
                                            k_pop[i].fn,
                                            k_pop[i].fitness])

            for indiv in k_pop:
                population.append(indiv)

            if self.log_dir is not None:
                # save the details of all individuals in population
                # for this generation
                self.log_individuals(population, current_it)

            current_it += 1
        

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

            # print("Best configuration")
            # self.best_config.print_details()

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
                        self.phase_1b(self.best_config,
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

        except Exception as e:
            print("Evolved individual was unable to be trained on full training set.")
            print("It is likely that the individual was overfit during the evolution process.")
            print("Try again with different parameters, such as a higher number of individuals or generations.")
            print("For debugging, the exception is printed below.")
            print(e)
            print(traceback.format_exc())
            sys.stdout.flush()
        
        return self

    
    def predict(self, train_features, train_target, test_features):

        test_features = np.array(test_features)

        if not self.best_config:
            raise RuntimeError(
                "Auto_PU has not yet been fitted to the data. \
                Please call fit() first."
            )

        try:
            return self.best_config.classifier_2.predict(test_features)
        except NotFittedError as e:
            try:
                clf = self.ext_fit(train_features, train_target)
                return clf.best_config.classifier_2.predict(test_features)
            except Exception as e:
                print("Error for individual with following configuration")
                self.best_config.print_details()
                print(e)
                print(traceback.format_exc())  
                sys.stdout.flush() 
