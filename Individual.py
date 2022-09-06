import random as rand
import numpy as np

from Classifiers import Classifiers

class Individual:
    """Class to store information of an individual."""

    def __init__(self,
                 it_counts=np.arange(start=1, stop=11, step=1),
                 rn_thresholds=np.arange(start=0.05, stop=0.55, step=0.05),
                 spy_rates=np.arange(start=0.05, stop=0.4, step=0.05),
                 spy_tolerances=np.arange(start=0, stop=0.1, step=0.01)):
            
        # initialise all values as 0, empty, None, or False
        self.fitness = self.surrogate_score = self.it_count_1a = self.rn_thresh_1a = \
            self.it_count_1b = self.rn_thresh_1b = self.spy_rate = self.spy_tolerance = \
            self.tp = self.fp = self.tn = self.fn = 0
        self.classifier_1a = self.classifier_1b = self.classifier_2 = None
        self.flag_1b = self.spies = False
        self.f_measure_list = []

        self.spy_rates = spy_rates
        self.spy_tolerances = spy_tolerances

        # the list of iteration counts that can be selected
        # default is 1 - 10 (inclusive), increments of 1
        self.it_counts = it_counts

        # the list of rn thresholds that can be selected
        # default is 0.05 - 0.5 (inclusive), increments of 0.05
        self.rn_thresholds = rn_thresholds

        # the list of classifiers that can be selected
        c = Classifiers()
        self.classifiers = c.classifiers 

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
        return rand.choice(self.it_counts)

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
        return rand.choice(self.rn_thresholds)

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
        return rand.choice(self.classifiers)
    
    def generate_spy_rate(self):
        """Return a random value from the list of spy rates.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        spy_rate: Float
            The float the use as the spy rate.
        
        """
        return rand.choice(self.spy_rates)
    
    def generate_spy_tolerances(self):
        """Return a random value from the list of spy tolerances.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        spy_tolerance: Float
            The float the use as the spy tolerance.
        
        """
        return rand.choice(self.spy_tolerances)

    def generate_bool(self):
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

    def mutate_it_count_1a(self):
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
        if self.it_count_1a > 1 and self.it_count_1a < 10:
            if rand.uniform(0, 1) < 0.5:
                self.it_count_1a += 1
            else:
                self.it_count_1a -= 1

        # if iteration count is 1, add 1
        elif self.it_count_1a == 1:
            self.it_count_1a += 1

        # if iteration count is 10, minus 1
        elif self.it_count_1a == 10:
            self.it_count_1a -= 1

        # return modified iteration count
        # self.it_count_1a = iteration_count
        return self

    def mutate_it_count_1b(self):
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
        if self.it_count_1b > 1 and self.it_count_1b < 10:
            if rand.uniform(0, 1) < 0.5:
                self.it_count_1b += 1
            else:
                self.it_count_1b -= 1

        # if iteration count is 1, add 1
        elif self.it_count_1b == 1:
            self.it_count_1b += 1

        # if iteration count is 10, minus 1
        elif self.it_count_1b == 10:
            self.it_count_1b -= 1

        # return modified iteration count
        # self.it_count_1a = iteration_count
        return self

    def mutate_rn_thresh_1a(self):
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
        if self.rn_thresh_1a > 0.05 and self.rn_thresh_1a < 0.5:
            if rand.uniform(0, 1) < 0.5:
                self.rn_thresh_1a = round(self.rn_thresh_1a + 0.05, 3)
            else:
                self.rn_thresh_1a = round(self.rn_thresh_1a - 0.05, 3)

        # if threshold is 0.05, add 0.05
        elif self.rn_thresh_1a == 0.05:
            self.rn_thresh_1a = round(self.rn_thresh_1a + 0.05, 3)

        # if threshold is 0.5, subtract 0.05
        elif self.rn_thresh_1a == 0.5:
            self.rn_thresh_1a = round(self.rn_thresh_1a - 0.05, 3)

        # return modified rn threshold
        return self

    def mutate_rn_thresh_1b(self):
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
        if self.rn_thresh_1b > 0.05 and self.rn_thresh_1b < 0.5:
            if rand.uniform(0, 1) < 0.5:
                self.rn_thresh_1b = round(self.rn_thresh_1b + 0.05, 3)
            else:
                self.rn_thresh_1b = round(self.rn_thresh_1b - 0.05, 3)

        # if threshold is 0.05, add 0.05
        elif self.rn_thresh_1b == 0.05:
            self.rn_thresh_1b = round(self.rn_thresh_1b + 0.05, 3)

        # if threshold is 0.5, subtract 0.05
        elif self.rn_thresh_1b == 0.5:
            self.rn_thresh_1b = round(self.rn_thresh_1b - 0.05, 3)

        # return modified rn threshold
        return self

    def mutate_spy_rate(self):
        """Mutate spy rate.

        Parameters
        ----------
        rn_threshold: float
            The spy rate to mutate.

        Returns
        -------
        rn_threshold: float
            The spy rate after mutation.

        """

        # if spy_rate is between 0.05 and 0.35, either add
        # or subtract
        if self.spy_rate > 0.05 and self.spy_rate < 0.35:
            if rand.uniform(0, 1) < 0.5:
                self.spy_rate = round(self.spy_rate + 0.05, 3)
            else:
                self.spy_rate = round(self.spy_rate - 0.05, 3)

        # if spy_rate is 0.05, add 0.05
        elif self.spy_rate == 0.05:
            self.spy_rate = round(self.spy_rate + 0.05, 3)

        # if spy_rate is 0.5, subtract 0.05
        elif self.spy_rate == 0.35:
            self.spy_rate = round(self.spy_rate - 0.05, 3)

        # return modified spy_rate
        return self

    def mutate_spy_tolerance(self):
        """Mutate spy tolerance.

        Parameters
        ----------
        rn_threshold: float
            The spy tolerance to mutate.

        Returns
        -------
        rn_threshold: float
            The spy tolerance after mutation.

        """

        # if spy_tolerance is between 0 and 0.1, either add
        # or subtract
        if self.spy_tolerance > 0 and self.spy_tolerance < 0.1:
            if rand.uniform(0, 1) < 0.5:
                self.spy_tolerance = round(self.spy_tolerance + 0.01, 3)
            else:
                self.spy_tolerance = round(self.spy_tolerance - 0.01, 3)

        # if spy_tolerance is 0, add 0.01
        elif self.spy_tolerance == 0:
            self.spy_tolerance = round(self.spy_tolerance + 0.01, 3)

        # if spy_tolerance is 0.1, subtract 0.01
        elif self.spy_tolerance == 0.1:
            self.spy_tolerance = round(self.spy_tolerance - 0.01, 3)

        # return modified spy_tolerance
        return self
    


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
        print("Phase 1 A iteration count:", self.it_count_1a)
        print("Phase 1 A RN threshold:", self.rn_thresh_1a)
        print("Phase 1 A classifier:", self.classifier_1a)
        print("Phase 1 B flag:", self.flag_1b)
        print("Phase 1 B iteration count:", self.it_count_1b)
        print("Phase 1 B RN threshold:", self.rn_thresh_1b)
        print("Phase 1 B classifier:", self.classifier_1b)
        print("Spies:", self.spies)
        print("Spy rate:", self.spy_rate)
        print("Spy tolerance:", self.spy_tolerance)
        print("Phase 2 classifier:", self.classifier_2)
        print("Fitness:", self.fitness)
        print("Surrogate score:", self.surrogate_score)
    
    def return_values(self):
        """Return the components of the individual.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        iteration_count_1_a: int
            The number of iterations to perform in Phase 1A.
        rn_theshold_1_a: float
            The value under which an instance is considered reliably
            negative in Phase 1A.
        classifier_1_a: Object
            The classifier to use in Phase 1A.
        flag_1_b: bool
            Whether to use Phase 1B.
        iteration_count_1_b: int
            The number of iterations to perform in Phase 1B.
        rn_threshold_1_b: float
            The value under which an instance is considered reliably
            negative in Phase 1B.
        classifier_1_b: Object
            The classifier to use in Phase 1B.
        classifier_2: Object
            The classifier to use in Phase 2.
        """

        return [self.it_count_1a, self.rn_thresh_1a, self.classifier_1a,
                self.flag_1b, self.it_count_1b, self.rn_thresh_1b,
                self.classifier_1b, self.spies, self.spy_rate,
                self.spy_tolerance, self.classifier_2] 