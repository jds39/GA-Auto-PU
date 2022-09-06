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
from deepforest import CascadeForestClassifier


class Classifiers: 

    def __init__(self, verbosity=0, random_state=42):

        self.classifiers = [GaussianNB(),
                            BernoulliNB(),
                            RandomForestClassifier(random_state=random_state),
                            DecisionTreeClassifier(random_state=random_state),
                            MLPClassifier(random_state=random_state, verbose=verbosity),
                            make_pipeline(StandardScaler(),
                                          SVC(gamma='auto', probability=True,
                                              random_state=random_state, verbose=verbosity)),
                            make_pipeline(StandardScaler(),
                                          SGDClassifier(random_state=random_state,
                                                        verbose=verbosity, loss="log")),
                            KNeighborsClassifier(),
                            LogisticRegression(random_state=random_state, verbose=verbosity),
                            CascadeForestClassifier(random_state=random_state, verbose=verbosity,
                                                 backend="sklearn"),
                            AdaBoostClassifier(random_state=random_state),
                            GradientBoostingClassifier(random_state=random_state,
                                                       verbose=verbosity),
                            LinearDiscriminantAnalysis(),
                            ExtraTreesClassifier(random_state=random_state, verbose=verbosity),
                            BaggingClassifier(random_state=random_state, verbose=verbosity),
                            ExtraTreeClassifier(random_state=random_state),
                            GaussianProcessClassifier(random_state=random_state),
                            HistGradientBoostingClassifier(verbose=verbosity,
                                                           random_state=random_state)
                      ] 