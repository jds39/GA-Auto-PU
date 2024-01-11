# Auto-PU

The Auto-PU systems are Automated Machine Learning (Auto-ML) systems for Positive-Unlabelled (PU) learning that constructs two-step PU learning algorithms for a specific input dataset. 

GA-Auto-PU is based on a genetic algorithm, described first in our short paper Saunders, J.D. and Freitas, A.A., 2022, July. GA-auto-PU: a genetic algorithm-based automated machine learning system for positive-unlabeled learning. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 288-291), and extended in our work Saunders, J.D. and Freitas, A.A., 2022, October. Evaluating a New Genetic Algorithm for Automated Machine Learning in Positive-Unlabelled Learning. In International Conference on Artificial Evolution (Evolution Artificielle) (pp. 42-57). Cham: Springer Nature Switzerland.

The BO-Auto-PU and EBO-Auto-PU systems are described in our work Saunders, J.D and Freitas, A.A., 2024, January. Automated machine learning for positive-unlabelled learning. ArXiv (add link after posting).

To run the code, ensure that both the Individuals.py and Classifiers.py files are in the same directory, then run:

    clf = Auto_PU()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

To run with spy-based methods:

    clf = Auto_PU(spies=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

If using BO- or EBO-Auto-PU, replace Auto_PU with BO_Auto_PU or EBO_Auto-PU, respectively.
