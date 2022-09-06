# GA-Auto-PU

GA-Auto-PU is an Automated Machine Learning (Auto-ML) system for Positive-Unlabelled (PU) learning that constructs two-step PU learning algorithms for a specific input dataset. 

The system is based on a genetic algorithm, the details of which can be found in our paper GA-Auto-PU: a genetic algorithm-based automated machine learning system for positive-unlabel learning by Jack D. Saunders & Alex A. Freitas in GECCO'22: Proceedings of the Genetic and Evolutionary Computation Conference Companion, July 22, pp-288-291. 

This repository now contains an updated version of the system, capable of generating individuals using spy-based two-step methods, as introduced in the soon to be published paper Evaluating a New Genetic Algorithm for Automated Machine Learning in Positive-Unlabelled Learning by Jack D. Saunders & Alex A. Freitas in Proceedings of the 15th Biennial International Conference on Artificial Evolution (EA 2022), November 22. 

To run the code, ensure that both the Individuals.py and Classifiers.py files are in the same directory, then run:

    clf = Auto_PU()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

To run with spy-based methods:

    clf = Auto_PU(spies=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
