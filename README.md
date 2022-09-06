# GA-Auto-PU

GA-Auto-PU is an Automated Machine Learning (Auto-ML) system for Positive-Unlabelled (PU) learning that constructs two-step PU learning algorithms for a specific input dataset. 

The system is based on a genetic algorithm, the details of which can be found in our paper GA-Auto-PU: a genetic algorithm-based automated machine learning system for positive-unlabel learning by Jack D. Saunders & Alex A. Freitas in GECCO'22: Proceedings of the Genetic and Evolutionary Computation Conference Companion, July 22, pp-288-291. 

To run the code

    clf = Auto_PU()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
