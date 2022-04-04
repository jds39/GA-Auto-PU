# GA-Auto-PU

GA-Auto-PU is an Automated Machine Learning (Auto-ML) system for Positive-Unlabelled (PU) learning that constructs two-step PU learning algorithms for a specific input dataset. 

The system is based on a genetic algorithm, the details of which can be found in our paper (cite). 

To run the code

  clf = Auto_PU()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
