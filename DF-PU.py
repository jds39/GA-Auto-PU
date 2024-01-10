from deepforest import CascadeForestClassifier
import numpy as np

class DFPU:
    def __init__(self, threshold=0.5, rn_pct = 0.01, iterations=10):
        self.threshold = threshold
        self.rn_pct = rn_pct
        self.iterations = iterations
        self.model =  CascadeForestClassifier(verbose=0, backend="sklearn")

    def fit(self, X_train, y_train):
        training_set = np.c_[X_train, y_train]

        # Separate positive (P) and unlabeled (U) samples from the training set based on labels
        U = np.array([x for x in training_set if x[-1] == 0])
        P = np.array([x for x in training_set if x[-1] == 1])

        # Empty list to store identified reliable negatives
        RN = []

        for _ in range(self.iterations):

            # Randomly select a subset of unlabeled samples
            rU = U[np.random.randint(U.shape[0], size=round(len(U) * float(1 / self.iterations))), :]
            
            # Concatenate positive and unlabeled samples
            PU = np.concatenate([P, rU])

            # Shuffle the combined samples
            np.random.shuffle(PU)
            
            # Train a deep forest on the combined samples
            clf = CascadeForestClassifier(random_state=42, verbose=0, backend="sklearn")
            clf.fit(PU[:, :-1], PU[:, -1])
            
            # Predict the probabilities of the unlabeled samples being positive
            y_prob = clf.predict_proba(U[:, :-1])[:, 0]
            y_prob = np.array(y_prob, subok=True, ndmin=2).T
            
            # Combine the unlabeled samples with their predicted probabilities
            U_prob = np.concatenate([U, np.array(y_prob)], axis=1)
            # Sort the combined samples based on the predicted probabilities
            U_prob = U_prob[np.argsort(U_prob[:, -1])]
            
            # Determine the number of unlabeled samples to select based on the specified proportion
            to_get = round(len(U) * self.rn_pct)
            # Append the selected unlabeled samples to the list of reliable negative instances
            RN.append(U_prob[:to_get])
            
        RNs = []
        # Flatten the list of selected samples
        RNs = [y for x in RN for y in x]

        # Convert the selected samples into a NumPy array and remove the last column (labels)
        RNs = np.array(RNs)
        RNs = RNs[:, :-1]
        
        # Combine the positive samples and the selected samples
        PRN = np.concatenate([P, RNs])
        # Shuffle the combined samples
        np.random.shuffle(PRN)
        
        # Train a deep forest on the combined samples
        self.model = CascadeForestClassifier(random_state=42, verbose=0, backend="sklearn")
        self.model.fit(PRN[:, :-1], PRN[:, -1])

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)