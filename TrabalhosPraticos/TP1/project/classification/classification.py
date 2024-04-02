from sklearn.neighbors import KNeighborsClassifier

class Classification: 

    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.__train_images = train_images
        self.__train_labels = train_labels
        self.__test_images = test_images
        self.__test_labels = test_labels
        self.__model = KNeighborsClassifier(n_neighbors=3)


    def fit(self):
        self.__model.fit(self.__train_images, self.__train_labels)


    def predict(self):
        return self.__model.predict(self.__test_images)


    def evaluate(self):
        return self.__model.score(self.__test_images, self.__test_labels)


    def roc_curve(self, threshold = None):
        if threshold is None:  
            return self.__model.predict_proba(self.__test_images)
        
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        return self.__model.predict_proba(self.__test_images) < threshold
    