from abc import abstractmethod

class Recognition: 

    @abstractmethod
    def fit(self, train_images, train_labels=None):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def project(self, test_images):
        raise NotImplementedError("Method not implemented")
    