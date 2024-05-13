from abc import abstractmethod


class Recognition:

    @abstractmethod
    def classify(self, image):
        raise NotImplementedError("Abstract method.")