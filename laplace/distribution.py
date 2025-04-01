import numpy as np


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # 1. Вычисляем медиану по каждому признаку (axis=0)
        median = np.median(x, axis=0)

        # 2. Вычисляем абсолютные отклонения от медианы
        abs_deviation = np.abs(x - median)

        # 3. Усредняем отклонения по объектам (axis=0)
        mad = np.mean(abs_deviation, axis=0)
        ####

        return mad

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis=0)
        self.scale = self.mean_abs_deviation_from_median(features)
        ####

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block

        return np.log(1 / (2 * self.scale) * np.exp(-np.abs(values -
                                                            self.loc) / self.scale))
        ####

    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
