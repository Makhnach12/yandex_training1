import random

import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            random_subset = random.choices(range(0, data_length), k=data_length)
            self.indices_list.append(random_subset)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = [data[i] for i in self.indices_list[bag]], [target[i] for i in self.indices_list[bag]]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        results = np.array([model.predict(data) for model in self.models_list])
        return np.mean(results, axis=0)
        # Your code here
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        for idx in range(len(self.data)):
            for i, indices in enumerate(self.indices_list):
                if idx not in indices:
                    list_of_predictions_lists[idx].append(
                        self.models_list[i].predict(np.array(self.data[idx]).reshape(1, -1))
                    )        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array(
            [
                np.mean(np.array(self.list_of_predictions_lists[i])) 
                if len(self.list_of_predictions_lists[i]) != self.num_bags else np.nan
                for i in range(len(list(self.list_of_predictions_lists)))
            ]
        )
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        indices = ~np.isnan(self.oob_predictions)
        print(np.mean((self.oob_predictions[indices] - self.target[indices])**2))
        return np.mean((self.oob_predictions[indices] - self.target[indices])**2)



