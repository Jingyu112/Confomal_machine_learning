#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:03:27 2024

@author: Jingyu
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin, TransformerMixin



class ConformalPredictTransformer(ClassifierMixin, TransformerMixin):
    
    """
    This class implements Conformal Prediction on classification problems in an inductive setting.
    
    References:
    -------------
    [1] Bhattacharyya, S. (2013b). Confidence in predictions from random tree ensembles. Knowledge and Information Systems, 35(2), 391-410.
    https://doi.org/10.1007/s10115-012-0600-z
    [2] Shi, F., Ong, C. S., & Leckie, C. (2013). Applications of Class-Conditional Conformal Predictor in Multi-class Classification (Vol. 13, pp. 235-239).
    https://doi.org/10.1109/icmla.2013.48
    
    
    Parameters:
    -------------
    model : scikit-learn Classifier
    calib_size : float or int, default 0.3, the size of calibration data
    random_state : int
    classwise : boolean, True for the case where data are differently distributed among classes, default False
    method : str, "Simple" or "KProx", default "Simple", "KProx" only valid for a RandomForestClassifier
    k : int, number of nearest neighbors taken into account in KProx method
    
    
    """
    
    def __init__(self, model, calib_size = 0.3, random_state = None, classwise = False, method = 'Simple', k = None):
        self.model = model
        self.calib_size = calib_size
        self.random_state = random_state
        self.classwise = classwise
        self.method = method
        self.k = k
        
        
        # check if the method and model are properly called:
        if self.method not in ('Simple', 'KProx'):
            raise ValueError(f"{self.method} should be either Simple or KProx")
            
        if not isinstance(self.model, RandomForestClassifier) and self.method == 'KProx':
            raise TypeError(f"KProx method is not available for {type(self.model)}")
            
            
    def get_params(self, deep=True):
        return {'model': self.model, 'calib_size': self.calib_size, 'random_state': self.random_state, 'k': self.k, 'classwise': self.classwise}
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
    def prox(self, leaf_xi, leaf_xj):
        """
        leaf_xi : array (n_trees, _ ), leaf nodes of data xi
        leaf_xj : array (n_trees, _ ), leaf nodes of data xj
        return : float, proximity of xi and xj

        """
        assert leaf_xi.ndim == 1
        assert leaf_xj.ndim == 1
        assert leaf_xi.shape == leaf_xj.shape
        proximity = np.sum(leaf_xi == leaf_xj)/leaf_xi.shape[0]
        return proximity
    
    
    
    
    def prox_score(self, leaf_xi, yi):
        
        """
        leaf_xi : array (n_trees, _ ), leaf nodes of data xi
        yi : integer, label of data xi
        return : float

        """
        # k-nearest indices of the same  class:
        same_label_indices = np.where(self.y_calib == yi)[0]
        
        # k-nearest indices of the other classes:
        diff_label_indices = np.where(self.y_calib != yi)[0]
        
        # check the number of same/diff_label_indices is larger than k:
        if self.k > min(len(same_label_indices), len(diff_label_indices)):
            raise ValueError(f"{self.k} is too large for given data")
        
        same_label_leaf = self.leaf_calib[same_label_indices]
        same_label_prox = np.apply_along_axis(self.prox, 1, same_label_leaf, leaf_xi)
        sum_same_label_prox = np.sum(same_label_prox[np.argsort(same_label_prox)[-(self.k+1):]]) - 1
        
        diff_label_leaf = self.leaf_calib[diff_label_indices]
        diff_label_prox = np.apply_along_axis(self.prox, 1, diff_label_leaf, leaf_xi)
        sum_diff_label_prox = np.sum(diff_label_prox[np.argsort(diff_label_prox)[-self.k:]])
        
        return sum_diff_label_prox/sum_same_label_prox
    
    
    def prox_score_test(self, leaf_xi, yi):
        
        """
        leaf_xi : array (n-trees, _ ), leaf nodes of data xi
        yi : integer, label of xi
        return : float
        
        Difference from prox_score(): this function is used for calculating prox_score of test data, since X_calib do not contain leaf_xi itself
        """
        # k-nearest indices of the same  class:
        same_label_indices = np.where(self.y_calib == yi)[0]
        
        # k-nearest indices of the other classes:
        diff_label_indices = np.where(self.y_calib != yi)[0]
        
        # check the number of same/diff_label_indices is larger than k:
        if self.k > min(len(same_label_indices), len(diff_label_indices)):
            raise ValueError(f"{self.k} is too large for given data")
        
        same_label_leaf = self.leaf_calib[same_label_indices]
        same_label_prox = np.apply_along_axis(self.prox, 1, same_label_leaf, leaf_xi)
        sum_same_label_prox = np.sum(same_label_prox[np.argsort(same_label_prox)[-self.k:]]) 
        
        diff_label_leaf = self.leaf_calib[diff_label_indices]
        diff_label_prox = np.apply_along_axis(self.prox, 1, diff_label_leaf, leaf_xi)
        sum_diff_label_prox = np.sum(diff_label_prox[np.argsort(diff_label_prox)[-self.k:]])
        
        return sum_diff_label_prox/sum_same_label_prox

        

    def fit_calib_scores(self, X_calib, y_calib):
        
        """
        X_calib : array-like calibration data
        y_calib : array-like calibration data labels
        return : non-conformity scores of calibration data, array (n, ) if classwise is True; dictionary otherwise

        """
        if self.method == 'KProx':
            self.leaf_calib = self.model.apply(X_calib)
            scores_calib = np.array([self.prox_score(x, y_) for x, y_ in zip(self.leaf_calib, y_calib)])
        
        elif self.method == 'Simple':
            probas = self.model.predict_proba(X_calib)
            scores_calib = 1 - probas[np.arange(len(y_calib)), y_calib]
            
        else:
            raise ValueError(f"{self.method} should be either Simple or KProx")
        
        
        if self.classwise :
            scores_calib_classwise = {}
            for label in self.classes :
                mask = (y_calib == label)
                scores_calib_classwise[label] = scores_calib[mask]
            return scores_calib_classwise
        
        else:
            return scores_calib
        
        
        
        
    def scores_predict(self, leaf_xi):
        
        """
        leaf_xi : array of leaf_test (n_trees, )
        return : array of scores_test (n_classes, )

        """
        scores_obs = np.array([self.prox_score_test(leaf_xi, label) for label in self.classes])
        
        return scores_obs
    
    
    
    def get_p_values(self, test_scores):
        
        """
        test_scores : array, scores of test data
        return : array, p-values in shape (n, n_classes)
    
        """
        gen = check_random_state(seed=123)
        sigma = gen.rand()
        
        p_values_list = []

        for test_score in test_scores:

            new_scores = np.append(self.scores_calib, test_score)
            
            p_values_list.append((len(new_scores[new_scores > test_score]) + sigma*(len(new_scores[new_scores == test_score])))/len(new_scores))
        
        return np.array(p_values_list)




    def get_p_values_classwise(self, test_scores):
        
        """
        test_scores : array, scores of test data
        return : array, p-values in shape (n, n_classes)
    
        """
        
        gen = check_random_state(seed=123)
        sigma = gen.rand()
        
        p_values_total = []
        for test_score in test_scores:
            p_values_list = []
            for label in self.classes:
                if label in self.scores_calib.keys():
                    new_scores = np.append(self.scores_calib[label], test_score[label])
                    p_values_list.append((len(new_scores[new_scores > test_score[label]]) + sigma*(len(new_scores[new_scores == test_score[label]])))/ len(new_scores))

                else:
                    p_values_total.append(0)
                    
            p_values_total.append(p_values_list)
        return np.array(p_values_total)
    
    
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        X_train, X_calib, y_train, self.y_calib = train_test_split(X, y, test_size=self.calib_size, random_state=self.random_state)
        
        
        # Fit model:
        self.model.fit(X_train, y_train)
        
        self.scores_calib = self.fit_calib_scores(X_calib, self.y_calib)
        
        
        
    def transform(self, X):
        
        if self.method == 'KProx':
            # compute leaf features for test data:
            leaf_test = self.model.apply(X)
            scores_test = np.apply_along_axis(self.scores_predict, 1, leaf_test)
            
        elif self.method == 'Simple':
            scores_test = 1 - self.model.predict_proba(X)
            
        else:
            raise ValueError(f"{self.method} should be either Simple or KProx")
        
            
        if self.classwise:
            p_values = self.get_p_values_classwise(scores_test)
        else:
            p_values = np.apply_along_axis(self.get_p_values, 1, scores_test)
 
        
        return p_values
    
    
    
    def predict(self, p_values, X_test=None):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
                
            p_values = self.transform(X_test)
            
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_values is given")
                
        predictions = p_values.argmax(axis=1)
        return predictions
    
    
    
    def predict_sets(self, p_values, X_test=None, alpha=0.1):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
            p_values = self.transform(X_test)
            
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_values is given")
                    
    
        # make prediction sets for test data given p_values_test and significance level:
        indices = np.where(p_values >= alpha, 1, 0)
        prediction_sets_list = []
        for i in range(len(p_values)):
            predict_set = np.where(indices[i, :] == 1)[0]
            prediction_sets_list.append(predict_set)
        return prediction_sets_list 
        
    
    


class ConformalPredictTransformerCV(ClassifierMixin, TransformerMixin):
    
    """
    This implements Conformal Prediction on classification problems with cross validation
    
    Parameters :
    -------------
    model : Classifier
    random_state : int, default None
    classwise : boolean, True for the case where data are differently distributed among classes, default False
    method : str, "Simple" or "KProx", default "Simple", "KProx" only valid for a RandomForestClassifier
    k : int, number of nearest neighbors taken into account in KProx method
    n_splits : int or float 
    shuffle : boolean
    
    """
        
    def __init__(self, model, random_state = None, classwise = False, method = 'Simple', k = None, n_splits = 3, shuffle = False):
        self.model = model
        self.random_state = random_state
        self.classwise = classwise
        self.method = method
        self.k = k
        self.n_splits = n_splits
        self.shuffle = shuffle
       
        
        # Check if the method and model are called properly:
        if self.method not in ('Simple', 'KProx'):
            raise ValueError(f"{self.method} should be either Simple or KProx")

        if not isinstance(self.model, RandomForestClassifier) and self.method == 'KProx':
            raise TypeError(f"Kprox method is not available for {type(self.model)}")
            
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        kf = StratifiedKFold(n_splits = self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        self.cp_list = []
        for train_index, calib_index in kf.split(X, y):
            
            cp = ConformalPredictTransformer(model = self.model, method = self.method, k = self.k, classwise = self.classwise)
            cp.model.fit(X[train_index], y[train_index].flatten())
            
            params_1 = {'classes': self.classes, 'y_calib': y[calib_index].flatten()}
            cp.set_params(**params_1)
            scores_calib = cp.fit_calib_scores(X[calib_index], y[calib_index].flatten())
            params_2 = {'scores_calib': scores_calib}
            cp.set_params(**params_2)
            self.cp_list.append(cp)
            
            
    def transform(self, X):
        
        p_values_list = []
        
        for i in range(self.n_splits):
            cp = self.cp_list[i]
            
            if self.method == 'KProx':  
                #compute leaf features for test data:
                leaf_test = cp.model.apply(X)
                scores_test = np.apply_along_axis(cp.scores_predict, 1, leaf_test)
                
            elif self.method == 'Simple':
                scores_test = 1 - cp.model.predict_proba(X)
                
            else:
                raise ValueError(f"{self.method} should be either Simple or KProx")
                
            if self.classwise :
                p_values = cp.get_p_values_classwise(scores_test)  # shape (n, n_classes)
            else:
                p_values = np.apply_along_axis(cp.get_p_values, 1, scores_test)   # shape (n, n_classes)
            
            p_values_list.append(p_values)
            
        p_values_avg = np.mean(np.array(p_values_list), axis=0)    # shape (n, n_classes)
        
        
        return p_values_avg
    
    
    
    
    def predict(self, p_values, X_test=None):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
            p_values = self.transform(X_test)
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_Values is given")
        
        predictions = p_values.argmax(axis=1)
        return predictions
    
    
    def predict_sets(self, p_values, X_test=None, alpha=0.1):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
            p_values = self.transform(X_test)
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_Values is given")
        
        
        # make prediction sets for test data given p_values_test and significance level:
        indices = np.where(p_values >= alpha, 1, 0)
        prediction_sets_list = []
        for i in range(len(p_values)):
            predict_set = np.where(indices[i, :] == 1)[0]
            prediction_sets_list.append(predict_set)
        return prediction_sets_list
    
    
    



class ConformalPredictor():
    
    """
    
    A wrapper for Conformal Prediction for classification problems with or without cross-validation
    The non-conformity function for Simple method is 1 - proba(X), and method KProx is not only valid for a RandomForestClassifier;
    

    References:
    -------------
    [1] Bhattacharyya, S. (2013b). Confidence in predictions from random tree ensembles. Knowledge and Information Systems, 35(2), 391-410.
    https://doi.org/10.1007/s10115-012-0600-z
    [2] Shi, F., Ong, C. S., & Leckie, C. (2013). Applications of Class-Conditional Conformal Predictor in Multi-class Classification (Vol. 13, pp. 235-239).
    https://doi.org/10.1109/icmla.2013.48
    
    
    
    Parameters :
    -------------
    model : Classifier
    inductive : boolean, True for the case without cross-validation , and False for the case with cross-validation
    calib_size : float or int, default 0.3, valid only for inductive setting
    random_state : int, default None
    classwise : boolean, True for the case where data are differently distributed among classes, default False
    method : str, "Simple" or "KProx", default "Simple", "KProx" only valid for a RandomForestClassifier
    k : int, number of nearest neighbors taken into account in KProx method
    n_splits : int or float 
    shuffle : boolean
    
    
    """
    
    
    def __init__(self, model, inductive=True, calib_size=0.3, classwise=False, method='Simple', k=None, random_state=None, n_splits=None, shuffle=False):
        self.model = model
        self.inductive = inductive
        self.calib_size = calib_size
        self.classwise = classwise
        self.method = method 
        self.k = k
        self.random_state = random_state
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.classes = []
        self.cp = []
        
        
        # check if the method and model are properly called:
        if self.method not in ('Simple', 'KProx'):
            raise ValueError(f"{self.method} should be either Simple or KProx")
            
        if not isinstance(self.model, RandomForestClassifier) and self.method == 'KProx':
            raise TypeError(f"KProx method is not available for {type(self.model)}")
            
        if not self.inductive and self.n_splits is None:
            raise ValueError("Cross Conformal Predictor needs an integer for n_splits")
            
            
            
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if self.inductive :
            self.cp = ConformalPredictTransformer(model = self.model, calib_size=self.calib_size, random_state=self.random_state, k = self.k, method = self.method, classwise = self.classwise)
        else:
            self.cp = ConformalPredictTransformerCV(model = self.model, k= self.k, method = self.method, classwise = self.classwise, n_splits = self.n_splits, random_state = self.random_state, shuffle = self.shuffle)
            
        self.cp.fit(X, y)
        
        
    def transform(self, X_test):
        
        p_values = self.cp.transform(X_test)
        return p_values
    
    
    
    
    def predict(self, p_values, X_test=None):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
            p_values = self.transform(X_test)
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_Values is given")
        
        predictions = p_values.argmax(axis=1)
        return predictions
    
    

    
    
    def predict_sets(self, p_values, X_test=None, alpha=0.1):
        if p_values is None:
            if X_test is None:
                raise ValueError("I need X_test if I don't have p_values")
            p_values = self.transform(X_test)
        else:
            if X_test is not None:
                raise ValueError("I don't accept 'X_test' if p_Values is given")
        
        
        # make prediction sets for test data given p_values_test and significance level:
        indices = np.where(p_values >= alpha, 1, 0)
        prediction_sets_list = []
        for i in range(len(p_values)):
            predict_set = np.where(indices[i, :] == 1)[0]
            prediction_sets_list.append(predict_set)
        return prediction_sets_list
    
        



















