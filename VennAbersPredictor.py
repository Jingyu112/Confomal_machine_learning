#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:49:12 2024

@author: a
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier




class VennAbersBinary():
    
    """
    Venn-Abers prediction for binary classification problems
    
    References :
    -------------
    [1] Johansson U , Lfstrm T , Bostrm H .Well-Calibrated and Sharp Interpretable Multi-Class Models[J].Springer, Cham, 2021.DOI:10.1007/978-3-030-85529-1_16.
    
    
    Parameters :
    -------------
    model : Classifier
    calib_size : float or int, default 0.3, the size of calibration data
    random_state : int
    
    """
    
    def __init__(self, model, calib_size=0.3, random_state=None):
        self.model = model
        self.calib_size = calib_size
        self.random_state = random_state
        
        
    def get_params(self, deep=True):
        return {'model': self.model, 'calib_size': self.calib_size, 'radnom_state': self.random_state}
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if len(self.classes) > 2:
            raise TypeError(f"{type(self)} is not appropriate for a multiclass case ")
            
        # split data into two folds:
        X_train, X_calib, y_train, self.y_calib = train_test_split(X, y, test_size=self.calib_size, random_state=self.random_state)
        
        self.model.fit(X_train, y_train)
        
        # get prediction scores on calibration data, shape (n, 2):
        self.scores_calib = self.model.predict_proba(X_calib)
        
        
        
    def predict_p0p1(self, X):
        """
        
        return two 1d arrays : p0, p1

        """
        scores = self.model.predict_proba(X)[:, 1]
        p0, p1 = [], []
        
        for sample_score in scores:
            
            isotonic_model_0 = IsotonicRegression(out_of_bounds='clip')   # for the case that true target is 0
            isotonic_model_1 = IsotonicRegression(out_of_bounds='clip')   # for the case that true target is 1
            
            total_scores = np.append(self.scores_calib[:, 1], sample_score)
            
            total_labels_0 = np.insert(0, 0, self.y_calib)
            total_labels_1 = np.insert(1, 0, self.y_calib)
            
            # fit isotonic models:
            isotonic_model_0.fit(total_scores, total_labels_0)
            isotonic_model_1.fit(total_scores, total_labels_1)
            
            # fit isotonic models:
            isotonic_model_0.fit(total_scores, total_labels_0)
            isotonic_model_1.fit(total_scores, total_labels_1)
            
            # compute the lower and upper bounds for simple bing a class 1:
            p0.append(isotonic_model_0.transform([sample_score])[0])
            p1.append(isotonic_model_1.transform([sample_score])[0])
            
        return np.array(p0).flatten(), np.array(p1).flatten()



    def predict_proba(self, X, p0, p1):
    
        """
        return a 2D array of regularized probability 
        """        
        probs = np.zeros((len(X), 2))
        probs[:, 1] = p1/(1 - p0 + p1)
        probs[:, 0] = 1 - probs[: , 1]
    
        return probs
        
    
    
    

class VennAbersMulti():
    """
    Venn-Abers prediction for multi-class classification problems using OneVsOne
    
    Parameters :
    -------------
    model : Classifier
    calib_size : float or int, default 0.3, the size of calibration data
    random_state : int
    strategy : str, 'OneVsOne' or 'OneVsRest', default 'OneVsOne'
        
    """
        
    def __init__(self, model, calib_size = 0.3, random_state=None, strategy='OneVsOne'):
        self.model = model
        self.calib_size = calib_size
        self.random_state = random_state
        self.strategy = strategy
        
        if self.strategy not in ('OneVsOne', 'OneVsRest'):
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'" )
            
            
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if len(self.classes) == 2:
            raise TypeError(f"{type(self)} is not appropriate for a binary case")
            
        self.vab_list = []
        
        if self.strategy == 'OneVsOne':
            # OneVsOne strategy consists in fitting one classifier per class pair:
            self.pairwise_id = []
            
            for i in range(len(self.classes)):
                for j in range(i+1, len(self.classes)):
                    self.pairwise_id.append([self.classes[i], self.classes[j]])
                
            self.clf_ovo = OneVsOneClassifier(self.model).fit(X, y)
            
            for pair_id, clf_ovo_estimator in enumerate(self.clf_ovo.estimators_):
                _pairwise_indices = (y == self.pairwise_id[pair_id][0]) + (y == self.pairwise_id[pair_id][1])
                
                _X = X[_pairwise_indices]
                _y = np.array(y[_pairwise_indices] == self.pairwise_id[pair_id][1]).reshape(-1, 1)
                
                # introduce VennAbersBinary predictor:
                _vab = VennAbersBinary(clf_ovo_estimator, self.calib_size, self.random_state)
                _vab.fit(_X, _y.flatten())
                
                self.vab_list.append(_vab)
                
                
        elif self.strategy == 'OneVsRest' :
            # OneVsRest strategy consists in fitting one classifier per class
            self.clf_ovr = OneVsRestClassifier(self.model).fit(X, y)
            
            for class_id, clf_ovr_estimator in enumerate(self.clf_ovr.estimators_):
                _y = (y == class_id)
                
                # introduce VennAbersBinary predictor :
                _vab = VennAbersBinary(clf_ovr_estimator, self.calib_size, self.random_state)
                _vab.fit(X, _y.flatten())
                
                self.vab_list.append(_vab)
                
        else:
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
            
            
            
            
            
    def predict_proba(self, X):
        
        self.regular_proba_list = []
        
        # compute probability intervals for each pair:
        for i, vab in enumerate(self.vab_list):
            
            _p0, _p1 = vab.predict_p0p1(X)
            _regular_proba = vab.predict_proba(X, _p0, _p1)
            
            self.regular_proba_list.append(_regular_proba)
            
        p_regular = np.zeros((len(X),  len(self.classes)))
        
        
        if self.strategy == 'OneVsOne':
            
            for i, cl_id in enumerate(self.classes):
                
                stack_i = [ p[:, 0].reshape(-1, 1) for i, p in enumerate(self.regular_proba_list) if self.pairwise_id[i][0] == cl_id ]
                stack_j = [ p[:, 1].reshape(-1, 1) for i, p in enumerate(self.regular_proba_list) if self.pairwise_id[i][1] == cl_id ]
                
                # p_stack : a list of (n_classes - 1) arrays of (n, 1)
                p_stack = stack_i + stack_j

                # aggregate pariwise probabilities using harmonic mean of inverses (considering overlapping between classes):
                p_regular[:, i] = self.regular_proba_list[i][:, 1]
            
            # Normalizing probabilities across classes :
            p_regular = p_regular/np.sum(p_regular, axis=1).reshape(-1, 1)
                
        
        elif self.strategy == 'OneVsRest':
            
            for i, cl_id in enumerate(self.classes):
                p_regular[:, i] = self.regular_proba_list[i][:, 1]
                
            # Normalizing probabilities across classes :
            p_regular = p_regular/np.sum(p_regular, axis=1).reshape(-1, 1)
            
        else:
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
            
        
        return p_regular
    
    
    
    

class VennAbersBinaryCV():
    
    """
    Venn-Abers prediction on binary classification problems with cross-validation
    
    Parameters :
    --------------
    model : Classifier
    random_state : int, default None
    n_splits : int, default 3, nb of splits for cross validation
    shuffle : boolean, default False
    
    """


    def __init__(self, model, random_state=None, n_splits=3, shuffle=False):
        self.model = model
        self.random_state = random_state
        self.n_splits = n_splits
        self.shuffle = shuffle
        
        
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if len(self.classes) > 2:
            raise TypeError(f"{type(self)} is not appropriate for a multiclass case")
            
            
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        self.scores_calib_list = []
        self.y_calib_list = []
        
        for train_index, test_index in kf.split(X, y):
            self.model.fit(X[train_index], y[train_index].flatten())
            
            scores_calib = self.model.predict_proba(X[test_index])
            self.scores_calib_list.append(scores_calib)
            self.y_calib_list.append(y[test_index])
            
            
            
    def predict_p0p1(self, X):
        
        p0_list, p1_list = [], []
        
        for i in range(self.n_splits):
            vab = VennAbersBinary(self.model)
            params_to_add = {'y_calib': self.y_calib_list[i], 'n_classes': 2, 'scores_calib': self.scores_calib_list[i]}
            vab.set_params(**params_to_add)
            p0, p1 = vab.predict_p0p1(X)
            p0_list.append(p0)
            p1_list.append(p1)
            
            
        p0_stack = np.hstack([p0.reshape(-1, 1) for p0 in p0_list])    # shape (n, n_splits)
        p1_stack = np.hstack([p1.reshape(-1, 1) for p1 in p1_list])    # shape (n, n_splits)
        
        return p0_stack, p1_stack
    
    
    
    def predict_proba(self, X, p0_stack, p1_stack):
        
        p_regular = np.zeros((len(X), 2))
        
        # average and regularize the probabilities from different folds (add regularization by squaring probas: penalize very confident predictions)
        p_regular[:, 1] = 1/self.n_splits * (
                            np.sum(p1_stack, axis=1) + 0.5*np.sum(p0_stack**2, axis=1) - 0.5 * np.sum(p1_stack**2, axis=1))
        p_regular[:, 0] = 1 - p_regular[:, 1]
        
        return p_regular
    
    
    
    
    
class VennAbersMultiCV():
    
    """
    Venn-Abers prediction on multiclass problems with cross validation
    
    Parameters :
    --------------
    model : Classifier
    random_state : int, default None
    n_splits : int, default 3
    shuffle : boolean
    strategy : str, either 'OneVsOne' or 'OneVsRest', default 'OneVsOne'
    
    """
        
    def __init__(self, model, random_state=None, n_splits=3, shuffle=False, strategy='OneVsOne'):
        self.model = model
        self.random_state = random_state
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.strategy = strategy
        
        if self.strategy not in ('OneVsOne', 'OneVsRest'):
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
            
            
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if len(self.classes) == 2:
            TypeError("Having only 2 classes, should use VennAbersBinaryCV predictor")
            
        self.vab_list = []
        
        
        if self.strategy == 'OneVsOne':
            # OneVsOne strategy consists in fitting one classifier per class pair:
            self.pairwise_id = []
            
            for i in range(len(self.classes)):
                for j in range(i+1, len(self.classes)):
                    self.pairwise_id.append([self.classes[i], self.classes[j]])
                
            self.clf_ovo = OneVsOneClassifier(self.model).fit(X, y)
            
            for pair_id, clf_ovo_estimator in enumerate(self.clf_ovo.estimators_):
                _pairwise_indices = (y == self.pairwise_id[pair_id][0]) + (y == self.pairwise_id[pair_id][1])
                
                _X = X[_pairwise_indices]
                _y = np.array(y[_pairwise_indices] == self.pairwise_id[pair_id][1]).reshape(-1, 1)
                
                # introduce VennAbersBinary predictor:
                _vab_cv = VennAbersBinaryCV(
                    clf_ovo_estimator, 
                    self.random_state,
                    self.n_splits,
                    self.shuffle 
                    )
                _vab_cv.fit(_X, _y.flatten())
                
                self.vab_list.append(_vab_cv)
                
                
        elif self.strategy == 'OneVsRest' :
            # OneVsRest strategy consists in fitting one classifier per class
            self.clf_ovr = OneVsRestClassifier(self.model).fit(X, y)
            
            for class_id, clf_ovr_estimator in enumerate(self.clf_ovr.estimators_):
                _y = (y == class_id)
                
                # introduce VennAbersBinary predictor :
                _vab_cv = VennAbersBinaryCV(
                    clf_ovr_estimator, 
                    self.random_state,
                    self.n_splits,
                    self.shuffle
                    )
                _vab_cv.fit(X, _y.flatten())
                
                self.vab_list.append(_vab_cv)
                
        else:
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
        
        
    
    def predict_proba(self, X):
        
        self.regular_proba_list = []
        
        # compute probability intervals for each pair:
        for i, vab in enumerate(self.vab_list):
            
            _p0, _p1 = vab.predict_p0p1(X)
            _regular_proba = vab.predict_proba(X, _p0, _p1)
            
            self.regular_proba_list.append(_regular_proba)
            
        p_regular = np.zeros((len(X),  len(self.classes)))
        
        
        if self.strategy == 'OneVsOne':
            
            for i, cl_id in enumerate(self.classes):
                
                stack_i = [ p[:, 0].reshape(-1, 1) for i, p in enumerate(self.regular_proba_list) if self.pairwise_id[i][0] == cl_id ]
                stack_j = [ p[:, 1].reshape(-1, 1) for i, p in enumerate(self.regular_proba_list) if self.pairwise_id[i][1] == cl_id ]
                
                # p_stack : a list of (n_classes - 1) arrays of (n, 1)
                p_stack = stack_i + stack_j

                # aggregate pariwise probabilities using harmonic mean of inverses (considering overlapping between classes):
                p_regular[:, i] = self.regular_proba_list[i][:, 1]
            
            # Normalizing probabilities across classes :
            p_regular = p_regular/np.sum(p_regular, axis=1).reshape(-1, 1)
                
        
        elif self.strategy == 'OneVsRest':
            
            for i, cl_id in enumerate(self.classes):
                p_regular[:, i] = self.regular_proba_list[i][:, 1]
                
            # Normalizing probabilities across classes :
            p_regular = p_regular/np.sum(p_regular, axis=1).reshape(-1, 1)
            
        else:
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
            
        
        return p_regular
    
    
    
    


class VennAbersPredictor():
    """
    A wrapper for Venn-Abers predictor for classification problems
    Available for binary, multi-class, binary with cross-validation and multi-class with cross validation scenarios
    
    References :
    -------------
    [1] Johansson U , Lfstrm T , Bostrm H .Well-Calibrated and Sharp Interpretable Multi-Class Models[J].Springer, Cham, 2021.DOI:10.1007/978-3-030-85529-1_16.
    
    Parameters :
    -------------
    model : Classifier
    inductive : boolean, True for simple Venn-Abers prediction, False for cross validation 
    calib_size : float or int, default 0.3, valid only for inductive Venn-Abers prediction
    random_state : int, default None, must be None if shuffle is False
    n_splits : int, only for Venn-Abers prediction with cross validation
    shuffle : boolean
        
    """
    
    def __init__(self, model, inductive = True, calib_size = 0.3, random_state = None, n_splits = None, shuffle = False, strategy = 'OneVsOne'):
        self.model = model
        self.inductive = inductive
        self.calib_size = calib_size
        self.random_state = random_state
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.strategy = strategy
        self.va_predictor = None
        
        
        if not self.inductive and self.n_splits is None:
            raise ValueError("Cross Venn-Abers predictor needs an integer for n_splits")
            
        if self.strategy not in ('OneVsOne', 'OneVsRest'):
            raise ValueError(f"{self.strategy} is not a valid strategy, must either be 'OneVsOne' or 'OneVsRest'")
            
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        
        if len(self.classes) == 2:
            if self.inductive :
                self.va_predictor = VennAbersBinary(self.model, self.calib_size, self.random_state)
            else:
                self.va_predictor = VennAbersBinaryCV(self.model, self.random_state, self.n_splits, self.shuffle)
                
        else:
            if self.inductive :
                self.va_predictor = VennAbersMulti(self.model, self.calib_size, self.random_state, self.strategy)
            else:
                self.va_predictor = VennAbersMultiCV(self.model, self.random_state, self.n_splits, self.shuffle, self.strategy)
        
        
        self.va_predictor.fit(X, y)
        
        
        
    def predict_proba(self, X):
        
        if type(self.va_predictor) in (VennAbersBinary, VennAbersBinaryCV):
            p0, p1 = self.va_predictor.predict_p0p1(X)
            self.p_regular = self.va_perdictor.predict_proba(X, p0, p1)
            
        else:
            self.p_regular = self.va_predictor.predict_proba(X)
            
        return self.p_regular
    
    
    def predict(self, X):
        
        if not hasattr(self, 'p_regular'):
            self.predict_proba(X)
            
        prediction = self.p_regular.argmax(axis=1)
        
        return prediction
    
    
        
        
        
    
        
        
        
        
    