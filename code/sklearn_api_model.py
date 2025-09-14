from sklearn.utils import check_random_state

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

import random

from scipy import stats
from forecasting_models.sklearn.score import *
from forecasting_models.sklearn.ordinal_classifier import *
from forecasting_models.sklearn.models import *
from skopt import BayesSearchCV, Optimizer
from skopt.space import Integer, Real

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.model_selection import cross_validate

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hinge_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
)

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.utils.validation import check_is_fitted

from imblearn.over_sampling import SMOTE

try:
    if torch.cuda.is_available():
        import cupy as cp
except:
    pass

np.random.seed(42)
random.seed(42)
random_state = check_random_state(42)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))
      
##########################################################################################
#                                                                                        #
#                                   Base class                                           #
#                                                                                        #
##########################################################################################

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, nbfeatures,
                 model_type, loss='logloss', name='Model',
                 dir_log = Path('../'), under_sampling='full',
                 over_sampling='full', target_name='nbsinister',
                 task_type='regression', post_process=None, n_run=1):
        """
        Initialize the CustomModel class.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        if task_type == 'ordinal-classification':
            self.best_estimator_ = OrdinalClassifier(model)
        else:
            self.best_estimator_ = model
        self.model_type = model_type
        self.name = self.best_estimator_.__class__.__name__ if name == 'Model' else name
        self.loss = loss
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.dir_log = dir_log
        self.final_score = None
        self.features_selected = None
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type
        self.post_process = post_process
        self.nbfeatures = nbfeatures
        self.over_sampling = over_sampling
        self.n_run = n_run
        self.metrics = {}

    def split_dataset(self, X, y, y_train_score, nb, is_unknowed_risk):        
        # Separate the positive and zero classes based on y
        if not is_unknowed_risk:
            positive_mask = y[self.target_name] > 0
            non_fire_mask = y[self.target_name] == 0
        else:
            non_fire_mask = (X['potential_risk'] > 0) & (y == 0)
            positive_mask = ~non_fire_mask

        X_positive = X[positive_mask]
        y_positive = y[positive_mask]
        y_train_score_positive = y_train_score[positive_mask]

        X_non_fire = X[non_fire_mask]
        y_non_fire = y[non_fire_mask]
        y_train_score_non_fire = y_train_score[non_fire_mask]

        # Sample non-fire data
        print(nb, len(X_non_fire))
        nb = min(len(X_non_fire), nb)
        if self.n_run == 1:
            sampled_indices = np.random.RandomState(42).choice(len(X_non_fire), nb, replace=False)
        else:
            sampled_indices = np.random.RandomState().choice(len(X_non_fire), nb, replace=False)
    
        X_non_fire_sampled = X_non_fire.iloc[sampled_indices] if isinstance(X, pd.DataFrame) else X_non_fire[sampled_indices]

        if not is_unknowed_risk:
            y_non_fire_sampled = y_non_fire.iloc[sampled_indices] if isinstance(y, pd.DataFrame) else y_non_fire[sampled_indices]
            y_train_score_non_fire_sampled = y_train_score_non_fire.iloc[sampled_indices] if isinstance(y_train_score, pd.Series) else y_train_score_non_fire[sampled_indices]
        else:
            print(np.unique(X_non_fire.iloc[sampled_indices]['potential_risk']))
            y_non_fire_sampled = X_non_fire.iloc[sampled_indices]['potential_risk']
            #y_train_score_non_fire_sampled = X_non_fire.iloc[sampled_indices]['potential_risk']

        X_combined = pd.concat([X_positive, X_non_fire_sampled]) if isinstance(X, pd.DataFrame) else np.concatenate([X_positive, X_non_fire_sampled])
        y_combined = pd.concat([y_positive, y_non_fire_sampled]) if isinstance(y, pd.DataFrame) else np.concatenate([y_positive, y_non_fire_sampled])
        y_train_score_combined = pd.concat([y_train_score_positive, y_train_score_non_fire_sampled]) if isinstance(y, pd.DataFrame) else np.concatenate([y_train_score_positive, y_train_score_non_fire_sampled])

        # Update X and y for training
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        y_train_score_combined.reset_index(drop=True, inplace=True)

        return X_combined, y_combined, y_train_score_combined
    
    def search_samples_proportion(self, X, y, X_val, y_val, X_test, y_test, y_train_score=None, y_val_score=None, y_test_score=None, is_unknowed_risk=False):

        if y_test_score is None:
            y_test_score = np.copy(y_test)

        if not is_unknowed_risk:
            test_percentage = np.arange(0.05, 1.05, 0.05)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []

        data_log = None
        if False:
            if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
        else:
            if (self.dir_log / 'metrics.pkl').is_file():
                print(f'Load metrics')
                find_log = True
                try:
                    data_log = read_object('metrics.pkl', self.dir_log)
                except:
                    self.metrics = {}
                    data_log = None
            else:
                xs = [0, 10]
                for x in xs:
                    other_model = f'{self.model_type}_search_full_{x}_all_one_{self.target_name}_{self.task_type}_{self.loss}'
                    if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                        data_log = read_object('metrics.pkl', self.dir_log)
                    if data_log is not None:
                        break
        
        print(self.metrics)
        print(f'data_log : {data_log}')
        if data_log is not None:
            try:
                self.metrics = data_log
                #test_percentage = self.metrics['test_percentage']
                under_prediction_score_scores = self.metrics['under_prediction_scores']
                over_prediction_score_scores = self.metrics['over_prediction_scores']
                iou_scores = self.metrics['iou_score']
            except Exception as e:
                print(e)
                self.metrics = {}
                data_log = None
                pass
            
            #test_percentage, under_prediction_score_scores, over_prediction_score_scores, iou_scores = data_log[0], data_log[1], data_log[2], data_log[3]
        
        doSearch = True
        if data_log is not None: #and self.n_run == data_log['n_run']:
            for i in range(0, len(iou_scores) - 1):
                try:
                    if data_log[test_percentage[i]]['iou_val'] > data_log[test_percentage[i + 1]]['iou_val']:
                        print(f"Last score {data_log[test_percentage[i]]['iou_val']} current score {data_log[test_percentage[i + 1]]['iou_val']} -> {test_percentage[i]}")
                        doSearch = True
                except Exception as e:
                    print(e)
                    doSearch = True
                    break

            if doSearch:
                start_test = np.argmax(iou_scores)
                #start_test = len(under_prediction_score_scores) - 1
        else:
            start_test = 0

        if doSearch:
            last_score = -math.inf if start_test == 0 else iou_scores[start_test - 1]
            for i in range(start_test, test_percentage.shape[0]):
                tp = test_percentage[i]
                if not is_unknowed_risk:
                    nb = int(tp * len(y[y[self.target_name] == 0]))
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y == 0)]))

                print(f'Trained with {tp} -> {nb} sample of class 0')

                if tp in self.metrics.keys():
                    change_value = True
                else:
                    change_value = False

                if True:
                    self.metrics[tp] = {}
                    self.metrics[tp]['f1'] = []
                    self.metrics[tp]['prec'] = []
                    self.metrics[tp]['recall'] = []
                    self.metrics[tp]['iou'] = []
                    self.metrics[tp]['iou_val'] = []
                    self.metrics[tp]['normalized_iou'] = []
                    self.metrics[tp]['normalized_f1'] = []
                else:
                    continue

                for run in range(self.n_run):
                    print(f'Run {run}')

                    X_combined, y_combined, y_train_score_combined = self.split_dataset(X, y, y_train_score, nb, is_unknowed_risk)
                    print(y_combined.shape, y_train_score_combined.shape)
                    print(f'Train mask X shape: {X_combined.shape}, y shape: {y_combined.shape}')

                    copy_model = copy.deepcopy(self)
                    if 'dual' in self.loss:
                        params_model = copy_model.best_estimator_.kwargs
                        params_model['y_train_origin'] = y_train_score_combined
                        copy_model.best_estimator_.update_params(params_model)

                    copy_model.under_sampling = 'full'

                    copy_model.fit(X_combined, y_combined, X_val, y_val, X_test=X_test, y_test=y_test, y_test_score=y_test_score, \
                                y_train_score=y_train_score, y_val_score=y_val_score, training_mode='normal', \
                                    optimization='skip', grid_params=None, fit_params={}, cv_folds=10)
                    
                    prediction = copy_model.predict(X_val)
                    metrics_run = evaluate_metrics(y_val, self.target_name, prediction)
                    self.metrics[tp]['iou_val'].append(metrics_run['iou'])
                    under_prediction_score_value = under_prediction_score(y_val_score, prediction)
                    over_prediction_score_value = over_prediction_score(y_val_score, prediction)

                    prediction = copy_model.predict(X_test)
                    metrics_run = evaluate_metrics(y_test, self.target_name, prediction)
                    self.metrics[tp]['f1'].append(metrics_run['f1'])
                    self.metrics[tp]['iou'].append(metrics_run['iou'])
                    self.metrics[tp]['recall'].append(metrics_run['recall'])
                    self.metrics[tp]['prec'].append(metrics_run['prec'])
                    self.metrics[tp]['normalized_iou'].append(metrics_run['normalized_iou'])
                    self.metrics[tp]['normalized_f1'].append(metrics_run['normalized_f1'])
                    
                    #under_prediction_score_value = under_prediction_score(y_val, prediction)
                    #over_prediction_score_value = over_prediction_score(y_val, prediction)
                    #iou = iou_score(y_val, prediction)
                    
                if self.n_run == 1:
                    self.metrics[tp]['var_f1'] = 0
                    self.metrics[tp]['IC_f1'] = (0, 0)
                    self.metrics[tp]['var_iou'] = 0
                    self.metrics[tp]['IC_iou'] = (0, 0)
                    self.metrics[tp]['var_Normalized_f1'] = 0
                    self.metrics[tp]['IC_Normalized_f1'] = (0, 0)
                    self.metrics[tp]['var_Normalized_iou'] = 0
                    self.metrics[tp]['IC_Normalized_iou'] = (0, 0)
                else:
                    # Calcul de la variance pour chaque métrique
                    f1_variance = np.var(self.metrics[tp]['f1'])
                    iou_variance = np.var(self.metrics[tp]['iou'])
                    normalized_f1_variance = np.var(self.metrics[tp]['normalized_f1'])
                    normalized_iou_variance = np.var(self.metrics[tp]['normalized_iou'])
                    
                    # Calcul de l'IC 95% pour chaque métrique
                    f1_ic = calculate_ic95(self.metrics[tp]['f1'])
                    iou_ic = calculate_ic95(self.metrics[tp]['iou'])
                    normalized_f1_ic = calculate_ic95(self.metrics[tp]['normalized_f1'])
                    normalized_iou_ic = calculate_ic95(self.metrics[tp]['normalized_iou'])
                    
                    # Ajout de la variance et de l'IC dans le dictionnaire avec des clefs spécifiques pour chaque métrique
                    self.metrics[tp]['var_f1'] = f1_variance
                    self.metrics[tp]['IC_f1'] = f1_ic
                    self.metrics[tp]['var_iou'] = iou_variance
                    self.metrics[tp]['IC_iou'] = iou_ic
                    self.metrics[tp]['var_Normalized_f1'] = normalized_f1_variance
                    self.metrics[tp]['IC_Normalized_f1'] = normalized_f1_ic
                    self.metrics[tp]['var_Normalized_iou'] = normalized_iou_variance
                    self.metrics[tp]['IC_Normalized_iou'] = normalized_iou_ic
                
                iou = np.mean(self.metrics[tp]['iou_val'])
                if not change_value:
                    iou_scores.append(iou)
                    #under_prediction_score_scores.append(under_prediction_score_value)
                    #over_prediction_score_scores.append(over_prediction_score_value)
                else:
                    iou_scores[i] = iou
                    #under_prediction_score_scores[i] = under_prediction_score_value
                    #over_prediction_score_scores[i] = over_prediction_score_value

                #save_object([test_percentage[:len(under_prediction_score_scores)], under_prediction_score_scores, over_prediction_score_scores, iou_scores], 'test_percentage_scores.pkl', self.dir_log)
                save_object(self.metrics, 'metrics.pkl', self.dir_log)
                
                print(f'Metrics achieved : {self.metrics[tp]}')

                if iou > last_score:
                    last_score = iou
                else:
                    break
        
        # Find the index where the two scores cross (i.e., where the difference changes sign)
        index_max = np.argmax(iou_scores)
        best_tp = test_percentage[index_max]
        self.metrics['best_tp'] = best_tp
        if doSearch:
            self.metrics['iou_score'] = iou_scores
            self.metrics['under_prediction_scores'] = under_prediction_score_scores
            self.metrics['over_prediction_scores'] = over_prediction_score_scores
            self.metrics['test_percentage'] = test_percentage
            self.metrics['run'] = self.n_run
            save_object(self.metrics, 'metrics.pkl', self.dir_log)

        print(f'All metrics : {self.metrics}')
        print(f'Metrics achieved with best percentage {best_tp} {self.metrics[best_tp]}')
        
        return best_tp

    def fit(self, X, y, X_val, y_val, X_test=None, y_test=None, y_val_score=None, y_train_score=None, y_test_score=None, training_mode='normal', optimization='skip', grid_params=None, fit_params={}, cv_folds=10):
        """
        Train the model.

        Parameters:
        - X: Training data.
        - y: Labels for the training data.
        - grid_params: Parameters to optimize
        - optimization: Optimization method to use ('grid' or 'bayes').
        - fit_params: Additional parameters for the fit function.
        """

        features = list(X.columns)
        if 'weight' in features:
            features.remove('weight')
        if 'potential_risk' in features:
            features.remove('potential_risk')

        if self.task_type == 'binary':
            y_test = (y_test > 0).astype(int) if y_test is not None else None
            y_test_score = (y_test_score > 0).astype(int) if y_test_score is not None else None
            y_val = (y_val > 0).astype(int)

        #importance_df = calculate_and_plot_feature_importance(X[features], y, features, self.dir_log / '../importance', self.target_name, task_type=self.task_type)
        #importance_df = calculate_and_plot_feature_importance_shapley(X[features], y, features, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        if self.nbfeatures != 'all':
            features = featuresAll[:int(self.nbfeatures)]
        
        #################################################### Handle over sampling ##############################################

        if self.under_sampling != 'full':
            old_shape = X.shape
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(y[y[self.target_name] > 0])
                except ValueError:
                    print(f'{self.under_sampling} with undefined factor, set to 1 -> {len(y[y > 0])}')
                    nb = len(y[y > 0])

                    X, y, y_score = self.split_dataset(X, y, y_train_score, nb, False)

                print(f'Original shape {old_shape}, Train mask X shape: {X.shape}, y shape: {y.shape}')

            elif self.under_sampling.find('search') != -1 or 'percentage' in self.under_sampling:

                if self.under_sampling.find('search') != -1:
                    ################# No risk sample ##################
                    best_tp_0 = self.search_samples_proportion(X, y, X_val, y_val, X_test, y_test, y_train_score, y_val_score, y_test_score, False)
                else:
                    vec = self.under_sampling.split('-')
                    best_tp_0 = float(vec[1])

                nb = int(best_tp_0 * len(y[y[self.target_name] == 0]))
                X, y, y_score = self.split_dataset(X, y, y_train_score, nb, False)
                if 'dual' in self.loss:
                    params_model = self.best_estimator_.kwargs
                    params_model['y_train_origin'] = y_score
                    self.best_estimator_.update_params(params_model)

                print(f'Original shape {old_shape}, Train mask X shape: {X.shape}, y shape: {y.shape}')

            else:
                raise ValueError(f'Unknow value of under_sampling -> {self.under_sampling}')
        
        ######################################### Handle under sampling #####################################

        if self.over_sampling == 'full':
            pass

        elif 'smote' in self.over_sampling:
            smote_coef = int(self.over_sampling.split('-')[1])
            y_negative = y[y[self.target_name] == 0].shape[0]
            
            y_one = max(y[y[self.target_name] == 1].shape[0], min(y[y[self.target_name] == 1].shape[0] * smote_coef, y_negative))
            y_two = max(y[y[self.target_name] == 2].shape[0], min(y[y[self.target_name] == 2].shape[0] *smote_coef, y_negative))
            y_three = max(y[y[self.target_name] == 3].shape[0], min(y[y[self.target_name] == 3].shape[0] * smote_coef, y_negative))
            y_four = max(y[y[self.target_name] == 4].shape[0], min(y[y[self.target_name] == 4].shape[0] * smote_coef, y_negative))
            
            if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
                """y_negative = y[y == 0].shape[0]
                y_one = y_negative * 0.01
                y_two = y_negative * 0.01
                y_three = y_negative * 0.01
                y_four = y_negative * 0.01
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})"""
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            X, new_y = smote.fit_resample(X, y[self.target_name])
            y = pd.DataFrame(index=np.arange(new_y.shape[0]))
            y[self.target_name] = new_y
        else:
            raise ValueError(f'Unknow value of under_sampling -> {self.over_sampling}')

        #print(f'Positive data after treatment : {positive_shape}, {y[y  > 0].shape}')
        for uy in np.unique(y[self.target_name]):
            print(f'Number of {uy} class : {y[y[self.target_name] == uy].shape}') 
        
        X_train = X[features]
        y_train = y
        sample_weight = X['weight']

        X_val = X_val[features]

        ##################################### Search for features #############################################
    
        if training_mode == 'features_search':
            features_selected, final_score = self.fit_by_features(X_train, y_train, X_val, y_val, X_test, y_test, featuresAll, sample_weight, False)
            self.features_selected, self.final_score = features_selected, final_score
            X = X[features_selected]
            
            df_features = pd.DataFrame(index=np.arange(0, len(features_selected)))
            df_features['features'] = features_selected
            df_features['iou_score'] = final_score
            save_object(df_features, f'{self.name}_features.csv', self.dir_log)
        else:
            self.features_selected = features
            df_features = pd.DataFrame(index=np.arange(0, len(self.features_selected)))
            df_features['features'] = self.features_selected
            df_features['iou_score'] = np.nan
            save_object(df_features, f'{self.name}_features.csv', self.dir_log)

        X_train = X[self.features_selected]

        ###############################################" Fit pararams depending of the model type #####################################
        
        fit_params = self.update_fit_params(X_val, y_val[self.target_name], sample_weight, self.features_selected)

        """if self.loss in ['softprob-dual', 'softmax-dual']:
            new_y_train = np.zeros((y_train.shape[0], 2))
            new_y_train[:, 0] = y_train
            new_y_train[:, 1] = y_train_score
            y_train = np.copy(new_y_train)
            del new_y_train"""

        """new_y_val = np.zeros((y_val.shape[0], 2))
            new_y_val[:, 0] = y_val
            new_y_val[:, 1] = y_val_score
            y_val = np.copy(new_y_val)
            del new_y_val

            new_y_test = np.zeros((y_test.shape[0], 2))
            new_y_test[:, 0] = y_test
            new_y_test[:, 1] = y_test_score
            y_test = np.copy(new_y_test)
            del new_y_test"""

        ######################################################### Training #########################################
        if optimization == 'grid':
            assert grid_params is not None
            grid_search = GridSearchCV(self.best_estimator_, grid_params, scoring=self.get_scorer(), cv=cv_folds, refit=False)
            grid_search.fit(X_train, y_train[self.target_name], **fit_params)
            best_params = grid_search.best_params_
            self.cv_results_ = grid_search.cv_results_
        elif optimization == 'bayes':
            assert grid_params is not None
            param_list = []
            for param_name, param_values in grid_params.items():
                if isinstance(param_values, list):
                    param_list.append((param_name, param_values))
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    param_list.append((param_name, param_values))
                else:
                    raise ValueError(
                        "Unsupported parameter type in grid_params. Expected list or tuple of size 2.")

            # Configure the parameter space for BayesSearchCV
            param_space = {}
            for param_name, param_range in param_list:
                if isinstance(param_range[0], int):
                    param_space[param_name] = Integer(
                        param_range[0], param_range[-1])
                elif isinstance(param_range[0], float):
                    param_space[param_name] = Real(param_range[0], param_range[-1], prior='log-uniform')

            opt = Optimizer(param_space, base_estimator='GP', acq_func='gp_hedge')
            bayes_search = BayesSearchCV(self.best_estimator_, opt, scoring=self.get_scorer(), cv=cv_folds, Refit=False)
            bayes_search.fit(X_train, y_train[self.target_name], **fit_params)
            best_params = bayes_search.best_estimator_.get_params()
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            best_params = self.best_estimator_.get_params()
            self.best_estimator_.fit(X_train, y_train[self.target_name], **fit_params)
        elif optimization == 'cv':
            self.cv_Model = cross_validate(self.best_estimator_, X_train, y_train[self.target_name], cv=cv_folds, scoring=iou_score, return_estimator=True)
            self.best_estimator_ = self.cv_Model['']
        else:
            raise ValueError("Unsupported optimization method")
        
        ########################### Fit final model on the entire dataset ###########################
        if optimization != 'skip' and optimization != 'cv':
            self.set_params(**best_params)
            self.best_estimator_.fit(X_train, y_train[self.target_name], **fit_params)

        ############################# Post process fit ##############################
        if self.post_process is not None:
            pred_val = self.best_estimator_.predict(X_val[self.features_selected])
            self.post_process.fit(pred_val, y_val[self.target_name], **{
                'disp':True,
                'method': 'bfgs'
            })

        self.metrics['final'] = evaluate_metrics(y_test, self.target_name, self.predict(X_test))
        #print(self.metrics)
        
    def get_model(self):
        return self.best_estimator_

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Data to predict labels for.

        Returns:
        - Predicted labels.
        """
        res = self.best_estimator_.predict(X[self.features_selected])
        if self.post_process is not None:
            res = self.post_process.predict(res)
        return res
        
    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        - X: Data to predict probabilities for.

        Returns:
        - Predicted probabilities.
        """

        if hasattr(self.best_estimator_, "predict_proba"):
            if isinstance(X, np.ndarray):
                return self.best_estimator_.predict_proba(X.reshape(-1, len(self.features_selected)))
            elif isinstance(X, torch.Tensor):
                return self.best_estimator_.predict_proba(X.detach().cpu().numpy().reshape(-1, len(self.features_selected)))
            return self.best_estimator_.predict_proba(X[self.features_selected])
        elif self.name.find('gam') != -1:
            res = np.zeros((X.shape[0], 2))
            res[:, 1] = self.best_estimator_.predict(X[self.features_selected])
            return res
        else:
            raise AttributeError(
                "The chosen model does not support predict_proba.")
        
    def get_all_scores(self, X, y_true, y_fire=None):
        prediction = self.predict(X)
        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            scores = calculate_signal_scores_for_training(prediction, y_true, y_fire)
        elif self.task_type == 'regression':
            raise ValueError(f'get_all_scores Not implemented for regression yet')
        return scores

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        #return calculate_signal_scores(y, y_pred)
        
        return iou_score(y, y_pred)
    
        if self.loss == 'area':
            return calculate_signal_scores(y, y_pred)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_params(self, deep=True):
        """
        Get the model's parameters.

        Parameters:
        - deep: If True, return the parameters for this model and nested models.

        Returns:
        - Dictionary of parameters.
        """
        params = {'model': self.best_estimator_,
                  'loss': self.loss, 'name': self.name}
        if deep and hasattr(self.best_estimator_, 'get_params'):
            deep_params = self.best_estimator_.get_params(deep=True)
            params.update(deep_params)
        return params

    def set_params(self, **params):
        """
        Set the model's parameters.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        best_estimator_params = {}
        for key, value in params.items():
            if key in ['model', 'loss', 'name']:
                setattr(self, key, value)
            else:
                best_estimator_params[key] = value

        if best_estimator_params != {}:
            self.best_estimator_.set_params(**best_estimator_params)

        return self

    def get_scorer(self):
        """
        Return the scoring function as a string based on the chosen loss function.
        """
        return iou_score
        if self.loss == 'logloss':
            return 'neg_logloss'
        elif self.loss == 'hinge_loss':
            return 'hinge'
        elif self.loss == 'accuracy':
            return 'accuracy'
        elif self.loss == 'mse':
            return 'neg_mean_squared_error'
        elif self.loss == 'rmse':
            return 'neg_root_mean_squared_error'
        elif self.loss == 'rmsle':
            return 'neg_root_mean_squared_log_error'
        elif self.loss == 'poisson':
            return 'neg_mean_poisson_deviance'
        elif self.loss == 'huber_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'log_cosh_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'tukey_biweight_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'exponential_loss':
            return 'neg_mean_squared_error'
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        
    def plot_features_importance(self, X_set, y_set, outname, dir_output, mode='bar', figsize=(50, 25), limit=10):
        """
        Display the importance of features using feature permutation.

        Parameters:
        - X_set: Data to evaluate feature importance.
        - y_set: Corresponding labels.
        - names: Names of the features.
        - outname : Name of the test set
        - dir_output: Directory to save the plot.
        - mode : mustache (boxplot) or bar.
        """
        names = X_set.columns
        result = permutation_importance(self.best_estimator_, X_set, y_set,
                                        n_repeats=10, random_state=42, n_jobs=-1, scoring=self.get_scorer())
        importances = result.importances_mean
        indices = importances.argsort()[-limit:]
        if mode == 'bar':
            plt.figure(figsize=figsize)
            plt.title(f"Permutation importances {self.name}")
            plt.bar(range(len(importances[indices])),
                    importances[indices], align="center")
            plt.xticks(range(len(importances[indices])), [
                       names[i] for i in indices], rotation=90)
            plt.xlim([-1, len(importances[indices])])
            plt.ylabel(f"Decrease in {self.get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) /
                        f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        elif mode == 'mustache' or mode == 'boxplot':
            plt.figure(figsize=figsize)
            plt.boxplot(importances[indices].T, vert=False, whis=1.5)
            plt.title(f"Permutation Importances {self.name}")
            plt.axvline(x=0, color="k", linestyle="--")
            plt.xlabel(f"Decrease in {self.get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) /
                        f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        else:
            raise ValueError(f'Unknown {mode} for ploting features importance but feel free to add new one')
        
        save_object(result, f"{outname}_permutation_importances.pkl", dir_output)

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode="bar", figsize=(50, 25), samples=None, samples_name=None):
        """
        Generate SHAP explanations for model predictions.
        :param df_set: Dataset to analyze.
        :param outname: Output file prefix.
        :param dir_output: Directory to save explanations.
        :param mode: SHAP visualization type ("bar" or "beeswarm").
        :param figsize: Figure size.
        :param samples: Specific samples to analyze.
        :param samples_name: Names of the samples.
        """
        # Compute SHAP values
        if self.model_type in ['xgboost', 'catboost']:
            explainer = shap.TreeExplainer(self.best_estimator_.model_)
        else:
            explainer = shap.Explainer(self.best_estimator_)
        
        shap_values = explainer.shap_values(df_set)
        n_classes = 5 if self.task_type == 'classification' else 2

        # Vérifier si la sortie SHAP est multi-classes
        if n_classes == 1:
            shap_values = shap_values[:, :, np.newaxis]

        df_features = []
        # Pour chaque classe, calculer et sauvegarder les résultats SHAP
        for class_idx in range(n_classes):
            # Calcul des valeurs SHAP moyennes et écarts-types
            shap_mean_abs = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            shap_std_abs = np.std(np.abs(shap_values[:, :, class_idx]), axis=0)

            df_shap = pd.DataFrame({
                "mean_abs_shap": shap_mean_abs,
                "stdev_abs_shap": shap_std_abs,
                "name": self.features_selected
            }).sort_values("mean_abs_shap", ascending=False)

            df_shap['class'] = class_idx
            df_features.append(df_shap)

            # Visualisation globale (summary_plot) pour chaque classe
            plt.figure(figsize=figsize)
            if mode == 'bar':
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=df_set,
                    feature_names=self.features_selected,
                    plot_type='bar',
                    show=False
                )
            elif mode == 'beeswarm':
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=df_set,
                    feature_names=self.features_selected,
                    show=False
                )

            print(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            plt.savefig(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            plt.close()

            # Visualisations spécifiques aux échantillons (force_plot)
            if samples is not None and samples_name is not None:

                for i, sample in enumerate(samples):
                    plt.figure(figsize=figsize)
                    shap.force_plot(
                        explainer.expected_value[class_idx],
                        shap_values[sample, :, class_idx],
                        features=df.iloc[sample].values,
                        feature_names=self.features_name,
                        matplotlib=True,
                        show=False
                    )

                    plt.savefig(
                        dir_output / f"{outname}_class_{class_idx}_{samples_name[i]}_shapley.png",
                        bbox_inches='tight'
                    )
                    plt.close()

        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)

    def update_fit_params(self, X_val, y_val, sample_weight, features_selected):
        if self.model_type == 'xgboost':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'sample_weight': sample_weight,
                'verbose': False,
                #'early_stopping_rounds' : 15
            }

        elif self.model_type == 'catboost':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'sample_weight': sample_weight,
                'verbose': False,
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'ngboost':
            fit_params = {
                'X_val': X_val[features_selected],
                'Y_val': y_val,
                'sample_weight': sample_weight,
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'rf':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'dt':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'eval_sample_weight': [X_val['weight']],
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif self.model_type == 'svm':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'poisson':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'gam':
            fit_params = {
            'weights': sample_weight
            } 

        elif self.model_type ==  'linear':
            fit_params = {}
        
        elif self.model_type ==  'lg':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type ==  'ordered':
            fit_params = {
                'disp':False,
                'method': 'bfgs'
            }

        else:
            raise ValueError(f"Unsupported model model_type: {self.model_type}")
        
        return fit_params

    def log(self, dir_output):
        assert self.final_score is not None
        check_and_create_path(dir_output)
        plt.figure(figsize=(15,5))
        plt.plot(self.final_score)
        x_score = np.arange(len(self.features_selected))
        plt.xticks(x_score, self.features_selected, rotation=45)
        plt.savefig(self.dir_log / f'{self.name}.png')

##########################################################################################
#                                                                                        #
#                                   Tree                                                 #
#                                                                                        #
##########################################################################################

class ModelTree(Model):
    def __init__(self, model, model_type, loss='logloss', name='ModelTree', under_sampling='full', target_name='nbsinister'):
        """
        Initialize the ModelTree class.

        Parameters:
        - model: The base model to use (must follow the sklearn API and support tree plotting).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        super().__init__(model=model, model_type=model_type, loss=loss, name=name, under_sampling=under_sampling, target_name=target_name)

    def plot_tree(self, features_name=None, class_names=None, filled=True, outname="tree_plot", dir_output=".", figsize=(20, 20)):
        """
        Plot a tree for tree-based models.

        Parameters:
        - feature_names: Names of the features.
        - class_names: Names of the classes (for classification tasks).
        - filled: Whether to color the nodes to reflect the majority class or value.
        - outname: Name of the output file.
        - dir_output: Directory to save the plot.
        """
        if isinstance(self.best_estimator_, DecisionTreeClassifier) or isinstance(self.best_estimator_, DecisionTreeRegressor):
            # Plot for DecisionTree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(self.best_estimator_, feature_names=features_name,
                              class_names=class_names, filled=filled)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, RandomForestClassifier) or isinstance(self.best_estimator_, RandomForestRegressor):
            # Plot for RandomForest - only the first tree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(self.best_estimator_.estimators_[
                              0], feature_names=features_name, class_names=class_names, filled=filled)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, XGBClassifier) or isinstance(self.best_estimator_, XGBRegressor):
            # Plot for XGBoost
            plt.figure(figsize=figsize)
            xgb_plot_tree(self.best_estimator_, num_trees=0)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, LGBMClassifier) or isinstance(self.best_estimator_, LGBMRegressor):
            # Plot for LightGBM
            plt.figure(figsize=figsize)
            lgb_plot_tree(self.best_estimator_, tree_index=0, figsize=figsize, show_info=[
                          'split_gain', 'internal_value', 'internal_count', 'leaf_count'])
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, NGBClassifier) or isinstance(self.best_estimator_, NGBRegressor):
            # Plot for NGBoost - not directly supported, but you can plot the base learner
            if hasattr(self.best_estimator_, 'learners_'):
                learner = self.best_estimator_.learners_[0][0]
                if hasattr(learner, 'tree_'):
                    plt.figure(figsize=figsize)
                    sklearn_plot_tree(
                        learner, feature_names=features_name, class_names=class_names, filled=filled)
                    plt.savefig(Path(dir_output) / f"{outname}.png")
                    plt.close('all')
                else:
                    raise AttributeError(
                        "The base learner of NGBoost does not support tree plotting.")
            else:
                raise AttributeError(
                    "The chosen NGBoost model does not support tree plotting.")
        else:
            raise AttributeError(
                "The chosen model does not support tree plotting.")
        
##########################################################################################
#                                                                                        #
#                                   Voting                                              #
#                                                                                        #
##########################################################################################

class ModelVoting(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', over_sampling='full', target_name='nbsinister', post_process=None, task_type='classification'):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.features = features
        self.name = name
        self.loss = loss
        X_train = None
        y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.target_name = target_name
        self.task_type = task_type

    def fit(self, X, y, X_val, y_val, X_test, y_test, training_mode='normal', optimization='skip', grid_params_list=None, fit_params_list=None, cv_folds=10, id_col=[], ids_columns=None):
        """
        Train each model on the corresponding data.

        Parameters:
        - X_list: List of training data for each model.
        - y_list: List of labels for the training data for each model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters to optimize for each model.
        - fit_params_list: List of additional parameters for the fit function for each model.
        - cv_folds: Number of cross-validation folds.
        """

        self.cv_results_ = []
        self.is_fitted_ = [True] * len(self.best_estimator_)
        self.weights_for_model = []
        self.weights_for_model_self = []
        self.weights_id_model = {}
        self.weights_id_model = {}
        if len(id_col) > 0:
            do_id_weight = True
            for id_tuple in id_col:
                id = id_tuple[0]
                vals = id_tuple[1]
                uvalues = np.unique(vals)
                self.weights_id_model[id] = {}
                for val in uvalues:
                    self.weights_id_model[id][val] = []
        else:
            do_id_weight = False

        targets = [col for col in y.columns if col not in ids_columns]
        for i, model in enumerate(self.best_estimator_):
            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name}')
            model.fit(X, y[ids_columns + [targets[i]]], X_val, y_val[ids_columns + [targets[i]]], X_test, y_test[ids_columns + [targets[i]]], y_train_score=y[targets[i]], y_val_score=y_val[targets[i]], y_test_score=y_test[targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)

            score_model = model.score(X_val, y_val[self.target_name])

            self.weights_for_model.append(score_model)
            print(f'Weight achieved : {score_model}')

            score_model = model.score(X_val, y_val[targets[i]])

            self.weights_for_model_self.append(score_model)
            print(f'Weight {targets[i]} achieved : {score_model}')

            if do_id_weight:
                for id_tuple in id_col:
                    id = id_tuple[0]
                    vals = id_tuple[1]
                    uvalues = np.unique(vals)
                    for val in uvalues:
                        mask = vals == val
                        score_model = model.score(X_val[mask], y_val[self.target_name][mask])
                        self.weights_id_model[id][val].append(score_model)
                    
                        #print(f'Weight {targets[i]} achieved on {id} {val}: {score_model}')
                    
        self.weights_for_model = np.asarray(self.weights_for_model)
        # Affichage des poids et des modèles
        print("\n--- Final Model Weights ---")
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            print(f"Model: {model.name}, Weight: {weight:.4f}")

        # Plot des poids des modèles
        model_names = [model.name for model in self.best_estimator_]
        plt.figure(figsize=(10, 10))
        plt.bar(model_names, self.weights_for_model, color='skyblue', edgecolor='black')
        plt.title('Model Weights', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)  # Rotation de 90° ici
        plt.tight_layout()
        plt.savefig(self.dir_log / 'weights_of_models.png')
        plt.close('all')

        save_object([model_names, self.weights_for_model, self.weights_for_model_self], f'weights.pkl', self.dir_log)
        
    def predict_with_weight(self, X, hard_or_soft='soft', weights_average='weight', weights2use=[], top_model='all'):
        
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)
        
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            models_to_mean = []
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.name not in models_list:
                    continue
                else:
                    pred = estimator.predict(X)
                    predictions.append(pred)

                models_to_mean.append(key[i])

            try:
                weights2use = weights2use[models_to_mean]
            except:
                pass
            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights2use)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba_with_weights(X, weights_average=weights_average, top_model=top_model, weights2use=weights2use)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions

    def predict_proba_with_weights(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', weights2use=[], id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        probas = []
        models_to_mean = []
        for i, estimator in enumerate(self.best_estimator_):
            if estimator.name not in models_list:
                continue
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                if proba.shape[1] != 5:
                    continue
                #print(estimator.name, np.asarray(probas).shape)
                models_to_mean.append(key[i])
                probas.append(proba)
            else:
                raise AttributeError(f"The model {estimator.name} does not support predict_proba.")
        try:
            weights2use = weights2use[models_to_mean]
        except:
            pass
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights2use)
        return aggregated_proba

    def predict(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask] = self.predict_with_weight(X[mask], hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction

        else:
            return self.predict_with_weight(X, hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

    def predict_proba(self, X, weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask] = self.predict_proba_with_weights(X[mask], hard_or_soft='soft', weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction

        else:
            return self.predict_proba_with_weights(X, hard_or_soft='soft', weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

    def aggregate_predictions(self, predictions_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        predictions_array = np.array(predictions_list)
        if len(weight2use) == 0 or weight2use is None:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weight2use.reshape(mask.shape[0], 1), axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weight2use[:, None], axis=0)
            aggregated_pred = weighted_sum / np.sum(weight2use)
            #aggregated_pred = np.max(predictions_array * weight2use[:, None], axis=0)
        
        return aggregated_pred

    def aggregate_probabilities(self, probas_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weight2use is None or len(weight2use) == 0:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weight2use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        #aggregated_proba = np.max(probas_array * weight2use[:, None, None], axis=0)
        return aggregated_proba

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
        return calculate_signal_scores(y_pred, y)
        if self.loss == 'area':
            return -smooth_area_under_prediction_loss(y, y_pred, loss=True)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_params(self, deep=True):
        """
        Get the ensemble model's parameters.

        Parameters:
        - deep: If True, return the parameters for this model and nested models.

        Returns:
        - Dictionary of parameters.
        """
        params = {'models': self.best_estimator_,
                  'loss': self.loss, 'name': self.name}
        if deep:
            for i, estimator in enumerate(self.best_estimator_):
                params.update({f'model_{i}': estimator})
                if hasattr(estimator, 'get_params'):
                    estimator_params = estimator.get_params(deep=True)
                    params.update({f'model_{i}__{key}': value for key, value in estimator_params.items()})
        return params
    
    def shapley_additive_explanation(self, X, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
        """
        Perform shapley additive explanation features on each estimator
        
        Parameters:
        - df_set_list : a list for len(self.best_estiamtor) size, with ieme element being the dataframe for ieme estimator 
        - outname : outname of the figure
        - mode : mode of ploting
        - figsize : figure size
        - samples : use for additional plot where the shapley additive explanation is done on each sample
        - samples_name : name of each sample 

        Returns:
        - None
        """

        for i, estimator in enumerate(self.best_estimator_):
            self.best_estimator_[i].shapley_additive_explanation(X[self.features_per_model[i]], f'{outname}_{i}', dir_output, mode, figsize, samples, samples_name)

    def set_params(self, **params):
        """
        Set the ensemble model's parameters.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        models_params = {}
        for key, value in params.items():
            if key in ['models', 'loss', 'name']:
                setattr(self, key, value)
            elif key.startswith('model_'):
                idx_and_param = key.split('__')
                if len(idx_and_param) == 1:
                    idx = int(idx_and_param[0].split('_')[1])
                    self.best_estimator_[idx] = value
                else:
                    idx = int(idx_and_param[0].split('_')[1])
                    param_name = idx_and_param[1]
                    if hasattr(self.best_estimator_[idx], 'set_params'):
                        self.best_estimator_[idx].set_params(**{param_name: value})
            else:
                # General parameter, set to all models
                for estimator in self.best_estimator_:
                    if hasattr(estimator, 'set_params'):
                        estimator.set_params(**{key: value})
        return self
    
    def log(self, dir_output):
        check_and_create_path(dir_output)
        print(self.features_per_model)
        for model_index in range(len(self.features_per_model)):
            plt.figure(figsize=(15,5))
            x_score = np.arange(len(self.features_per_model[model_index]))
            plt.plot(x_score, self.final_scores[model_index])
            plt.xticks(x_score, self.features_per_model[model_index], rotation=45)
            plt.savefig(self.dir_log / f'{self.best_estimator_[model_index].name}_{model_index}.png')
