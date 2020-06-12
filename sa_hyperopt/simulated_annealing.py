import logging
import random

import numpy as np
from sklearn.model_selection import cross_validate
import pandas as pd


class SimulatedAnnealingSearchCV:

    def __init__(self, model, param_distributions: dict, scoring: str, n_jobs=1, cv: int = 5, seed: int = 0):
        self.model = model
        self.params = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params_ = self.params.copy()
        self.best_estimator_ = self.model
        self.best_score_ = None
        self.score_tracker = []
        self.iterations = None
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    @staticmethod
    def calculate_reset_threshold(initial_temperature: float, cooling_rate: float) -> int:
        approx_iterations = int(-np.log(initial_temperature) / (np.log(1 - cooling_rate))) + 1
        return int(approx_iterations / 5 + 0.5)

    @staticmethod
    def calculate_weights_for_param_distributions(params_dict: dict, keys: list) -> list:
        weights = [len(params_dict[k]) for k in keys]
        total_weight = sum(weights)
        return [x / total_weight for x in weights]

    def save_best_hyperparameter_config(self, score_best: float, params_best: dict, best_estimator):
        self.best_score_ = score_best
        self.best_params_ = params_best.copy()
        self.best_estimator_ = best_estimator

    def reset_to_current_best(self, threshold: int):
        params_current = self.best_params_.copy()
        score_current = self.best_score_
        iter_since_best = 0
        logging.info(
            " No global improvements in the last {} iterations. Search restarted from best configuration so far".format(
                threshold))
        return params_current, score_current, iter_since_best

    def fit(self, X: pd.DataFrame, y: pd.Series, cooling_rate: float = 0.01, initial_temperature: float = 10.0):
        params_current = {k: random.choice(v) for k, v in self.params.items()}
        keys = [k for k, v in self.params.items() if len(v) > 1]
        weights = self.calculate_weights_for_param_distributions(self.params, keys)

        self.best_params_ = params_current.copy()
        self.model.set_params(**params_current)
        cv_results = cross_validate(self.model, X=X, y=y, n_jobs=self.n_jobs, cv=self.cv, scoring=self.scoring)
        score_current = cv_results['test_score'].mean()
        self.best_score_ = score_current

        past_configs = []
        past_configs.append(params_current.values())

        temperature = initial_temperature
        no_improvement_threshold = self.calculate_reset_threshold(initial_temperature, cooling_rate)
        iter_since_best = 0
        self.iterations = 0

        while temperature > 1:
            key = random.choices(keys, weights)[0]
            params_temp = params_current.copy()
            params_temp[key] = update_value(self.params[key], params_current[key])

            if params_temp.values() in past_configs:
                continue
            past_configs.append(params_temp.values())

            self.model.set_params(**params_temp)
            cv_results = cross_validate(self.model, X=X, y=y, n_jobs=self.n_jobs, cv=self.cv, scoring=self.scoring,
                                        return_estimator=True)
            score_temp = cv_results['test_score'].mean()
            params_new, score_new, prob = metropolis(params_current, params_temp, score_current, score_temp,
                                                     temperature)

            self.score_tracker.append(score_new)
            self.iterations += 1
            print_log(score_temp, score_current, self.best_score_, score_new, prob, self.iterations)

            if score_temp >= self.best_score_:
                self.save_best_hyperparameter_config(score_temp, params_new, cv_results['estimator'][0])
                iter_since_best = 0
            else:
                iter_since_best += 1

            params_current, score_current = params_new, score_new
            temperature *= (1 - cooling_rate)

            if iter_since_best >= no_improvement_threshold:
                params_current, score_current, iter_since_best = self.reset_to_current_best(no_improvement_threshold)

        logging.info(" Best score found was {:.4f}".format(-self.best_score_))


def metropolis(config_old: dict, config_new: dict, score_old: float, score_new: float, temp: float):
    delta = (score_old - score_new) * 750
    prob = np.exp(-delta / temp)

    if score_new >= score_old:
        return config_new, score_new, prob
    elif np.random.rand() <= prob:
        return config_new, score_new, prob
    else:
        return config_old, score_old, prob


def update_value(values: list, old_value):
    values_list = values.copy()
    values_list.remove(old_value)
    new_value = random.choice(values_list)

    return new_value


def print_log(score_temp, score_current, score_best, score_new, prob, iterations):
    if score_temp > score_current:
        logging.info(" Iteration {}: Local improvement from {:.4f} to {:.4f}, parameters updated".format(
            iterations, score_current, score_temp))
        if score_temp > score_best:
            logging.info(
                " -> Global improvement from {:.4f} to {:.4f}, parameters updated".format(score_best, score_temp))
    elif score_new != score_current:
        logging.info(" Iteration {}: No improvement from {:.4f} to {:.4f} but parameters updated".format(
            iterations, score_current, score_temp))
        logging.debug(" Probability threshold for acceptance was {:.3f}".format(prob))
    else:
        logging.info(" Iteration {}: No improvement from {:.4f} to {:.4f}, parameters unchanged".format(
            iterations, score_current, score_temp))
        logging.debug(" Probability threshold for acceptance was {:.3f}".format(prob))
