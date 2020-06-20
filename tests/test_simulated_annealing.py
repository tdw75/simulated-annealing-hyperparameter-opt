from unittest import TestCase

from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

from sa_hyperopt.hyperparameter_configs import get_lightgbm_parameters
from sa_hyperopt.simulated_annealing import SimulatedAnnealingSearchCV


class SimulatedAnnealingSearchCVTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        results = load_breast_cancer()
        cls.X = pd.DataFrame(results['data'], columns=list(results['feature_names']))
        cls.y = pd.Series(results['target'])
        cls.params = get_lightgbm_parameters('binary')
        cls.model = LGBMClassifier()
        cls.args = {'cv': 5,
                    'scoring': 'brier_score_loss',
                    'n_jobs': -1}
        cls.sim = SimulatedAnnealingSearchCV(cls.model, param_distributions=cls.params, seed=0, **cls.args)
        cls.sim.fit(cls.X, cls.y, initial_temperature=10)

    def test_that_search_returns_expected_score(self):
        score_expected = 0.0182
        self.assertEqual(score_expected, round(-self.sim.best_score_, 4))

    def test_that_update_value_does_not_return_old_value(self):
        values = [0, 1]
        old_value = 1
        new_value_expected = 0
        new_value = self.sim.update_value(values, old_value)

        self.assertEqual(new_value_expected, new_value)

    def test_that_weights_are_correctly_calculated(self):
        params_dict = {'a': [0, 1, 2],
                       'b': [5, 10]}
        weight_a = 3 / 5
        weight_b = 2 / 5
        weights = self.sim.calculate_weights_for_param_distributions(params_dict, params_dict.keys())
        self.assertListEqual([weight_a, weight_b], weights)
