from app.assets import RISK_MODELS_DIR, LognormalRisk
from app.criteria import CodeMassesPre, DesignCriterionFactory
from unittest.case import TestCase
from app.design import ReinforcedConcreteFrame
from app.fem import ElasticBeamColumn, ElementTypes, FiniteElementModel
import numpy as np
from .test import DESIGN_FIXTURES_PATH, DESIGN_MODELS_PATH


class LognormalAssetTest(TestCase):
    def test_expected_value_and_variance(self):
        drift = 0.01
        drifts = drift * np.ones(10000)
        probs = [1 - 0.732, 0.732 - 0.5, 0.5 - 0.18, 0.18, 0, 0, 0]
        medians = [1e-12, 0.0837, 0.2101, 0.3263, 0.803, 1.994, 3.49]
        std_log = [0.01, 0.59, 0.59, 0.63, 0.67, 0.3, 0.03]
        means = [med * np.exp(0.5 * std**2) for med, std in zip(medians, std_log)]
        variances = [
            mean**2 * (np.exp(std**2) - 1) for mean, std in zip(means, std_log)
        ]
        expected_value = sum([prob * expected for prob, expected in zip(probs, means)])
        risk: LognormalRisk = LognormalRisk.from_file(
            RISK_MODELS_DIR / "original_concrete_column_risk.yml"
        )
        computed = risk.expected_loss(drift)
        self.assertAlmostEqual(expected_value, computed, 2)
        mean_squared = sum(
            [
                prob * (mean**2 + var)
                for prob, mean, var in zip(probs, means, variances)
            ]
        )
        exact = mean_squared - expected_value**2
        simulated = np.var(risk.simulate_losses(drifts))
        self.assertAlmostEqual(exact, simulated, delta=0.01)
