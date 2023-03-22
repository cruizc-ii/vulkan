from app.loss import LossAggregator
from app.assets import RISK_MODELS_DIR, LognormalRisk
from unittest.case import TestCase
from app.strana import IDA
import numpy as np
from .test import LOSS_FIXTURES_PATH


class LognormalExpectedLossTest(TestCase):
    """
    this tests makes sure that the simulation of damage states and losses
    from fragility+vulnerability definition is correct.
    """

    maxDiff = None
    path = None
    risk = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = RISK_MODELS_DIR
        cls.risk = LognormalRisk.from_file(
            RISK_MODELS_DIR / "generic_nsd_risk_test.yml"
        )

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        return super().tearDown()

    def test_expected_loss(self):
        # given a fixed drift 0.01
        # the expected loss given a drift is
        # E(L|u) = sum_dsi E(L|dsi) Pr(dsi | u)
        # E(L) is the mean of a lognormal parametrized by median and stdlog of the values
        # mean = median * exp(1/2 stdlog**2)
        exact = np.array(
            [
                (1 - 0.732) * 1e-10,
                +(0.732 - 0.5) * 0.00307 * np.exp(0.5 * 0.59**2),
                +(0.5 - 0.18) * 0.0083 * np.exp(0.5 * 0.59**2),
            ]
        )
        drift = 0.01
        computed = self.risk.expected_loss(drift)
        self.assertAlmostEqual(computed, exact.sum(), 2)

    def test_simulate_expected_loss(self):
        drift = 0.01
        expected_loss = self.risk.expected_loss(drift)
        drifts = drift * np.ones(10000)
        # simulate 1000 drifts
        losses = self.risk.simulate_losses(drifts)
        avg = np.average(losses)
        # todo@carlo law of large numbers
        self.assertAlmostEqual(avg, expected_loss, 2)


class LossComputationTest(TestCase):
    """
    make sure that the UI computation behaves correctly when
    we have an IDA_analysis_file we simulate N accels and just use
    LossView to compute all losses for all assets
    and write those computations to .csv
    and the assets + statistics (summary) to the .yml
    """

    maxDiff = None
    NUM_SIMULATIONS = 3
    path = None
    ida_file = None
    ida = None
    loss_summary_name = None
    agg = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = LOSS_FIXTURES_PATH
        cls.ida_file = str((cls.path / "ida-loss-tests.yml").resolve())
        cls.ida = IDA.from_file(cls.ida_file)
        cls.loss_summary_name = "loss-view-test"
        cls.agg = LossAggregator(
            ida_model_path=cls.ida_file,
            name=cls.loss_summary_name,
        )
        cls.agg.run()

    def test_simulate_losses_write_to_file(self):
        """it should know to use the correct SummaryEDP for each asset"""
        self.agg.to_file(self.path)
        summary = LossAggregator.from_file(self.path / f"{self.loss_summary_name}.yml")
        self.assertEqual(self.agg.to_dict, summary.to_dict)

    def test_aggregate_by_floor(self):
        """it should know how to aggregate by storey, name, category"""
        df = self.agg.aggregate_src_df(self.agg.loss_models_df, "floor")
        self.assertEqual(list(df.columns), [1, 2, 3, "total"])

    def test_aggregate_by_category(self):
        df = self.agg.aggregate_src_df(self.agg.loss_models_df, "category")
        self.assertEqual(
            list(df.columns),
            ["contents", "nonstructural", "structural", "total"],
        )

    def test_aggregate_by_name(self):
        df = self.agg.aggregate_src_df(self.agg.loss_models_df, "name")
        self.assertTrue("total" in df.columns)
        self.assertTrue(len(list(df.columns)) > 2)

    # def test_deaggregate_by_collapse(self):
    #     self.assertTrue(False)
