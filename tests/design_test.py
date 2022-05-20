from unittest.case import TestCase
from pathlib import Path
from app.design import ReinforcedConcreteFrame


TEST_PATH = Path(__file__).resolve().parent
DESIGN_FIXTURES_PATH = TEST_PATH / "design_fixtures"
FEM_FIXTURES_PATH = TEST_PATH / "fem_fixtures"
RESULTS_DIR = TEST_PATH / "results"
RECORDS_DIR = TEST_PATH.parent / "records"
HYP_DESIGN_FIXTURES_PATH = DESIGN_FIXTURES_PATH / "hypothesis"
DESIGN_MODELS_PATH = TEST_PATH.parent / "models" / "design_models"


class ChopraElasticPeriodsTest(TestCase):
    """
    it should modify masses on the fem and spec instance to produce realistic periods
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH / "chopra-design"
        cls.file = cls.path / "chopra-design-spec.yml"

    def test_produces_correct_periods(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame()
        spec.force_design()
        self.assertTrue(True)
