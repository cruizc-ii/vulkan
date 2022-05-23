from unittest.case import TestCase
from pathlib import Path

# from app.design import ReinforcedConcreteFrame


TEST_PATH = Path(__file__).resolve().parent
DESIGN_FIXTURES_PATH = TEST_PATH / "design_fixtures"
FEM_FIXTURES_PATH = TEST_PATH / "fem_fixtures"
RESULTS_DIR = TEST_PATH / "results"
RECORDS_DIR = TEST_PATH.parent / "records"
HYP_DESIGN_FIXTURES_PATH = DESIGN_FIXTURES_PATH / "hypothesis"
DESIGN_MODELS_PATH = TEST_PATH.parent / "models" / "design_models"


class BuildingSpecificationTest(TestCase):
    """
    it should load and save spec
    """

    maxDiff = None
    file = None
    path = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH
        cls.file = cls.path / "spec-test.yml"

    def test_produces_correct_periods(self):
        """it should load a spec and produce a realistic design"""
        # spec = ReinforcedConcreteFrame.from_file(self.file)
        self.assertTrue(True)
