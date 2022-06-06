from unittest.case import TestCase
from pathlib import Path
from app.design import ReinforcedConcreteFrame


TEST_PATH = Path(__file__).resolve().parent
DESIGN_FIXTURES_PATH = TEST_PATH / "design_fixtures"
FEM_FIXTURES_PATH = TEST_PATH / "fem_fixtures"
RESULTS_DIR = TEST_PATH / "results"
RECORDS_DIR = TEST_PATH.parent / "records"
HYP_DESIGN_FIXTURES_PATH = DESIGN_FIXTURES_PATH / "hypothesis"
DESIGN_MODELS_PATH = TEST_PATH.parent / "models" / "design"


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
        # cls.file = cls.path / "spec-test.yml"

    def test_write_to_file(self):
        """it should write all writable fields + FEM"""
        spec = ReinforcedConcreteFrame(
            name="spec-test",
            bays=[5.0],
            storeys=[3.0, 3.0],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        spec.to_file(DESIGN_FIXTURES_PATH)
        spec.fem.to_file(DESIGN_FIXTURES_PATH / "fem-from-spec.yml")
        spec.to_file(DESIGN_MODELS_PATH)

        new = ReinforcedConcreteFrame.from_file(
            DESIGN_FIXTURES_PATH / f"{spec.name}.yml"
        )

        # self.assertEqual(len(new.fems), len(DesignCriterionFactory.default_criteria()))
        # self.assertTrue(all([isinstance(f, FiniteElementModel) for f in new.fems]))
