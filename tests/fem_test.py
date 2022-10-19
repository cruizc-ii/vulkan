from app.strana import Recorder, StructuralResultView
from pathlib import Path
from app.fem import BilinFrame, ShearModel, PlainFEM
from unittest.case import TestCase
from app.criteria import CodeMassesPre, DesignCriterionFactory
from app.design import ReinforcedConcreteFrame
from app.fem import FiniteElementModel
from .test import DESIGN_FIXTURES_PATH, DESIGN_MODELS_PATH, FEM_FIXTURES_PATH
import numpy as np

from app.utils import eigenvectors_similar


"""

these tests should be able to load fem.yml and produce opensees-compatible str(fem)

"""


class ShearModelTest(TestCase):
    """
    it should load MDOF from a file correctly
    - [x] compute lateral resistances correctly from the nodes+elements definition
    - [x] write to a .yml correctly
    - [x] write a correct model to .tcl with equalDOF
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = FEM_FIXTURES_PATH / "mdof-frame-unit-stiffness.yml"
        with open(FEM_FIXTURES_PATH / "expected-mdof-opensees.tcl", "r") as file:
            cls.expected_model_str = file.read()

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_load_from_yaml(self) -> None:
        mdof = ShearModel.from_file(self.file)
        self.assertEqual(mdof.model_str, self.expected_model_str)

    def test_to_file(self):
        mdof = ShearModel.from_file(self.file)
        mdof.to_file(FEM_FIXTURES_PATH / "mdof-fem-after-process.yml")

    def test_to_tcl(self) -> None:
        mdof = ShearModel.from_file(self.file)
        mdof.to_tcl(FEM_FIXTURES_PATH / "mdof-opensees.tcl")


class SkyCivElasticFrameFEMTest(TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = FEM_FIXTURES_PATH / "skyciv-1st-1bay"
        cls.file = cls.path / "skyciv-elastic-rigid-beams-frame-test.yml"
        with open(cls.path / "expected-elastic-rigid-beams-skyciv.tcl", "r") as file:
            cls.expected_model_str = file.read()

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_load_from_yaml(self) -> None:
        """should produce correct .tcl string repr"""
        frame = PlainFEM.from_file(self.file)
        self.assertEqual(str(frame), self.expected_model_str)

    def test_standalone_gravity_analysis(self) -> None:
        frame = PlainFEM.from_file(self.file)
        view: StructuralResultView = frame.gravity(self.path)
        reactions = view.reactions()
        V = reactions["V"]
        P = reactions["P"]
        M = reactions["M"]
        expected_P = [150, 150]
        expected_V = [37.24, -37.24]
        expected_M = [-31.03, 31.03]
        self.assertTrue(
            np.allclose(P, expected_P, rtol=1e-3),
        )
        self.assertTrue(
            np.allclose(V, expected_V, rtol=1e-3),
        )
        self.assertTrue(
            np.allclose(M, expected_M, rtol=1e-3),
        )

    def test_static_analysis(self) -> None:
        """it should run a static analysis with lateral forces provided (WITH GRAVITY if model has masses)"""
        frame = PlainFEM.from_file(self.file)
        view: StructuralResultView = frame.static(self.path, forces_per_storey=[200.0])
        reactions = view.reactions()
        V = reactions["V"]
        P = reactions["P"]
        M = reactions["M"]
        expected_P = [90.65, 209.35]
        expected_V = [-62.76, -137.24]
        expected_M = [129.95, 192.0]
        self.assertTrue(
            np.allclose(P, expected_P, rtol=1e-2),
        )
        self.assertTrue(
            np.allclose(V, expected_V, rtol=1e-2),
        )
        self.assertTrue(
            np.allclose(M, expected_M, rtol=1e-2),
        )


class ElastoplasticFrameTest(TestCase):
    maxDiff = None
    path = None
    file = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = FEM_FIXTURES_PATH / "nonlin-frame"
        cls.file = cls.path / "elastoplastic-frame-test-input.yml"
        with open(cls.path / "expected-elastoplastic-frame.tcl", "r") as file:
            cls.expected_model_str = file.read()

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_standalone_gravity_analysis(self) -> None:
        frame = BilinFrame.from_file(self.file)
        view = frame.gravity(self.path)
        reactions = view.reactions_env()
        V = reactions["V"].values.flatten()
        P = reactions["P"].values.flatten()
        M = reactions["M"].values.flatten()
        Py = 1440
        # why isnt the distribution like this???
        # expected_P = [Py / 4, Py / 2, Py / 4]
        # self.assertTrue(
        #     np.allclose(P, expected_P, rtol=0.2),
        # )
        self.assertEqual(len(P), 3)
        self.assertAlmostEqual(sum(P), Py, delta=5.0)
        self.assertEqual(len(V), 3)
        self.assertEqual(len(M), 3)

    def test_static_analysis(self) -> None:
        """it should run a static analysis with lateral forces provided (WITH GRAVITY if model has masses)"""
        frame = BilinFrame.from_file(self.file)
        view = frame.static(self.path, forces_per_storey=[200.0])
        reactions = view.reactions_env()
        V = reactions["V"].values.flatten()
        P = reactions["P"].values.flatten()
        M = reactions["M"].values.flatten()
        self.assertAlmostEqual(sum(V), 400, 2)
        self.assertEqual(len(P), 3)
        self.assertEqual(len(M), 3)

    def test_elastic_eigen_equals_plastic_eigen(self):
        frame = PlainFEM.from_file(self.file)
        view = frame.get_and_set_eigen_results(self.path)
        Te = view.periods
        Phi_e = view.vectors
        frame = BilinFrame.from_file(self.file)
        view = frame.get_and_set_eigen_results(self.path)
        Tp = view.periods
        Phi_p = view.vectors

        self.assertTrue(eigenvectors_similar(Tp, Te))
        self.assertTrue(eigenvectors_similar(Phi_p, Phi_e))

    def test_pushover_elastic_range(self) -> None:
        frame = BilinFrame.from_file(self.file)
        view = frame.pushover(self.path, drift=0.01)
        reactions = view.reactions_env()
        V = reactions["V"].values.flatten()
        roof_disp = view.peak_roof_disp()
        expected_roof_disp = 0.07
        self.assertAlmostEqual(roof_disp, expected_roof_disp, 2)
        self.assertEqual(len(V), 3)

    def test_pushover_plastic_range(self) -> None:
        """it should converge and have Vb = Vy = 2 sum My/L"""
        target_drift = 0.05
        frame = BilinFrame.from_file(self.file)
        view: StructuralResultView = frame.pushover(self.path, drift=target_drift)
        reactions = view.reactions_env()
        V = reactions["V"].values.flatten()
        Vb = view.peak_base_shear()
        self.assertTrue(np.allclose(Vb, V))
        V = reactions["V"].values.flatten()
        expected_V = 400 * np.array([2, 2, 2]) / 3.5
        roof_disp = view.peak_roof_disp()
        expected_roof_disp = target_drift * frame.height
        self.assertAlmostEqual(roof_disp, expected_roof_disp, 3)
        self.assertAlmostEqual(sum(Vb), 685.7, delta=10.0)
        self.assertTrue(
            np.allclose(Vb, expected_V, rtol=1e-1),
        )

    def test_generates_pushover_fig(self) -> None:
        frame = BilinFrame.from_file(self.file)
        fig = frame.pushover_figs(self.path, drift=0.10)
        self.assertIsNotNone(fig)
        fig = frame.pushover_figs(self.path, drift=0.10)
        self.assertIsNotNone(fig)
        fig = frame.pushover_figs(self.path, drift=0.10)
        self.assertIsNotNone(fig)
        fig = frame.pushover_figs(self.path, drift=0.10)
        self.assertIsNotNone(fig)
