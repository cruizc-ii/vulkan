from app.utils import EDP, SummaryEDP
from app.hazard import Hazard, Record
from unittest.case import TestCase
from app.strana import (
    AnalysisTypes,
    IDA,
    StructuralAnalysis,
    StructuralResultView,
    TimehistoryRecorder,
)
import numpy as np
from scipy.linalg import eigh

from app.criteria import CodeMassesPre, DesignCriterionFactory
from app.design import ReinforcedConcreteFrame, BuildingSpecification
from app.fem import FiniteElementModel, ShearModel
from .test import (
    DESIGN_FIXTURES_PATH,
    DESIGN_MODELS_PATH,
    STRANA_FIXTURES_PATH,
    LOSS_FIXTURES_PATH,
    RECORDS_DIR,
)
from app.utils import eigenvectors_similar


class ShearModelAnalysisTest(TestCase):
    """
    example 1281 from Chopra with unit masses.

    - [x] it should compute correct eigenvalues+eigenvectors
    - [x] it should return correct stiffness matrix + K_static
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:

        cls.path = STRANA_FIXTURES_PATH / "mdof-frame"
        cls.file = cls.path / "chopra1281-mdof-fem.yml"
        cls.fem = ShearModel.from_file(cls.file)
        cls.strana = StructuralAnalysis(results_path=cls.path, fem=cls.fem)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_static_stiffness_matrix(self) -> None:
        K_result = self.strana.K_static
        k = 12 * 0.083333333 * 1.0 / 1.0**3
        nDOF = 5
        K_true = (
            2 * k * np.eye(nDOF)
            - k * np.eye(nDOF, k=1)
            - k * np.eye(nDOF, k=-1)
            - np.zeros(nDOF)
        )
        K_true[4, 4] = k
        self.assertTrue(np.allclose(K_result, K_true))

    def test_modal_chopra(self) -> None:
        """
        vals, vecs = eigh(ktt, m) -> returns normalized vectors the way we need it!
        where ktt = Ke.loc[mass_ix, mass_ix] stiffness at mass nodes
        """
        # true_vectors = np.array(
        #     [
        #         [0.59688479, 0.54852873, 0.45573414, -0.32601868, 0.16989112],
        #         [0.54852873, 0.16989112, -0.32601868, 0.59688479, -0.45573414],
        #         [0.45573414, -0.32601868, -0.54852873, -0.16989112, 0.59688479],
        #         [0.32601868, -0.59688479, 0.16989112, -0.45573414, -0.54852873],
        #         [0.16989112, -0.45573414, 0.59688479, 0.54852873, 0.32601868],
        #     ]
        # )
        chopra_omegas_sq = np.array([0.285, 0.831, 1.31, 1.682, 1.919]) ** 2
        chopra_vectors = np.array(
            [
                [0.334, 0.641, 0.895, 1.078, 1.173],
                [-0.895, -1.173, -0.641, 0.334, 1.078],
                [1.173, 0.334, -1.078, -0.641, 0.895],
                [-1.078, 0.895, 0.334, -1.173, 0.641],
                [0.641, -1.078, 1.173, -0.895, 0.334],
            ]
        ).T  # as columns.
        m = np.eye(5)
        strana = StructuralAnalysis(results_path=self.path, fem=self.fem)
        ktt = strana.K_static
        vals, _ = eigh(ktt, m)
        self.assertTrue(np.allclose(chopra_omegas_sq, vals, rtol=0.02))
        m_chopra = 100 / 386.1 * m
        _, vecs = eigh(ktt, m_chopra)
        self.assertTrue(eigenvectors_similar(vecs, chopra_vectors, rtol=1e-3))

    def test_modal_results(self) -> None:
        strana = StructuralAnalysis(results_path=self.path, fem=self.fem)
        view = self.strana.modal()
        omegas_analysis = view.omegas
        vectors_analysis = view.vectors
        m = np.eye(5)
        ktt = strana.K_static
        vals, vecs = eigh(ktt, m)
        self.assertTrue(
            np.allclose(np.sqrt(vals), np.array(omegas_analysis), rtol=0.01)
        )
        self.assertTrue(
            eigenvectors_similar(np.array(vectors_analysis), vecs, rtol=0.01)
        )


class Chopra1326Test(TestCase):
    """
    example 1281 from Chopra.
    the unfortunate thing is that the normalized modal vectors
    are done with respect to a 100kips/g masses. (weird) instead of unit masses
    so our masses are
    g = 32.17 ft/s^2
    and we have to create our model that way
    masses = 3.1084.
    k = 31.54 kips/in ->  378.48 kips / ft -> 12 E I / L^3 -> with E=1 and L=12 -> Ix = 378.48 * 12**2 = 4541.76
    hs = [0, 12, 24, 36, 48, 60]

    it should
        - [x] compute correct eigenvalues
        - [x] return correct s_n (mass distribution factors)
        - [x] effective modal mass M* and modal height h*
        - [x] M* and h* satisfy:
            sum M* = sum mj
            sum h*M* = sum hj mj
            Vbn = M*n
            Mbn = h*n Vbn
            ... this is satisfed when having the same M* and h* as chopra
        - [x] return correct u_j and gamma_j (displacements and drifts)
        ... this is satisfied when having the same s_n as chopra.. since it depends on u_j
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:

        cls.path = STRANA_FIXTURES_PATH / "chopra-1326"
        cls.file = cls.path / "chopra1326-fem.yml"
        cls.fem = ShearModel.from_file(cls.file)
        cls.strana = StructuralAnalysis(results_path=cls.path, fem=cls.fem)
        cls.view = cls.strana.modal()

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_periods(self) -> None:
        """it should produce the correct periods"""
        periods = self.view.periods
        true_periods = np.array([2.0, 0.6852, 0.43460, 0.3383, 0.2966])
        self.assertTrue(np.allclose(periods, true_periods, rtol=0.01))

    def test_modal_expansion_of_inertial_forces(self) -> None:
        """it should produce the correct modal expansion of inertial forces"""
        view = self.fem.get_and_set_eigen_results(self.path)
        s = view.inertial_forces
        true_s = (
            np.array(
                [
                    [0.356, 0.684, 0.956, 1.150, 1.252],
                    [0.301, 0.394, 0.215, -0.112, -0.362],
                    [0.208, 0.059, -0.191, -0.113, 0.159],
                    [0.106, -0.088, -0.033, 0.116, -0.063],
                    [0.029, -0.049, 0.053, -0.040, 0.015],
                ]
            ).T
            @ self.fem._mass_matrix
        )
        self.assertTrue(np.allclose(s, true_s, rtol=0.01))

    def test_effective_masses_and_heights(self) -> None:
        true_M = self.fem._mass_matrix @ np.array([4.398, 0.436, 0.121, 0.037, 0.008])
        true_h = 12 * np.array([3.51, -1.2, 0.76, -0.59, 0.52])
        view = self.fem.get_and_set_eigen_results(self.path)
        M = view.effective_masses
        h = view.effective_heights
        self.assertTrue(np.allclose(M, np.diag(true_M), rtol=0.05))
        self.assertTrue(np.allclose(h, true_h, rtol=0.05))

    def test_shears(self) -> None:
        m = self.fem._mass_matrix[0, 0]
        true_V5 = m * np.array([1.252, -0.362, 0.159, -0.063, 0.015])
        true_Vb = m * np.array([4.398, 0.436, 0.121, 0.037, 0.008])
        view = self.fem.get_and_set_eigen_results(self.path)
        V = view.shears
        self.assertTrue(np.allclose(V[-1], true_V5, rtol=0.01))
        self.assertTrue(np.allclose(V[0], true_Vb, rtol=0.03))

    def test_moments(self) -> None:
        m = self.fem._mass_matrix[0, 0]
        true_M = 12 * m * np.array([15.45, -0.525, 0.092, -0.022, 0.004])
        view = self.fem.get_and_set_eigen_results(self.path)
        M = view.overturning_moments
        self.assertTrue(np.allclose(M[0], true_M, rtol=0.05))
        # pretty print matrix.
        # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        #     print(M)

    # def test_modal_equalities(self) -> None:
    #     """it should assert that
    #     Vb = M*
    #     Mb = h* M*
    #     sum M* = total mass
    #     sum M*h* = hj mj
    #     """
    #     results = self.fem.get_and_set_eigen_results(self.path)
    #     self.assertTrue(False)

    # def test_displacements_and_drifts(self) -> None:
    #     view = self.fem.get_and_set_eigen_results(self.path)
    #     self.assertTrue(False)

    def test_generate_rayleigh_damping(self) -> None:
        """it should compute a0, a1 such that for a given damping % z_i
        all the modes that contribute significantly (in modal mass participation) are close to z_i"""
        # TODO; choose best mode, not 0 and 1.
        view = self.fem.get_and_set_eigen_results(self.path)
        a0, a1 = self.fem._rayleigh()
        omegas = np.array(view.omegas)
        modal_damping = a0 / (2 * omegas) + a1 * omegas / 2
        self.assertTrue(np.allclose(modal_damping[0:1], [0.05, 0.05]))

    def test_run_standalone_timehistory(self) -> None:
        """it should generate results that are easily queried and correct."""
        rec = Record(str(RECORDS_DIR / "elCentro.csv"))
        view = self.strana.timehistory(rec, scale=-32.17, gravity_loads=False)
        view.to_file()

        u5x = view.view_displacements_envelope(6, 1)
        self.assertAlmostEqual(u5x, 6.85 / 12, delta=1e-2)

        # # roof lateral force. (5th storey static lateral force (not element shears!))
        # Vs = view.view_node_reactions_envelope(6, 1)
        # self.assertAlmostEqual(Vs, 28.7, delta=1.0)
        # self.assertAlmostEqual(Vs, 35.217, delta=1.0)

        # env = view.view_base_reactions_envelope()
        # Vb = env[EDP.shear.value]
        # self.assertAlmostEqual(Vb.values.sum(), 73.278, delta=1)

        # # overturning moments.. the sum of all node moments
        # moments = view.view_moments()
        # self.assertAlmostEqual(moments.values.sum(axis=1).max(), 2597, delta=20)

    # def test_load_view_directly_from_file(self) -> None:
    #     rec = Record(str(RECORDS_DIR / "elCentro.csv"))
    #     view = self.strana.timehistory(rec, scale=-32.17, gravity_loads=False)
    #     view.to_file()

    #     # load Results directly and check against what we got from the run
    #     view2 = StructuralResultView.from_file(self.path / "timehistory")
    #     # self.assertAlmostEqual(
    #     #     Vb.values.sum(),
    #     #     view.view_base_reactions_envelope().values.sum(),
    #     #     delta=1e-5,
    #     # )
    #     self.assertTrue(False)

    # def test_static_analysis(self) -> None:
    #     """given lateral force vector, perform static analyses and return view to results"""
    #     # TODO; overturning moments, base shear timehistory
    #     self.assertTrue(False)

    # def test_pushover(self) -> None:
    #     """given a shape vector, perform u-roof-controlled pushover, return view to results """
    #     self.assertTrue(False)


class IDATest(TestCase):
    """
    it should be able to load designs with multiple fems
    + hazard instance
    and perform runs with simple interface
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.IDA_TESTS = STRANA_FIXTURES_PATH / "ida-tests"
        cls.hazard_path = str((cls.IDA_TESTS / "ida-hazard-test.yml").resolve())
        cls.design_path = str((cls.IDA_TESTS / "ida-design-test.yml").resolve())

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        """it should init as a class"""

        ida = IDA(
            name="test",
            hazard_abspath=self.hazard_path,
            design_abspath=self.design_path,
            start=0.2,
            stop=0.8,
            step=0.3,
        )
        self.assertIsInstance(ida._hazard, Hazard)
        self.assertIsInstance(ida._design, BuildingSpecification)
        intensities = np.array([0.2, 0.5, 0.8])
        self.assertTrue(np.allclose(ida._intensities[0], intensities))

    def test_persistence(self):
        """it should save to file and be able to read it again"""

        ida = IDA(
            name="ida-to-yaml",
            hazard_abspath=self.hazard_path,
            design_abspath=self.design_path,
            start=0.1,
            stop=0.8,
            step=0.1,
        )
        filepath = self.IDA_TESTS
        ida.to_file(filepath)
        new = IDA.from_file(filepath / f"{ida.name}.yml")
        self.assertEqual(new.start, 0.1)
        self.assertEqual(new.stop, 0.8)
        self.assertEqual(new.step, 0.1)
        self.assertEqual(new.hazard_abspath, self.hazard_path)
        self.assertEqual(new.design_abspath, self.design_path)
        intensities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.assertTrue(np.allclose(new._intensities[0], intensities))
        self.assertIsInstance(new._hazard, Hazard)
        self.assertIsInstance(new._design, BuildingSpecification)

    def test_run(self):
        """
        it should run gravity + timehistory, it should return a view to result objects
        """
        ida = IDA(
            name="ida-loss-tests",
            hazard_abspath=self.hazard_path,
            design_abspath=self.design_path,
            start=0.1,
            stop=0.2,
            step=0.1,
        )
        results_df = ida.run(
            results_dir=self.IDA_TESTS,
            run_id="__single_run",
        )
        self.assertEqual(
            set(results_df.record.values),
            set(["98_SCT190985NS.csv", "6_JA241093EW.csv"]),
        )
        self.assertEqual(
            set(results_df.columns),
            {
                "intensity",
                "record",
                "path",
                "intensity_str",
                "accel",
                "sup",
                "inf",
                "freq",
                "collapse",
            }
            | set(SummaryEDP.list()),
        )
        pfas = results_df["pfa"]
        self.assertTrue(all(pfas.apply(lambda lst: len(lst) == 3)))
        ida.to_file(LOSS_FIXTURES_PATH)
