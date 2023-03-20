from unittest.case import TestCase
from app.criteria import (
    CodeMassesPre,
    DesignCriterionFactory,
    ForceBasedPre,
    LoeraPreArctan,
    LoeraPre,
    ShearStiffnessRetryPre,
    EulerShearPre,
)
from app.design import ReinforcedConcreteFrame
from app.fem import FiniteElementModel
from .test import DESIGN_FIXTURES_PATH, DESIGN_MODELS_PATH
import numpy as np


class BuildingSpecificationTest(TestCase):
    """
    it should load and save spec to .yml
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
        self.assertDictEqual(spec.to_dict, new.to_dict)
        self.assertEqual(len(new.fems), len(DesignCriterionFactory.public_options()))
        self.assertTrue(all([isinstance(f, FiniteElementModel) for f in new.fems]))


class EulerShearPreDesignTest(TestCase):
    """
    it should set inertias and radii to realistic values
    and do a reasonable job at grouping them by sqrt(storeys)
    """

    file = None
    path = None
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH
        cls.file = cls.path / "euler-design-spec.yml"

    def test_produces_correct_stiffnesses(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame.from_file(self.file)
        spec.force_design(DESIGN_FIXTURES_PATH)
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 2
        )  # groups correctly
        self.assertTrue(len(np.unique(spec.fem.storey_radii)) == 2)  # groups correctly

    def test_produces_correct_groupings_2(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-2st",
            storeys=[4.5, 3.0, 3.0],
            bays=[6.0, 6.0],
            damping=0.10,
            masses=[150.0],
            design_criteria=[EulerShearPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 3
        )  # groups correctly

        self.assertTrue(len(np.unique(spec.fem.storey_radii)) == 3)  # groups correctly

    def test_produces_correct_groupings_3(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-4st",
            storeys=[4.5, 3.0, 3.0, 3.0],
            bays=[6.0, 6.0],
            damping=0.10,
            masses=4 * [25.0],
            design_criteria=[EulerShearPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        spec.force_design(DESIGN_FIXTURES_PATH)
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 2
        )  # groups correctly

        self.assertTrue(len(np.unique(spec.fem.storey_radii)) == 2)  # groups correctly

    def test_produces_correct_groupings_4(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-4st",
            storeys=[4.5] + 7 * [3.0],
            bays=[6.0, 6.0],
            damping=0.10,
            masses=8 * [25.0],
            design_criteria=[EulerShearPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        spec.force_design(DESIGN_FIXTURES_PATH)
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 4
        )  # groups correctly

        self.assertTrue(len(np.unique(spec.fem.storey_radii)) == 4)  # groups correctly


class LoeraPreDesignTest(TestCase):
    """
    it should set inertias and radii to realistic values
    using the given masses
    """

    file = None
    path = None
    maxDiff = None
    RTOL = 0.15

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH
        cls.file = cls.path / "loera-design-spec.yml"

    def test_produces_correct_stiffnesses(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame.from_file(self.file)
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_col_radii = [0.12, 0.12, 0.08, 0.08]
        expected_beam_radii = [0.09, 0.06]
        self.assertTrue(
            np.allclose(spec.fem.column_radii, expected_col_radii, rtol=self.RTOL)
        )
        self.assertTrue(
            np.allclose(spec.fem.beam_radii, expected_beam_radii, rtol=self.RTOL)
        )

    def test_produces_correct_stiffnesses_3st(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame(
            name="code-masses-test-1",
            storeys=[4.2, 3.5, 3.5],
            bays=[6.0, 6.0],
            design_criteria=[LoeraPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_col_radii = 3 * [0.40] + 3 * [0.30] + 3 * [0.20]
        expected_storey_Ixs = [0.053, 0.024, 0.0059]
        expected_storey_stiffnesses = [206_000, 159_000, 39_700]
        self.assertTrue(
            np.allclose(spec.fem.column_radii, expected_col_radii, rtol=self.RTOL)
        )
        self.assertTrue(
            np.allclose(spec.fem.storey_inertias, expected_storey_Ixs, rtol=self.RTOL)
        )
        self.assertTrue(
            np.allclose(
                spec.fem.storey_stiffnesses, expected_storey_stiffnesses, rtol=self.RTOL
            )
        )


class CodeMassesPreDesignTest(TestCase):

    file = None
    path = None
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH

    def test_produces_correct_masses_1(self):
        design = ReinforcedConcreteFrame(
            name="code-masses-test-1",
            storeys=[8.0],
            bays=[5.0, 8.0, 5.0],
            design_criteria=[CodeMassesPre.__name__],
        )
        expected_masses = [
            (5 + 8 + 5) ** 2 * CodeMassesPre.CODE_UNIFORM_LOADS_kPA / 9.81
        ]
        design.force_design(self.path)
        # self.assertTrue(np.allclose(design.masses, [10.0], rtol=1e-3))
        self.assertTrue(np.allclose(expected_masses, design.fem.masses, rtol=1e-3))

    def test_produces_correct_masses_2(self):
        design = ReinforcedConcreteFrame(
            name="code-masses-test-2",
            storeys=[3.0, 3.0, 4.0, 5.0],
            bays=[5.0, 3.0, 5.0],
            design_criteria=[CodeMassesPre.__name__],
        )
        expected_masses = [
            (5 + 3 + 5) ** 2 * CodeMassesPre.CODE_UNIFORM_LOADS_kPA / 9.81
        ]
        design.force_design(self.path)
        # self.assertTrue(np.allclose(design.masses, [10.0, 10.0, 10.0, 10.0], rtol=1e-3))
        self.assertTrue(np.allclose(expected_masses, design.fem.masses, rtol=1e-3))

    def test_produces_correct_masses_3(self):
        design = ReinforcedConcreteFrame(
            name="code-masses-test-3",
            storeys=10 * [5.0],
            bays=5 * [4.0],
            design_criteria=[CodeMassesPre.__name__],
        )
        expected_masses = [20**2 * CodeMassesPre.CODE_UNIFORM_LOADS_kPA / 9.81]
        design.force_design(self.path)
        # self.assertTrue(np.allclose(design.masses, 10 * [10.0], rtol=1e-3))
        self.assertTrue(np.allclose(expected_masses, design.fem.masses, rtol=1e-3))


class ShearStiffnessRetryPreTest(TestCase):
    """
    it should return a somewhat realistic PREDesign
    both in stiffnesses and masses.
    the spec of what is 'realistic' is captured in the following formula for the period
    period = num_storeys/8
    masses = 1T/m2
    """

    maxDiff = None
    rtol_periods = 0.3
    rtol_weight = 0.3

    def test_produces_realistic_periods_and_stiffnesses_1storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-1st-1",
            storeys=[8.0],
            bays=[6.0, 6.0, 6.0, 6.0],
            damping=0.50,
            design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_1storeys_2(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-1st-2",
            storeys=[3.0],
            bays=[6.0, 6.0, 6.0, 0.6],
            damping=0.50,
            masses=[10.0],
            design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_1storeys_3(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-1st-3",
            storeys=[3.0],
            bays=[6.0, 6.0, 6.0, 0.6],
            damping=0.50,
            design_criteria=[LoeraPreArctan.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_3storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-3st",
            storeys=[4.5, 3.2, 3.2],
            bays=[6.0, 6.0],
            damping=0.10,
            design_criteria=[LoeraPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 3
        )  # groups correctly
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_4storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-4st",
            storeys=[4.5, 3.0, 3.0, 3.0],
            bays=[6.0, 6.0],
            damping=0.10,
            # masses=[100.0],
            # design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
            design_criteria=[LoeraPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 2
        )  # groups correctly
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_9storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-9st",
            storeys=[4.5] + 8 * [3.0],
            bays=[8.0, 5.0],
            damping=0.10,
            masses=[144.0],
            design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 3
        )  # groups correctly
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_10storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-10st",
            storeys=[4.5] + 9 * [3.0],
            bays=[3.0, 8.0, 3.0],
            damping=0.10,
            masses=[144.0],
            design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 4
        )  # groups correctly
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check

    def test_produces_realistic_periods_and_stiffnesses_20storeys_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test-stiffness-retry-pre-20st",
            storeys=[4.5] + 19 * [3.0],
            bays=5 * [6.0],
            damping=0.36,
            masses=[1800],
            design_criteria=[EulerShearPre.__name__, ShearStiffnessRetryPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        self.assertAlmostEqual(
            spec.fem.periods[0],
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            len(np.unique(spec.fem.storey_inertias)) == 4
        )  # groups correctly
        self.assertTrue(all(spec.fem.periods[0] > spec.fem.periods[1:]))  # sanity check


class ForcePreDesignTest(TestCase):
    """
    it should return a somewhat realistic PREDesign
    both in stiffnesses and masses.

    the spec of what is 'realistic' is captured in the following formula for the period

    period = 0.1*num_storeys i.e. 4 storeys ~ 0.4 s main period
    all periods below are smaller just for sanity check.

    weight is somewhat realistic area * num_storeys * 1 t/m2
    """

    maxDiff = None
    rtol_periods = 0.2
    rtol_weight = 0.2

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH / "force-based"
        cls.file = cls.path / "force-pre-design-spec.yml"

    def test_produces_realistic_periods_and_stiffnesses_1storey_1(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5],
            bays=[6.0, 6.0],
            damping=0.05,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_1storey_2(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[3.0],
            bays=[8.0],
            damping=0.05,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_2(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 3.0],
            bays=[3.0, 6.0, 3.0],
            damping=0.15,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_4(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame.from_file(self.file)
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_6(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 4.5] + 4 * [3.0],
            bays=[3.0, 7.0, 7.0, 3.0],
            damping=0.22,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_10(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 4.5] + 8 * [3.0],
            bays=[3.0, 7.0, 7.0],
            damping=0.12,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_15(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 4.5] + 13 * [3.0],
            bays=[3.0, 6.0, 6.0, 3.0],
            damping=0.3,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_20(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 4.5] + 18 * [3.0],
            bays=[6, 8, 8, 6],
            damping=0.01,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )

    def test_produces_realistic_periods_and_stiffnesses_30(self):
        spec = ReinforcedConcreteFrame(
            name="force-based-design-test",
            storeys=[4.5, 4.5, 4.5] + 27 * [3.0],
            bays=[6, 6, 6, 6],
            damping=0.05,
            design_criteria=[ForceBasedPre.__name__],
        )
        spec.force_design(DESIGN_FIXTURES_PATH)
        expected_period = spec.miranda_fundamental_period
        expected_weight = (
            CodeMassesPre.CODE_UNIFORM_LOADS_kPA * spec.width**2 * spec.num_storeys
        )
        self.assertAlmostEqual(
            spec.fem.period,
            expected_period,
            delta=expected_period * self.rtol_periods,
        )
        self.assertTrue(
            np.allclose(spec.fem.weight, expected_weight, rtol=self.rtol_weight)
        )


class Chopra1326RSADesign(TestCase):
    """
    pp. 573 and 576 Chopra.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH / "chopra1326RSA"
        cls.file = cls.path / "chopra-rsa-spec.yml"

    def test_computes_modal_forces_using_spectra(self):
        """it should load a spec and produce a realistic design"""
        spec = ReinforcedConcreteFrame.from_file(self.file)
        spec.fems = spec.force_design(DESIGN_FIXTURES_PATH)
        fem = spec.fem
        self.assertTrue(
            np.allclose(fem.periods, [2.0, 0.6852, 0.4346, 0.3383, 0.2966], rtol=0.01)
        )
        expected_forces = np.array(
            [
                [4.9, 9.4, 13.14, 15.82, 17.21],
                [16.934, 22.18, 12.11, -6.31, -20.38],
                [16.925, 4.82, -15.554, -9.25, 12.92],
                [8.33, -6.92, -2.58, 9.064, -4.951],
                [2.19, -3.684, 4.01, -3.061, 1.141],
            ]
        ).T
        self.assertTrue(
            np.allclose(
                spec.fem.extras["forces"], expected_forces, rtol=0.1
            )  # 10% tolerance because of possible spectra differences.
        )

    def test_computes_design_forces_SRSS(self):
        first_storey_force = 25.87
        spec = ReinforcedConcreteFrame.from_file(self.file)
        spec.fems = spec.force_design(DESIGN_FIXTURES_PATH)
        last_storey_force = 30.07
        self.assertTrue(
            np.allclose(
                spec.fem.extras["design_forces"][0], first_storey_force, rtol=1e-2
            )
        )
        self.assertTrue(
            np.allclose(
                spec.fem.extras["design_forces"][-1], last_storey_force, rtol=1e-2
            )
        )


class CDMXDesignTest(TestCase):
    """
    it should create a RC elastic frame design
    and that is realistic in strengths for nonlinear analysis
    """

    maxDiff = None
    path = None
    raw_file = None
    designed_file = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DESIGN_FIXTURES_PATH / "cdmx-design"
        cls.raw_file = cls.path / "kaushik-raw-input.yml"
        cls.designed_file = cls.path / "kaushik-designed.yml"

    def test_design(self):
        """contain My and Vy needed for nonlin implementation."""
        spec = ReinforcedConcreteFrame.from_file(self.raw_file)
        spec.force_design(DESIGN_FIXTURES_PATH)
        spec.to_file(self.path)
        Vd = spec.fem.extras["design_shears"]
        self.assertTrue(np.allclose(sum(Vd), 400, atol=30.0))

    # def test_correct_nonlin_behavior(self):
    #     target_drift = 0.10
    #     spec = ReinforcedConcreteFrame.from_file(self.designed_file)
    #     view = spec.fem.pushover(self.path, drift=target_drift)
    #     roof_disp = view.peak_roof_disp()
    #     expected_roof_disp = spec.fem.height*target_drift
    #     self.assertAlmostEqual(roof_disp, expected_roof_disp, 3)
    #     Vb = view.peak_base_shear()
    #     self.assertAlmostEqual(sum(Vb), 685.7, delta=30.0)
    #     expected_V = 400 * np.array([2, 2, 2, 2]) / 4.4
    #     self.assertTrue(
    #         np.allclose(Vb, expected_V, rtol=1e-2),
    #     )

    # def test_resisting_moments_srss(self) -> None:
    #     # do kaushik!
    #     cdmx = ReinforcedConcreteFrame.from_file(self.file)
    #     self.assertTrue(
    #         # np.allclose(design.fem.extras["forces"], expected_forces, rtol=0.15)
    #         False
    #     )

    # def test_design(self) -> None:
    #     """it should compute Vy, uy, ductility."""
    #     # Vy approxeq total_mass * Sa(T1, xi)
    #     # this would imply Vy/weight = Cs (seismic coeeff)
    #     # seismic_coeff realistic?
    #     # is yield drift realistic? 1%

    #     # check column moments to make sure they are My.
    #     # check column My/EI and compare against known empirical results
    #     self.assertTrue(False)

    # def test_timehistory(self) -> None:
    #     """it should produce realistic results"""
    #     self.assertTrue(False)
