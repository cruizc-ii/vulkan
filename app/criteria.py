from __future__ import annotations
from abc import ABC, abstractmethod
from app.hazard import Spectra
from app.utils import GRAVITY, chunk_arrays, regula_falsi
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from app.codes import BuildingCode, CDMXBuildingCode
from app.fem import (
    BilinFrame,
    FEMFactory,
    FiniteElementModel,
    ShearModel,
    PlainFEM,
    IMKFrame,
)
from pathlib import Path
from functools import partial


class DesignNotValidException(Exception):
    pass


@dataclass
class DesignCriterion(ABC):
    fem: FiniteElementModel
    specification: "BuildingSpecification"  # noqa: F821

    @abstractmethod
    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        """raise DesignNotValid error for flow control."""
        pass


class EulerShearPre(DesignCriterion):
    EULER_LAMBDA_BOTTOM = 15
    EULER_LAMBDA_TOP = 30

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        """
        RC columns have a slenderness ratio LAMBDA of about 30-70.
        Mexican code recommends <35 to not be considered slender.
        radius of gyration 'rx' of a circular column is R/sqrt(2) therefore a first approx of the radius
        and therefore of Ix = pi r^4 / 4, is:  R = 1.4142 H / lambda

        this class 'chunks' or groups inertias by sqrt(storeys) and goes from
        robust -> slender from the first floor to the last floor of the building.
        """

        mdof = ShearModel.from_spec(self.specification)
        ns = self.specification.num_storeys
        storeys = np.array(self.specification.storeys)
        euler_lambdas = np.linspace(self.EULER_LAMBDA_BOTTOM, self.EULER_LAMBDA_TOP, ns)
        Ixs, radii = [], []

        for lambda_i, columns, storey_height in zip(
            euler_lambdas, mdof.columns_by_storey, storeys
        ):
            radius = 1.414 * storey_height / lambda_i
            radii.append(radius)
            Ix = np.pi * radius**4 / 4
            Ixs.append(Ix)

        chunk_size = np.floor(np.sqrt(ns))
        new_radii = chunk_arrays(radii, chunk_size=chunk_size)
        new_Ixs = chunk_arrays(Ixs, chunk_size=chunk_size)
        for new_Ix, new_radius, columns in zip(
            new_Ixs, new_radii, mdof.columns_by_storey
        ):
            for col in columns:
                col.Ix = float(new_Ix)
                col.radius = float(new_radius)

        # results = mdof.get_and_set_eigen_results(results_path=results_path)
        # mdof.extras["moments"] = results.Mb.tolist()
        return mdof


class CodeMassesPre(DesignCriterion):
    """
    sets masses on spec and fem equal to 1t/m2 = 10kPa assuming area = bay.length **2
    therefore mass will be sigma*area/2 since this frame is taking half the board's mass
    """

    CODE_UNIFORM_LOADS_kPA = 10.0  # 1 t/m2
    SLAB_AREA_PERCENTAGE = 1  # part of the slab mass goes to this frame's beams

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = PlainFEM.from_spec(self.specification)
        uniform_load_kPa = kwargs.get("uniform_load_kPa") or self.CODE_UNIFORM_LOADS_kPA
        masses = np.array(
            [
                uniform_load_kPa
                * self.SLAB_AREA_PERCENTAGE
                * self.fem.length**2
                / GRAVITY
                for _ in range(self.fem.num_modes)
            ]
        )
        fem._update_masses_in_place(masses.tolist())
        fem.get_and_set_eigen_results(results_path=results_path)
        return fem


class LoeraPreArctan(DesignCriterion):
    def __init__(self, *args, **kwargs):
        ns = self.specification.num_storeys
        # stress_pct_empirical = (1 - np.exp(-0.05 * ns)) / (
        #     1 + np.exp(-0.05 * ns)
        # )  # smaller -> stiffer
        self.STRESS_PCT_EMPIRICAL = 2 / np.pi * np.arctan(0.05 * ns)
        super().__init__(*args, **kwargs)


class LoeraPre(DesignCriterion):
    """
    X-section column area such that working stress is a percentage of f'c under gravity conditions
    beam depth is a percentage of the column
    """

    WORKING_STRESS_PCT = 0.1  # measure of flexibility, higher -> slender
    BEAM_TO_COLUMN_RATIO = 0.8
    _COLUMN_CRACKED_INERTIAS = 1.0
    _BEAM_CRACKED_INERTIAS = 1.0
    STRESS_PCT_EMPIRICAL: float | None = None

    def __init__(self, *args, **kwargs):
        if not self.STRESS_PCT_EMPIRICAL:
            self.STRESS_PCT_EMPIRICAL = self.WORKING_STRESS_PCT
        super().__init__(*args, **kwargs)

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = PlainFEM.from_spec(self.specification)
        weights = np.array(self.fem.cumulative_weights)
        cols_st = fem.columns_by_storey
        beams_st = fem.beams_by_storey

        for W, columns, beams in zip(weights, cols_st, beams_st):
            num_cols = len(columns)
            weight_per_column = W / num_cols
            area_needed = weight_per_column / (
                self.STRESS_PCT_EMPIRICAL * self.specification.fc
            )
            column_radius = float(np.sqrt(area_needed / np.pi))
            beam_radius = column_radius * self.BEAM_TO_COLUMN_RATIO
            Ix = np.pi * column_radius**4 / 4
            for col in columns:
                col.Ix = float(Ix) * self._COLUMN_CRACKED_INERTIAS
                col.radius = column_radius

            for beam in beams:
                beam.radius = beam_radius
                beam.Ix = (np.pi * beam.radius**4 / 4) * self._BEAM_CRACKED_INERTIAS

        fem.get_and_set_eigen_results(results_path)
        return fem


class ShearStiffnessRetryPre(DesignCriterion):
    """
    does a more realistic grouping of sections

    16 storey building has only 4 groups of inertias
    9 storey building has only 3 groups of inertias

    attempts to attain the empirical period of buildings ns/8
    varying the stiffnesses of the model.
    """

    PERIOD_TOLERANCE_PCT: float = 0.2
    MAX_ITERATIONS = 20

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        target_period = self.specification.miranda_fundamental_period
        ns = self.specification.num_storeys
        chunk_size = np.floor(np.sqrt(ns))
        leftmost_column_Ixs = np.array([c[0].Ix for c in self.fem.columns_by_storey])
        columns_Ixs = chunk_arrays(leftmost_column_Ixs, chunk_size=chunk_size)

        mdof = ShearModel(**{**self.fem.to_dict, "_inertias": columns_Ixs.tolist()})

        def target_fn(factor: float) -> float:
            _inertias = factor * columns_Ixs
            mdof = ShearModel.from_spec(
                self.specification, _inertias=_inertias.tolist()
            )
            fx = target_period - mdof.period
            return fx

        period_difference_pct = (target_period - mdof.period) / target_period
        # use regula_falsi to get to the desired period, fix (a,b) as follows:
        # if pct>0 we are too stiff, need to make more flexible, [0.01, 1]
        # if pct<0 we are too flexible, need to make stiffer, increase inertias [1, 100]
        a = 0.01 if period_difference_pct > 0 else 1
        b = 1 if period_difference_pct > 0 else 100
        inertia_factor, period_difference = regula_falsi(
            target_fn,
            a,
            b,
            tol=self.PERIOD_TOLERANCE_PCT * target_period,
            iter=self.MAX_ITERATIONS,
        )  # if inertias are not close by 2 orders of magnitude, something is wrong with our predesign methods. either the mass is too big/ or the storeys are too small/big. we have to abort

        column_inertias = inertia_factor * columns_Ixs
        mdof = ShearModel(**{**self.fem.to_dict, "_inertias": column_inertias.tolist()})
        return mdof


class ForceBasedPre(DesignCriterion):
    """
    will take in any spec and return a 'realistic' pre-design both
    in stiffnesses (element dimensions) and masses (weights)
    to be used before CDMXDesignQ or any other Code-based force design.
    """

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = self.fem
        for index, _class in enumerate(
            [
                CodeMassesPre,
                EulerShearPre,
                ShearStiffnessRetryPre,
            ]
        ):
            instance: DesignCriterion = _class(
                specification=self.specification, fem=fem
            )
            filepath = results_path / _class.__name__
            fem = instance.run(results_path=filepath, *args, **kwargs)
        return fem


class ShearRSA(DesignCriterion):
    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = ShearModel(**self.fem.to_dict)
        results = fem.get_and_set_eigen_results(results_path)
        S = results.S
        spectra: Spectra = self.specification._design_spectra[self.__class__.__name__]
        As = np.array(spectra.get_ordinates_for_periods(fem.periods))
        forces = S.dot(np.eye(len(As)) * As)
        peak_forces = np.sqrt(np.sum(forces**2, axis=1))
        fem.extras["S"] = S
        fem.extras["forces"] = forces.tolist()
        fem.extras["design_forces"] = peak_forces.tolist()
        return fem


@dataclass
class CDMX2017Q1(DesignCriterion):
    Q: int = 1

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        from app.strana import RSA

        fem = PlainFEM.from_spec(self.specification)
        criterion = ForceBasedPre(specification=self.specification, fem=fem)
        designed_fem = criterion.run(results_path=results_path, *args, **kwargs)

        data = designed_fem.to_dict
        fem = ShearModel(**data)
        code = CDMXBuildingCode(Q=self.Q)
        strana = RSA(results_path=results_path, fem=fem, code=code)
        design_moments, peak_shears, cs = strana.srss()

        fem = BilinFrame.from_elastic(
            fem=designed_fem, design_moments=design_moments, Q=self.Q
        )
        fem.extras["column_design_moments"] = design_moments
        fem.extras["design_shears"] = peak_shears
        fem.extras["c_design"] = cs
        return fem


@dataclass
class CDMX2017Q4(CDMX2017Q1):
    Q: int = 4


@dataclass
class CDMX2017Q1IMK(CDMX2017Q1):
    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = super().run(results_path=results_path, *args, **kwargs)
        fem = IMKFrame(**fem.to_dict)
        return fem


@dataclass
class CDMX2017Q4IMK(CDMX2017Q4):
    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = super().run(results_path=results_path, *args, **kwargs)
        fem = IMKFrame(**fem.to_dict)
        return fem


class DesignCriterionFactory:
    seeds = {
        EulerShearPre.__name__: EulerShearPre,
        LoeraPre.__name__: LoeraPre,
        ForceBasedPre.__name__: ForceBasedPre,
        CodeMassesPre.__name__: CodeMassesPre,
        ShearStiffnessRetryPre.__name__: ShearStiffnessRetryPre,
        ShearRSA.__name__: ShearRSA,
        CDMX2017Q1.__name__: CDMX2017Q1,
        CDMX2017Q4.__name__: CDMX2017Q4,
        CDMX2017Q1IMK.__name__: CDMX2017Q1IMK,
        CDMX2017Q4IMK.__name__: CDMX2017Q4IMK,
    }

    public_seeds = {
        CDMX2017Q1.__name__: CDMX2017Q1,
        CDMX2017Q4.__name__: CDMX2017Q4,
        CDMX2017Q1IMK.__name__: CDMX2017Q1IMK,
        CDMX2017Q4IMK.__name__: CDMX2017Q4IMK,
    }

    DEFAULT: str = CDMX2017Q1.__name__

    def __new__(cls, name) -> DesignCriterion:
        return cls.seeds[name]

    @classmethod
    def add(cls, name, seed):
        cls.seeds[name] = seed

    @classmethod
    def options(cls) -> list:
        return list(cls.seeds.keys())

    @classmethod
    def public_options(cls) -> list:
        return list(cls.public_seeds.keys())
