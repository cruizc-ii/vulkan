from __future__ import annotations
from abc import ABC, abstractmethod
from app.hazard import Spectra
from app.utils import GRAVITY
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

METERS_TO_FEET = 1 / 0.3048


def put_mass(nodes_with_mass, masses):
    for node, mass in zip(nodes_with_mass, masses):
        node.mass = mass


def put_mass_spec(nodes, masses):
    nodes_with_mass = [n for n in nodes if n["mass"] is not None]
    for node, mass in zip(nodes_with_mass, masses):
        node["mass"] = mass


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
    EULER_LAMBDA = 25

    def run(self, results_path: Path, *args, **kwargs) -> ShearModel:
        """
        RC columns have a slenderness ratio LAMBDA of about 30-70.
        Mexican code recommends <35 to not be considered slender.
        radius of gyration 'rx' of a circular column is R/sqrt(2) therefore a first approx of the radius
        and therefore of Ix = pi r^4 / 4, is:  R = 1.4142 H / lambda
        """
        mdof = ShearModel.from_spec(self.specification)
        storeys = np.array(self.specification.storeys)
        radii = 1.414 * storeys / self.EULER_LAMBDA
        Ixs = (np.pi * radii**4) / 4
        cols_st = mdof.columns_by_storey
        for Ix, columns, radius in zip(Ixs, cols_st, radii):
            for col in columns:
                col.Ix = float(Ix)
                col.radius = float(radius)

        results = mdof.get_and_set_eigen_results(results_path=results_path)
        mdof.extras["moments"] = results.Mb.tolist()
        return mdof


class CodeMassesPre(DesignCriterion):
    """
    sets masses on spec and fem equal to 1t/m2 = 10kPa assuming area = bay.length **2
    therefore mass will be sigma*area/2 since this frame is taking half the board's mass
    """

    CODE_UNIFORM_LOADS_kPA = 10.0  # 1 t/m2
    SLAB_AREA_PERCENTAGE = 0.5  # part of the slab mass goes to this frame's beams

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = PlainFEM.from_spec(self.specification)
        masses = np.array(
            [
                self.CODE_UNIFORM_LOADS_kPA
                * self.SLAB_AREA_PERCENTAGE
                * self.fem.length**2
                / GRAVITY
                for _ in range(self.fem.num_modes)
            ]
        )
        put_mass(fem.mass_nodes, masses.tolist())
        put_mass_spec(self.specification.nodes.values(), masses.tolist())
        self.specification.masses = masses.tolist()
        self.specification.uniform_beam_loads_by_mass = [
            GRAVITY * mass / self.specification.width for mass in masses.tolist()
        ]
        fem.get_and_set_eigen_results(results_path=results_path)
        return fem


class LoeraPre(DesignCriterion):
    """
    X-section column area such that working stress is a percentage of f'c under gravity conditions
    beam depth is a percentage of the column
    """

    WORKING_STRESS_PCT = 0.05
    BEAM_TO_COLUMN_RATIO = 0.5
    _COLUMN_CRACKED_INERTIAS = 1.0
    _BEAM_CRACKED_INERTIAS = 1.0

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = PlainFEM.from_spec(self.specification)
        weights = np.array(self.fem.cumulative_weights)
        cols_st = fem.columns_by_storey
        beams_st = fem.beams_by_storey
        for W, columns, beams in zip(weights, cols_st, beams_st):
            num_cols = len(columns)
            weight_per_column = W / num_cols
            area_needed = weight_per_column / (
                self.WORKING_STRESS_PCT * self.specification.fc
            )
            radius = np.sqrt(area_needed / np.pi)
            Ix = (np.pi * radius**4) / 4
            for col in columns:
                col.Ix = float(Ix) * self._COLUMN_CRACKED_INERTIAS
                col.radius = float(radius)

            for beam in beams:
                beam.Ix = (
                    float(Ix / self.BEAM_TO_COLUMN_RATIO**4)
                    * self._BEAM_CRACKED_INERTIAS
                )
                beam.radius = float(radius * self.BEAM_TO_COLUMN_RATIO)

        fem.get_and_set_eigen_results(results_path)
        return fem


class ChopraPeriodsPre(DesignCriterion):
    period_tolerance_percentage: float = 0.25
    MAX_ITERATIONS = 10
    building_code: Optional[BuildingCode] = None

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        """
        Chopra & Goel (2000) Building period formulas for estimating seismic displacements
        concrete T = 0.028 H**0.8 where H is in feet (this is almost the same as num_storeys/10 surprisingly!)

        this procedure MUTATES masses in fem and spec!
        to make the building to be somewhat close to a realistic period.
        For equal masses and storey stiffness the fundamental frequency is
        w1 = 0.285 (k/m)^0.5
        T1 = 2pi/w1
        we can estimate the storey mass to be therefore as
        m = k (0.285 T1)**2/(2pi)**2
        """
        fundamental_period = 0.016 * (self.specification.height * METERS_TO_FEET) ** 0.9
        data = self.fem.to_dict
        data.pop("model")
        fem: FiniteElementModel = FEMFactory(
            **data, model=ShearModel.__name__
        )  # make a copy
        storey_stiffnesses = fem.storey_stiffnesses
        masses = np.array(
            [k * (0.285 * fundamental_period / 6.2832) ** 2 for k in storey_stiffnesses]
        )
        iteration = 0
        err = 1

        while abs(err) > self.period_tolerance_percentage:
            put_mass(fem.mass_nodes, masses.tolist())
            model_period = fem.get_and_set_eigen_results(
                results_path=results_path
            ).periods[0]
            err = (fundamental_period - model_period) / fundamental_period
            iteration += 1
            if abs(err) <= self.period_tolerance_percentage:
                break
            masses = np.sqrt(np.exp(err)) * masses
            if iteration > self.MAX_ITERATIONS:
                raise DesignNotValidException(
                    f"Computed FEM period {model_period:.2f} is not close to expected period {fundamental_period:.2f}, maybe stiffnesses are unreal? "
                )
        put_mass_spec(self.specification.nodes.values(), masses.tolist())
        self.specification.masses = masses.tolist()
        return fem


class ForceBasedPre(DesignCriterion):
    """
    will take in any spec and return a realistic pre-design both
    in stiffnesses (element dimensions) and masses (weights)
    """

    def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
        fem = self.fem
        for index, _class in enumerate(
            [
                CodeMassesPre,
                LoeraPre,
            ]
        ):
            instance: DesignCriterion = _class(
                specification=self.specification, fem=fem
            )
            filepath = results_path / _class.__name__
            fem = instance.run(results_path=filepath, *args, **kwargs)
        return fem


# class ShearRSA(DesignCriterion):
#     def run(self, results_path: Path, *args, **kwargs) -> FiniteElementModel:
#         fem = ShearModel(**self.fem.to_dict)
#         results = fem.get_and_set_eigen_results(results_path)
#         S = results.S
#         spectra: Spectra = self.specification._design_spectra[self.__class__.__name__]
#         As = np.array(spectra.get_ordinates_for_periods(fem.periods))
#         forces = S.dot(np.eye(len(As)) * As)
#         peak_forces = np.sqrt(np.sum(forces**2, axis=1))
#         fem.extras["S"] = S
#         fem.extras["forces"] = forces.tolist()
#         fem.extras["design_forces"] = peak_forces.tolist()
#         return fem


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


class DesignCriterionFactory:
    seeds = {
        # EulerShearPre.__name__: EulerShearPre,
        # LoeraPre.__name__: LoeraPre,
        # ChopraPeriodsPre.__name__: ChopraPeriodsPre,
        ForceBasedPre.__name__: ForceBasedPre,
        # CodeMassesPre.__name__: CodeMassesPre,
        CDMX2017Q1.__name__: CDMX2017Q1,
        # CDMX2017Q4.__name__: CDMX2017Q4,
        # CDMX2017Q1IMK.__name__: CDMX2017Q1IMK,
    }

    default_seeds = {
        CDMX2017Q1.__name__: CDMX2017Q1,
        # CDMX2017Q1IMK.__name__: CDMX2017Q1IMK,
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
    def default_criteria(cls) -> list:
        return [name for name in cls.default_seeds.keys()]
