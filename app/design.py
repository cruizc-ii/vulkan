from __future__ import annotations
import numpy as np
from .utils import NamedYamlMixin, DesignException
from dataclasses import dataclass, field
from abc import ABC
from app.utils import GRAVITY, DESIGN_DIR, METERS_TO_FEET
from app.criteria import DesignCriterionFactory
from app.fem import FiniteElementModel, PlainFEM, FEMFactory
from pathlib import Path
from app.hazard import Spectra
from app.criteria import DesignCriterion


@dataclass
class BuildingSpecification(ABC, NamedYamlMixin):
    """
    2d rectangular frame
    - no set-backs & no holes
    - fixed base columns, uniform loads by storey.
    - masses per storey lumped at leftmost node
    """

    name: str
    storeys: list[float] = field(default_factory=lambda: [4.0])
    bays: list[float] = field(default_factory=lambda: [8.0])
    masses: list[float] = field(default_factory=list)
    damping: float = 0.05
    num_frames: int = 1
    design_criteria: list[str] = field(
        default_factory=DesignCriterionFactory.public_options
    )
    design_spectra: dict[dict[str, str], str] = field(default_factory=dict)

    _design_criteria: list["DesignCriterion"] | None = None
    _design_spectra: list["Spectra"] | None = None
    _adjacency: list[tuple[float, int, dict[str, object]]] | None = None

    _DEFAULT_RESULTS_PATH: Path = DESIGN_DIR
    columns: list[float] | None = None
    floors: list[float] | None = None
    height: float | None = None
    width: float | None = None
    chopra_fundamental_period_plus1sigma: float | None = None
    miranda_fundamental_period: float | None = None

    num_storeys: float | None = None
    num_floors: float | None = None
    num_bays: float | None = None
    num_cols: float | None = None

    occupancy: str | None = None
    fems: list[FiniteElementModel] = field(default_factory=list)

    Icols: float = 0.001
    Ibeams: float = 0.001
    Ecols: float = 30e6
    Ebeams: float = 30e6

    def __post_init__(self):
        self.__pre_init__()
        if all([isinstance(f, dict) for f in self.fems]):
            # print([f["model"] for f in self.fems])
            self.fems = [FEMFactory(**data) for data in self.fems]

    def __pre_init__(self) -> None:
        self._design_criteria = [
            DesignCriterionFactory(c) for c in self.design_criteria
        ]
        self.num_storeys = len(self.storeys)
        self.num_floors = self.num_storeys + 1
        self.num_bays = len(self.bays)
        self.num_cols = self.num_bays + 1
        self.floors = [0.0] + self.storeys
        self.columns = [0.0] + self.bays
        self.height = sum(self.storeys)
        self.width = sum(self.bays)
        self.chopra_fundamental_period_plus1sigma = (
            0.023 * (self.height * METERS_TO_FEET) ** 0.9
        )
        self.miranda_fundamental_period = self.num_storeys / 8

        if self.occupancy is None:
            from app.occupancy import BuildingOccupancy

            self.occupancy = BuildingOccupancy.DEFAULT

        if len(self.masses) == 1:
            self.masses = [self.masses[0] for _ in range(self.num_storeys)]
        elif len(self.masses) == 0:
            self.masses = [1 * self.width**2 for _ in range(self.num_storeys)]

        self._design_spectra = {
            criteria: Spectra(**data) for criteria, data in self.design_spectra.items()
        }
        self.__set_up__()

    def __set_up__(self):
        columns, colIDs = {}, {}
        beams, beamIDs = {}, {}
        nodes, fixed_nodes = {}, []
        _adjacency = []
        masses_by_storey = {}
        eID, nodeID = -1, -1
        y = 0
        for iy, st in enumerate(self.floors):
            y += st
            x = 0
            for ix, bay in enumerate(self.columns):
                x += bay
                nodeID += 1
                nodes[nodeID] = dict(
                    x=x, y=y, mass=None, floor=iy + 1, storey=iy, bay=ix, column=ix + 1
                )
                if iy > 0:
                    # for floors above ground level, create the columns
                    eID += 1
                    col_coords = (
                        ix,
                        iy,
                    )  # given in a natural coordinate system (col, floor)
                    i, j = nodeID - self.num_cols, nodeID
                    col = dict(
                        id=eID,
                        i=i,
                        j=j,
                        coords=col_coords,
                        # element_type=ElementTypes.COLUMN.value,
                        element_type="column",
                        storey=iy,
                        floor=iy,
                        bay=ix,
                        column=ix + 1,
                        E=self.Ecols,
                        Ix=self.Ibeams,
                        length=st,
                    )
                    _adjacency.append((i, j, col))
                    columns[col_coords] = col
                    colIDs[eID] = col
                    if ix > 0:
                        # if we are not in the first column, create the beam.
                        eID += 1
                        beam_coords = (ix, iy)
                        i = nodeID - 1
                        j = nodeID
                        beam = dict(
                            id=eID,
                            i=i,
                            j=j,
                            coords=beam_coords,
                            # element_type=ElementTypes.BEAM.value,
                            element_type="beam",
                            storey=iy,
                            floor=iy + 1,
                            bay=ix,
                            E=self.Ebeams,
                            Ix=self.Ibeams,
                            length=bay,
                        )
                        _adjacency.append((i, j, beam))
                        beams[beam_coords] = beam
                        beamIDs[eID] = beam
                    else:
                        # for the first column, add the storey mass to the nodeID id.
                        mass = self.masses[iy - 1]
                        masses_by_storey[nodeID] = mass
                        nodes[nodeID]["mass"] = mass

                else:
                    # for ground floor, add the fixed nodeID coordinates.
                    nodes[nodeID]["fixed"] = True
                    fixed_nodes.append(nodeID)

        self._adjacency = _adjacency
        self.nodes = nodes
        self.fixed_nodes = fixed_nodes

    def _update_masses_in_place(
        self, new_masses: list[float] | np.ndarray[float]
    ) -> None:
        self.masses = new_masses
        return

    @property
    def summary(self) -> dict:
        return {
            "design name": self.name,
            "damping": self.damping,
            "storeys": self.num_storeys,
            "weight": self.weight_str,
            "bays": self.num_bays,
            "occupancy": self.occupancy,
            "num frames": self.num_frames,
            "criteria": self.design_criteria[-1]
            if len(self.design_criteria) > 0
            else None,
        }

    @property
    def fem(self):
        return self.fems[-1]

    @property
    def total_mass(self) -> float:
        return sum(self.masses)

    @property
    def weight_str(self) -> str:
        return f"{self.total_mass * GRAVITY:.1f}"

    def force_design(
        self,
        results_path: Path,
        seed_class: FiniteElementModel | None = None,
        *args,
        **kwargs
    ) -> list[FiniteElementModel]:
        fem = None
        if seed_class is not None:
            fem = seed_class.from_spec(self)
        elif len(self.fems) > 0:
            fem = self.fem
        self.fems = self.design(
            results_path=results_path, criteria=self._design_criteria, fem=fem,
            *args, **kwargs
        )
        return self.fems

    def design(
        self,
        results_path: Path,
        criteria: list[DesignCriterion],
        fem: FiniteElementModel | None = None,
        *args,
        **kwargs,
    ) -> list[FiniteElementModel]:
        if results_path is None:
            results_path = self._DEFAULT_RESULTS_PATH / self.name
        else:
            results_path = results_path / self.name

        if fem is None and len(self.fems) > 0:
            fem = self.fem

        fems = []
        for _class in criteria:
            instance: DesignCriterion = _class(specification=self, fem=fem)
            instance_path = results_path / instance.__class__.__name__
            fem = instance.run(results_path=instance_path, *args, **kwargs)
            fems.append(fem)

        return fems


@dataclass
class ReinforcedConcreteFrame(BuildingSpecification):
    fc: float = 30e3
    Ec: float = 30e6
    fy: float = 420e3
    Es: float = 200e6

    def __post_init__(self):
        # WARNING: this is an inconsistency with out design procedures
        # each BC defines its own Ec, but this property is independent of legal matters!
        self.Ec = 4.4e6 * (self.fc / 1e3) ** 0.5
        self.Ecols = self.Ec
        self.Ebeams = self.Ec
        return super().__post_init__()
