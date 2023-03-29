from __future__ import annotations
from abc import ABC, abstractmethod
from app.assets import (
    AssetFactory,
    RiskAsset,
    RiskModelFactory,
    Asset,
)
from app.utils import (
    OPENSEES_ELEMENT_EDPs,
    OPENSEES_EDPs,
    OPENSEES_REACTION_EDPs,
)
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
from app.utils import GRAVITY, SummaryEDP, YamlMixin, ASSETS_PATH
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line, Pie
from app.concrete import DesignException, RectangularConcreteColumn


class ElementTypes(Enum):
    BEAM = "beam"
    COLUMN = "column"
    SPRING_BEAM = "spring_beam"
    SPRING_COLUMN = "spring_column"
    SLAB = 'slab'


class FEMValidationException(Exception):
    """Created model is invalid"""

    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload  # you could add more args

    def __str__(self):
        return str(
            self.message
        )  # __str__() obviously expects a string to be returned, so make sure not to send any other data types


@dataclass
class FE(ABC):
    id: str | None = None  # opensees tag

    @staticmethod
    def string_ids_for_list(fems: list["FE"]) -> str:
        return " ".join([str(fe.id) for fe in fems])


@dataclass
class Node(FE, YamlMixin):
    x: float | None = None
    y: float | None = None
    mass: float | None = None
    fixed: bool = False
    column: int | None = None  # 1 for first column
    bay: int | None = None  # 0 for first column
    floor: int | None = None  # 1 for y=0
    storey: int | None = None  # 0 for y=0
    free_dofs: int | None = 3
    zerolen: int | None = None  # wrt which id it is fixed to
    orientation: str | None = None
    base: bool | None = False  # where base shear is recorded

    def __post_init__(self):
        if self.mass is not None and self.mass <= 0:
            raise FEMValidationException("Mass must be positive! " + f"{str(self)}")
        if self.fixed:
            self.free_dofs = 0

    def __str__(self) -> str:
        s = f"node {self.id} {self.x} {self.y}"
        if self.mass:
            s += f" -mass {self.mass:.4f} 1e-9 1e-9"
        s += "\n"
        if self.fixed:
            s += f"fix {self.id} 1 1 1\n"

        if self.zerolen is not None:
            s += f"equalDOF {self.zerolen} {self.id} 1 2\n"

        return s


@dataclass
class DiaphragmNode(Node):
    """
    WARNING; big assumption: node 0 is always fixed,
    we will restrict rotations with EqualDOF
    vertical (y axis) movement is not restricted
    but can be is artificially restricted with high axial stiffness
    (if we restrict both y,r we will not be able to use the default eigen)
    (because the default solver only works for n-1 DOFS, returns mass normalized vectors)
    """

    def __post_init__(self):
        self.free_dofs = 2
        return super().__post_init__()

    def __str__(self) -> str:
        string = super().__str__()
        if not self.fixed:
            string += f"equalDOF 0 {self.id} 3\n"
        return string


@dataclass
class BeamColumn(FE, YamlMixin):
    """
    circular & axially rigid.
    natural coordinates bay:ix storey:iy
    """

    type: str = ElementTypes.BEAM.value
    model: str = "BeamColumn"
    i: int | None = None
    j: int | None = None
    radius: float | None = None
    Ix: float | None = None
    E: float = 1.0
    A: float = 100_000
    storey: int | None = None
    floor: int | None = None
    bay: str | None = None  # 0 for first column
    transf: int = 1
    length: float = 1.0
    category: str = "structural"
    CRACKED_INERTIA_FACTOR: float = 1.0

    def __post_init__(self):
        self.transf = 1 if self.type == ElementTypes.BEAM.value else 2
        if self.Ix and not self.radius:
            self.radius = (4 * self.Ix / np.pi) ** 0.25
        elif self.radius and not self.Ix:
            self.Ix = self.CRACKED_INERTIA_FACTOR * np.pi * self.radius**4 / 4
        elif not self.Ix and not self.radius:
            raise Exception("BeamColumn requires Ix or radius")

    def __str__(self) -> str:
        return f"element elasticBeamColumn {self.id} {self.i} {self.j} {self.A:.5g} {self.E:.6g} {self.Ix:.6g} {self.transf}\n"


@dataclass
class ElasticBeamColumn(RiskAsset, BeamColumn):
    """
    we include nonlinear properties so it can load nonlinear models as well
    so we can transition from elastic -> plastic
    """

    model: str = "ElasticBeamColumn"
    k: float | None = None
    My: float | None = None
    Vy: float | None = None
    alpha: str = "0.0055"
    EA: float = 1e9
    integration_points: int = 5
    _DOLLARS_PER_UNIT_VOLUME: float = 1.500
    _SLAB_PCT: float = 5.0
    Q: float = 1.0

    def __repr__(self) -> str:
        return 'ElasticBeamColumn 1'

    def __post_init__(self):
        if self.E <= 0 or self.Ix <= 0:
            raise FEMValidationException(
                "Properties must be positive! " + f"{str(self)}"
            )
        if self.name is None:
            if self.type == ElementTypes.COLUMN.value:
                if self.Q == 1:
                    self.name = "FragileConcreteColumnAsset"
                else:
                    self.name = "ConcreteColumnAsset"
            else:
                self.name = "ConcreteBeamAsset"
        self._risk = RiskModelFactory(self.name)
        self.k = self.get_and_set_k()
        BeamColumn.__post_init__(self)
        RiskAsset.__post_init__(self)
        if self.radius:
            self.get_and_set_net_worth()
        self.node = self.i

    def dollars(self, *, strana_results_df):
        print(f"Structural element {self.name=} {self.node=} {self.rugged=}")
        strana_results_df["collapse_losses"] = (
            strana_results_df["collapse"]
            .apply(lambda r: self.net_worth if r else 0)
            .values
        )
        if self.type == ElementTypes.COLUMN.value:
            strana_results_df = self.dollars_for_storey(
                strana_results_df=strana_results_df
            )
        elif self.type == ElementTypes.BEAM.value:
            self.node = self.j
            dollars_for_j = self.dollars_for_node(strana_results_df=strana_results_df)[
                "losses"
            ].values
            self.node = self.i
            dollars_for_i = self.dollars_for_node(strana_results_df=strana_results_df)[
                "losses"
            ].values
            df = pd.DataFrame(dict(i=dollars_for_i, j=dollars_for_j))
            df["peak"] = df.apply(max, axis=1)
            strana_results_df["losses"] = df.peak.values
        losses = strana_results_df[["collapse_losses", "losses"]].apply(max, axis=1)
        return losses

    @property
    def area(self) -> float:
        return 3.1416 * self.radius**2

    @property
    def volume(self) -> float:
        return self.area * self.length

    def get_and_set_net_worth(self) -> float:
        """
        this is where size, steel % matters
        regression: prices per m2 of slab as function of thickness
        price_m2 = 7623 * thickness
        """
        net_worth = (
            (1 + ((self.Q - 1) * 0.08)) * self._DOLLARS_PER_UNIT_VOLUME * self.volume
        )
        self.net_worth = (
            net_worth
            if self.type == ElementTypes.COLUMN.value
            else self._SLAB_PCT * net_worth
        )
        return self.net_worth

    def get_and_set_k(self) -> float:
        k = 12 * self.E * self.Ix / self.length**3
        self.k = k
        return self.k

    @staticmethod
    def from_adjacency(
        adjacency: list[tuple[int, int, dict]]
    ) -> list["ElasticBeamColumn"]:
        return [
            ElasticBeamColumn(
                id=props["id"],
                i=i,
                j=j,
                E=props["E"],
                Ix=props["Ix"],
                length=props["length"],
                type=props["element_type"],
                storey=props["storey"],
                bay=props["bay"],
                floor=props.get("floor"),
                # My=props.get("My"),
                # Vy=props.get("Vy"),
            )
            for i, j, props in adjacency
        ]

@dataclass
class ConcreteElasticSlab(YamlMixin, RiskAsset):
    id: int | None = None
    type: str = ElementTypes.SLAB.value
    name: str = "ConcreteSlabAsset"
    model: str = "ConcreteElasticSlab"
    length: float | None = None
    thickness: float = 0

    def __str__(self) -> str:
        # must return empty for OpenSees
        return ''

    def __post_init__(self):
        if self.thickness < 0.08:
            self.thickness = 0.08
        if self.thickness > 0.30:
            self.thickness = 0.30
        area = self.length**2
        self.net_worth = area * (40 + 3 * 100*self.thickness)  # lstsq regression on median prices
        # self.net_worth = self.net_worth / 1000 # in 1k usd
        self.type = ElementTypes.SLAB.value
        return super().__post_init__()

@dataclass
class BilinBeamColumn(ElasticBeamColumn):
    model: str = "BilinBeamColumn"
    Vy: float | None = None

    def __repr__(self) -> str:
        return 'BilinBeamColumn My={self.My:.0f}'

    def __post_init__(self):
        self.model = self.__class__.__name__
        return super().__post_init__()

    @property
    def EI(self) -> float:
        return self.E * self.Ix

    def get_Vy(self) -> float:
        return 2 * self.My / self.length

    def __str__(self) -> str:
        s = ""
        s += f"uniaxialMaterial Elastic %(elastic{self.id})d {self.EA}\n"
        s += f"uniaxialMaterial Steel01 %(plastic{self.id})d {self.My} {self.EI} {self.alpha}\n"
        s += f"section Aggregator %(agg{self.id})d %(elastic{self.id})d P %(plastic{self.id})d Mz\n"
        s += f"element forceBeamColumn {self.id} {self.i} {self.j} {self.integration_points} %(agg{self.id})d {self.transf}\n"
        return s


@dataclass
class IMKSpring(RectangularConcreteColumn, ElasticBeamColumn):
    recorder_ix: int = 0
    model: str = "IMKSpring"
    ASPECT_RATIO_B_TO_H: float = 0.4  # b = kappa * h, h = (12 Ix / kappa )**(1/4)
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    Ks: float | None = None
    Ke: float | None = None
    Kb: float | None = None
    Ic: float | None = None
    secColTag: int | None = None
    imkMatTag: int | None = None
    elasticMatTag: int | None = None
    Vy: float | None = None

    def __repr__(self) -> str:
        return f'IMKSpring My={self.My:.0f}'

    def __str__(self) -> str:
        s = f"uniaxialMaterial Elastic {self.elasticMatTag} 1e9\n"
        s += f"uniaxialMaterial ModIMKPeakOriented {self.imkMatTag} {self.Ks:.2f} {self.alpha_postyield} {self.alpha_postyield} {self.My:.2f} {-self.My:.2f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} 1. 1. 1. 1. {self.theta_cap_cyclic:.8f} {self.theta_cap_cyclic:.8f} {self.theta_pc_cyclic:.8f} {self.theta_pc_cyclic:.8f} 1e-6 1e-6 {self.theta_u_cyclic:.8f} {self.theta_u_cyclic:.8f} 1. 1.\n"
        s += f"section Aggregator {self.secColTag} {self.elasticMatTag} P {self.imkMatTag} Mz\n"
        s += f"element zeroLengthSection  {self.id}  {self.i} {self.j} {self.secColTag}\n"
        return s

    @classmethod
    def from_bilin(cls, **data):
        Ix = data["Ix"]
        h = (12 * Ix / cls.ASPECT_RATIO_B_TO_H) ** 0.25
        b = h * cls.ASPECT_RATIO_B_TO_H
        print(f'{b=:.2f}, {h=:.2f} {Ix=:.3f}, {data["My"]}')
        designed = False
        maxIter = 5
        i = 0
        while not designed:
            if i >= maxIter:
                raise DesignException("Could not design even while making bigger")
            try:
                instance = cls(**{"h": h, "b": b, "EFFECTIVE_INERTIA_COEFF": 1, **data})
                designed = True
            except DesignException as e:
                print(e)
                h = h * 1.1
                b = b * 1.1
                i += 1
        return instance

    def __post_init__(self):
        from app.strana import EDP

        super().__post_init__()
        self.model = self.__class__.__name__
        self.radius = self.h / 2
        self.Ks = self.My / self.theta_y
        self.Ke = 6 * self.E * self.Ix / self.length
        self.Kb = self.Ks * self.Ke / (self.Ks - self.Ke)
        self.Ic = self.Kb * self.length / 6 / self.E
        self.secColTag = self.id + 100000
        self.imkMatTag = self.id + 200000
        self.elasticMatTag = self.id + 300000
        self.net_worth = self.net_worth if self.net_worth is not None else self.cost
        self._risk.edp = EDP.spring_moment_rotation_th.value
        self._risk.losses = self.losses

    def losses(self, xs: list[pd.DataFrame]) -> list[float]:
        costs = [self.park_ang_kunnath(df) for df in xs]
        print(costs)
        return costs

    def dollars(self, *, strana_results_df):
        print(f"Structural element {self.name=} {self.node=} {self.rugged=}")
        strana_results_df["collapse_losses"] = (
            strana_results_df["collapse"]
            .apply(lambda r: self.net_worth if r else 0)
            .values
        )
        # treat columns differently regarding shear collapse
        # if self.type == ElementTypes.COLUMN.value:
        #     strana_results_df = self.dollars_for_storey(
        #         strana_results_df=strana_results_df
        #     )
        # elif self.type == ElementTypes.BEAM.value:
        dollars = self.dollars_for_node(
            strana_results_df=strana_results_df,
            ix=self.recorder_ix,
            ele_type=self.type,
        )["losses"].values
        strana_results_df["losses"] = dollars
        losses = strana_results_df[["collapse_losses", "losses"]].apply(max, axis=1)
        return losses
