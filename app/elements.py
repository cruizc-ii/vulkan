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
    INFLATION,
    CollapseTypes,
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
    SLAB = "slab"


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
    category: str | None = "structural"
    CRACKED_INERTIA_FACTOR: float = 1.0

    def __repr__(self) -> str:
        return f"BeamColumn Ix={self.Ix:.3f}"

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
        return f"ElasticBeamColumn {self.length:.1f}m"

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

    def dollars(self, *, strana_results_df, views_by_path: dict, **kwargs):
        print(f"Structural element {self.name=} {self.node=} {self.rugged=}")
        # strana_results_df["collapse_losses"] = (
        #     strana_results_df["collapse"]
        #     .apply(lambda r: self.net_worth if r != CollapseTypes.NONE.value else 0)
        #     .values
        # )
        if self.type == ElementTypes.COLUMN.value:
            strana_results_df = self.dollars_for_storey(
                strana_results_df=strana_results_df,
                views_by_path=views_by_path,
                **kwargs,
            )
        elif self.type == ElementTypes.BEAM.value:
            self.node = self.j
            dollars_for_j = self.dollars_for_node(
                strana_results_df=strana_results_df,
                views_by_path=views_by_path,
                **kwargs,
            )
            self.node = self.i
            dollars_for_i = self.dollars_for_node(
                strana_results_df=strana_results_df,
                views_by_path=views_by_path,
                **kwargs,
            )
            df = pd.DataFrame(dict(i=dollars_for_i, j=dollars_for_j))
            df["peak"] = df.apply(max, axis=1)
            strana_results_df = df.peak.values
        # losses = strana_results_df[["collapse_losses", "losses"]].apply(max, axis=1)
        # strana_results_df.drop("collapse_losses", axis=1, inplace=True)
        return strana_results_df

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

    @property
    def Kbc(self) -> float:
        return 6 * self.E * self.Ix / self.length

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
    area: float | None = None

    def __repr__(self) -> str:
        return f"ConcreteElasticSlab t={self.thickness:.2f}m"

    def __str__(self) -> str:
        # must return empty for OpenSees, as this is a virtual element
        return ""

    def __post_init__(self):
        self.type = ElementTypes.SLAB.value
        if self.net_worth is None:
            if self.thickness < 0.08:
                self.thickness = 0.08
            if self.thickness > 0.30:
                self.thickness = 0.30
            from app.criteria import CodeMassesPre

            area = CodeMassesPre.SLAB_AREA_PERCENTAGE * self.length**2
            self.area = area
            cost_per_unit_area = (
                20 + 2.54 * 100 * self.thickness
            )  # lstsq regression on median prices GUÍA DE REFERENCIA PARA FORMULAR EL CATÁLOGO DE CONCEPTOS DEL PRESUPUESTO BASE DE OBRA PÚBLICA. veracruz
            self.net_worth = INFLATION * cost_per_unit_area * area
            self.net_worth = self.net_worth / 1000  # in 1k usd
        return super().__post_init__()


@dataclass
class BilinBeamColumn(ElasticBeamColumn):
    model: str = "BilinBeamColumn"
    Vy: float | None = None

    def __repr__(self) -> str:
        return "BilinBeamColumn My={self.My:.0f}"

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
    name: str = "ConcreteBeamColumnIMK"
    ASPECT_RATIO_B_TO_H: float = 0.4  # b = kappa * h, h = (12 Ix / kappa )**(1/4)
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    Ks: float | None = None
    Ke: float | None = None
    Kb: float | None = None
    Ks_original: float | None = None
    alpha_postyield_member: float = 0.03  # why 3%? Haselton says 13%.
    residual_My_ratio: float | None = 0.05
    Mr: float | None = None
    n: int = 10
    alpha_postyield: float | None = None
    theta_y_member: float | None = None
    theta_pc_member: float | None = None
    alpha_pc: float | None = None
    alpha_pc_member: float | None = None
    Ic: float | None = None
    secColTag: int | None = None
    imkMatTag: int | None = None
    elasticMatTag: int | None = None
    Vy: float | None = None
    Ke_Ks_ratio: float | None = None
    x: float | None = None
    y: float | None = None
    theta_r_member: float | None = None
    theta_r: float | None = None
    theta_r_cyclic: float | None = None
    theta_u: float = 10.0
    theta_pc_cyclic: float | None = None
    theta_cap_cyclic: float | None = None
    theta_u_cyclic: float | None = None
    betaParkAng: float | None = None
    betaJiangCheng: float | None = None
    gammaParkAng: float | None = None
    gammaJiangCheng: float | None = None

    def __repr__(self) -> str:
        return f"IMKSpring My={self.My:.0f} kN-m"

    def __str__(self) -> str:
        # s = f"uniaxialMaterial Elastic {self.elasticMatTag} 1e9\n"
        s = ""
        s += f"uniaxialMaterial ModIMKPeakOriented {self.imkMatTag} {self.Ks:.2f} {self.alpha_postyield:.5f} {self.alpha_postyield:.5f} {self.My:.2f} {-self.My:.2f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} {self.gammaJiangCheng:.3f} 1. 1. 1. 1. {self.theta_cap_cyclic:.6f} {self.theta_cap_cyclic:.6f} {self.theta_pc_cyclic:.6f} {self.theta_pc_cyclic:.6f} {self.residual_My_ratio} {self.residual_My_ratio} {self.theta_u:.6f} {self.theta_u:.6f} 1. 1.\n"
        # s += f"section Aggregator {self.secColTag} {self.elasticMatTag} P {self.imkMatTag} Mz\n"
        # s += f"element zeroLengthSection  {self.id}  {self.i} {self.j} {self.secColTag}\n"
        s += f"element zeroLength {self.id} {self.i} {self.j} -mat {self.imkMatTag} -dir 6\n"
        return s

    @property
    def properties(self):
        return f"My {self.My:.2f} mu {self.ductility:.3f} θy {self.theta_y:.4f} θpl {self.theta_cap_cyclic:.4f} θpc {self.theta_pc_cyclic:.4f} θu {self.theta_u:.4f} Λ {1./self.betaJiangCheng:.1f} alpha {self.alpha_postyield:.4f}\n"

    @classmethod
    def from_bilin(cls, **data):
        Ix = data["Ix"]
        h = (12 * Ix / cls.ASPECT_RATIO_B_TO_H) ** 0.25
        b = h * cls.ASPECT_RATIO_B_TO_H
        designed = False
        maxIter = 5
        i = 0
        while not designed:
            if i >= maxIter:
                print(f'{b=:.2f}, {h=:.2f} {Ix=:.3f}, My={data["My"]:.1f}')
                raise DesignException(
                    f"Section is too small for moment {data['My']:.1f}"
                )
            try:
                instance = cls(**{"h": h, "b": b, "EFFECTIVE_INERTIA_COEFF": 1, **data})
                designed = True
            except DesignException as e:
                print(e)
                h = h * 1.1
                b = b * 1.1
                i += 1
        instance.net_worth = instance.compute_net_worth()
        return instance

    def __post_init__(self):
        from app.strana import EDP

        self.Ke = 6 * self.E * self.Ix / self.length
        self.model = self.__class__.__name__
        if self.type == ElementTypes.SPRING_COLUMN.value and self.Q == 4:
            # indirectly take into account recommendations for ductile design from BCs, I do not like this part, it does not feel right.
            # discutir esto en conclusiones de la tesis
            # es una inconsistencia del procedimiento de diseno
            self.s = min([0.1, self.b / 4, self.h / 4])
        # self.Kb = 1e9
        # self.Kb = self.Ks * self.Ke / (self.Ks - 2 * self.Ke)
        self.radius = self.h / 2
        self.secColTag = self.id + 100000
        self.imkMatTag = self.id + 200000
        self.elasticMatTag = self.id + 300000

        super().__post_init__()

        self._risk.edp = EDP.spring_moment_rotation_th.value
        self._risk.losses = self.losses
        self.Mr = self.residual_My_ratio * self.My
        self.theta_y_member = (
            self.theta_y if not self.theta_y_member else self.theta_y_member
        )
        self.Ks_original = self.My / self.theta_y_member
        self.Ke_Ks_ratio = self.Ke / self.Ks_original

        self.theta_cap_member = (
            0.1
            * (1 + 0.55 * self.alpha_slippage)
            * 0.16**self.nu
            * (0.02 + 0.40 * self.pw) ** 0.43
            * 0.54 ** (0.01 * self.fpcMPa)
        )
        self.ductility_member = (
            self.theta_y_member + self.theta_cap_member
        ) / self.theta_y_member

        self.theta_pc_member = 0.76 * 0.031**self.nu * (0.02 + 40 * self.pw) ** 1.02
        self.theta_pc_member = (
            self.theta_pc_member if self.theta_pc_member < 0.10 else 0.10
        )
        # cyclic capacity
        n0 = self.nu if self.nu else 0.2
        n0 = n0 if n0 >= 0.2 else 0.2
        pt = self.pt * 100 if self.pt > 0.0075 else 0.75
        pw = self.pw * 100
        self.betaJiangCheng = (
            0.818 ** (100 * self.alpha_confinement * self.pw * self.fyw / self.fpc)
            * (0.023 * self.L / self.h + 3.352 * self.nu**2.35)
            + 0.039
        )
        self.betaParkAng = 0.7**pw * (
            0.073 * self.L / self.d + 0.24 * n0 + 0.314 * pt - 0.447
        )
        self.gammaParkAng = 1.0 / self.betaParkAng
        self.gammaJiangCheng = 1.0 / self.betaJiangCheng
        #
        # BEGIN MODIFY SPRING PROPERTIES
        #
        self.Ks = (self.n + 2) * self.Ke
        self.theta_y = self.My / self.Ks
        # modify cap properties, assuming ductility is postyield
        self.ductility = (
            self.n * (self.ductility_member - 1) * (1 - self.alpha_postyield_member)
            + self.ductility_member
        )

        self.theta_cap = self.theta_y_member * self.ductility

        self.alpha_postyield = (self.alpha_postyield_member) / (
            self.n + 2 - self.n * self.alpha_postyield_member
        )
        self.theta_cap_cyclic = (
            0.7 * self.theta_cap_member
        )  # Haselton et al. Calibration paper
        pc_slope = self.alpha_postyield_member * self.Ks_original
        self.Mc = (
            self.My + pc_slope * self.theta_cap_cyclic
        )  # My + delta M, original theta_cap

        alpha_pc = (self.Mr - self.Mc) / self.theta_pc_member / self.Ks_original
        mu_pc = (self.theta_y_member + self.theta_pc_member) / self.theta_y_member
        mu_pc_spring = self.n * (mu_pc - 1) * (1 - alpha_pc) + mu_pc
        print(f"{mu_pc=:} {mu_pc_spring=:} {alpha_pc=:} {self.theta_pc_member=:}")
        self.theta_pc = self.theta_y_member * mu_pc_spring

        self.theta_pc_cyclic = 0.5 * self.theta_pc
        self.theta_r_member = (
            self.theta_y_member + self.theta_cap_member + self.theta_pc_member
        )
        self.theta_r = self.theta_y + self.theta_cap + self.theta_pc
        self.theta_r_cyclic = (
            self.theta_y + self.theta_cap_cyclic + self.theta_pc_cyclic
        )

        self.gammaJiangCheng = (self.n + 2) * self.gammaJiangCheng

        self.Et = self.gammaJiangCheng * self.My * self.theta_cap_cyclic

    def losses(self, xs: list[pd.DataFrame]):
        return [self.park_ang_kunnath_DS(df) for df in xs]

    def dollars(self, *, strana_results_df, views_by_path: dict, **kwargs):
        # print(f"IMKSpring.dollars {self.name=} {self.node=} {self.rugged=}")
        # TODO: treat columns differently regarding shear collapse
        # if self.type == ElementTypes.COLUMN.value:
        #     strana_results_df = self.dollars_for_storey(
        #         strana_results_df=strana_results_df
        #     )
        losses = self.dollars_for_node(
            strana_results_df=strana_results_df,
            views_by_path=views_by_path,
            ix=self.recorder_ix,
            ele_type=self.type,
            **kwargs,
        )
        strana_results_df["losses"] = losses
        return losses

    def moment_rotation_member_df(self):
        rots = [
            0,
            self.theta_y_member,
            self.theta_y_member + self.theta_cap_member,
            self.theta_r_member,
            0.2,
            0.2 + 1e-3,
        ]
        Ms = [
            0,
            self.My,
            self.Mc,
            self.residual_My_ratio * self.My,
            self.residual_My_ratio * self.My,
            0,
        ]
        df = pd.DataFrame(Ms, index=rots)
        return df

    def moment_rotation_member_figure(self):
        df = self.moment_rotation_member_df()
        fig = df.plot()
        fig.update_layout(
            xaxis_title="rot (rad)",
            yaxis_title="M (kNm)",
            title_text=f"spring M-rot backbone",
        )
        return fig

    def moment_rotation_df(self):
        rots = [
            0,
            self.theta_y,
            self.theta_y + self.theta_cap,
            self.theta_r,
            # 1,
            # 1 + 1e-3,
        ]
        Ms = [
            0,
            self.My,
            self.Mc,
            self.residual_My_ratio * self.My,
            # self.residual_My_ratio * self.My,
            # 0,
        ]
        df = pd.DataFrame(Ms, index=rots)
        return df

    def moment_rotation_figure(self):
        df = self.moment_rotation_df()
        fig = df.plot()
        fig.update_layout(
            xaxis_title="rot (rad)",
            yaxis_title="M (kNm)",
            title_text=f"modified spring M-rot backbone",
        )
        return fig
