from __future__ import annotations
from abc import ABC, abstractmethod
import inspect

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
from app.concrete import RectangularConcreteColumn


class ElementTypes(Enum):
    BEAM = "beam"
    COLUMN = "column"
    SPRING_BEAM = "spring_beam"
    SPRING_COLUMN = "spring_column"


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
    type: str = ElementTypes.BEAM.value
    model: str = "BeamColumn"
    i: int | None = None
    j: int | None = None
    E: float = 1.0
    Ix: float = 1.0
    A: float = 100_000
    radius: float | None = None
    storey: int | None = None
    floor: int | None = None
    bay: str | None = None  # 0 for first column
    transf: int = 1
    length: float = 1.0
    category: str = "structural"

    def __post_init__(self):
        self.transf = 1 if self.type == ElementTypes.BEAM.value else 2

    def __str__(self) -> str:
        return f"element elasticBeamColumn {self.id} {self.i} {self.j} {self.A:.5g} {self.E:.6g} {self.Ix:.6g} {self.transf}\n"


@dataclass
class ElasticBeamColumn(RiskAsset, BeamColumn):
    """
    circular & axially rigid.
    natural coordinates bay:ix storey:iy

    we include nonlinear properties so it can load nonlinear models as well..
    this is bad design.?
    I can transition from elastic -> plastic
    """

    model: str = "ElasticBeamColumn"
    k: float | None = None
    My: float | None = None
    alpha: str = "0.0055"
    EA: float = 1e9
    integration_points: int = 5
    _DOLLARS_PER_UNIT_VOLUME: float = 1.500
    _SLAB_PCT: float = 5.0
    Q: float = 1.0

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
        self.k = self.get_k()
        super().__post_init__()
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

    def get_k(self) -> float:
        return 12 * self.E * self.Ix / self.length**3

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
class BilinBeamColumn(ElasticBeamColumn):
    model: str = "BilinBeamColumn"
    Vy: float | None = None

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
    ASPECT_RATIO_B_TO_H: float = 0.5  # b = kappa * h, h = (12 Ix / kappa )**(1/4)
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
        return cls(**{"h": h, "b": b, "EFFECTIVE_INERTIA_COEFF": 1, **data})

    def __post_init__(self):
        from app.strana import EDP

        super().__post_init__()
        self.model = self.__class__.__name__
        self.radius = self.h
        self.Ks = self.My / self.theta_y
        self.Ke = 6 * self.E * self.Ix / self.length
        self.Kb = self.Ks * self.Ke / (self.Ks - self.Ke)
        self.Ic = self.Kb * self.length / 6 / self.E
        self.secColTag = self.id + 100000
        self.imkMatTag = self.id + 200000
        self.elasticMatTag = self.id + 300000
        self.net_worth = self.cost
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


@dataclass
class FiniteElementModel(ABC, YamlMixin):
    nodes: list[Node]
    elements: list[BeamColumn]
    damping: float
    model: str = "ABC"
    num_frames: int = 1
    num_storeys: float | None = None
    num_floors: float | None = None
    num_bays: float | None = None
    num_cols: float | None = None
    occupancy: str | None = None
    periods: list = field(default_factory=list)
    frequencies: list = field(default_factory=list)
    omegas: list = field(default_factory=list)
    values: list = field(default_factory=list)
    vectors: list = field(default_factory=list)
    a0: float | None = None
    a1: float | None = None
    extras: dict = field(default_factory=dict)
    _transf_beam: int = 1
    _transf_col: int = 2
    nonstructural_elements: list[Asset] = field(default_factory=list)
    contents: list[Asset] = field(default_factory=list)
    pushover_abs_path: str | None = None
    _pushover_view = None

    def __str__(self) -> str:
        h = "#!/usr/local/bin/opensees\n"
        h += f"# {self.model}, storeys {self.num_storeys}, bays {self.num_bays}\n"
        h += "wipe\n"
        h += "model BasicBuilder -ndm 2 -ndf 3\n"
        t = f"geomTransf Linear {self._transf_beam}\n"
        t += f"geomTransf PDelta {self._transf_col}\n"
        return h + t + self.nodes_str + self.elements_str

    @property
    def rayleigh_damping_string(self) -> str:
        return f"rayleigh {self.a0} 0 {self.a1} 0\n"

    def __post_init__(self):
        from app.occupancy import BuildingOccupancy

        if self.occupancy is None:
            self.occupancy = BuildingOccupancy.DEFAULT
        if isinstance(self.nodes[0], dict):
            self.nodes = [Node(**data) for data in self.nodes]
        if isinstance(self.elements[0], dict):
            self.elements = [ElementFactory(**data) for data in self.elements]
        if len(self.nonstructural_elements) > 0 and isinstance(
            self.nonstructural_elements[0], dict
        ):
            self.nonstructural_elements = [
                AssetFactory(**data) for data in self.nonstructural_elements
            ]
        else:
            self.build_and_place_assets()
        if len(self.contents) > 0 and isinstance(self.contents[0], dict):
            self.contents = [AssetFactory(**data) for data in self.contents]
        else:
            self.build_and_place_assets()

        self.model = self.__class__.__name__

        if self.pushover_abs_path:
            from app.strana import StructuralResultView

            self._pushover_view = StructuralResultView(
                abs_folder=self.pushover_abs_path
            )

        self.validate()

    @classmethod
    def from_spec(cls, spec) -> "FiniteElementModel":
        nodes = [Node(id=id, **info) for id, info in spec.nodes.items()]
        elements = ElasticBeamColumn.from_adjacency(spec._adjacency)
        fem = cls(
            nodes=nodes,
            elements=elements,
            damping=spec.damping,
            occupancy=spec.occupancy,
            num_frames=spec.num_frames,
            num_cols=spec.num_cols,
            num_bays=spec.num_bays,
            num_floors=spec.num_floors,
            num_storeys=spec.num_storeys,
        )
        return fem

    @property
    def summary(self) -> dict:
        df, ndf = self.pushover_dfs
        Vy = df["sum"].max()
        ix = df["sum"].idxmax()
        cs = ndf["sum"].max()
        uy = df.loc[ix]["u"]
        drift_y = ndf.loc[ix]["u"]
        stats = {
            "net worth [$]": self.total_net_worth,
            "elements net worth [$]": self.elements_net_worth,
            "nonstructural net worth [$]": self.nonstructural_net_worth,
            "contents net worth [$]": self.contents_net_worth,
            "Vy [kN]": Vy,
            "uy [m]": uy,
            "cs [1]": cs,
            "drift_y [%]": 100 * drift_y,
            "c_design [1]": self.extras.get("c_design"),
            "period [s]": self.periods[0] if len(self.periods) > 0 else None,
            "_pushover_x": df["u"].to_list(),
            "_pushover_y": df["sum"].to_list(),
            "_norm_pushover_x": ndf["u"].to_list(),
            "_norm_pushover_y": ndf["sum"].to_list(),
        }
        return stats

    @property
    def elements_assets(self) -> list["Asset"]:
        eles = [ele for ele in self.elements if Asset in inspect.getmro(ele.__class__)]
        return eles

    @property
    def assets(self):
        return self.elements_assets + self.nonstructural_elements + self.contents

    # @abstractmethod
    # def build_and_place_slabs(self) -> list[Asset]:
    #     """needs to include the slabs based on the beams"""

    def build_and_place_assets(self) -> list[Asset]:
        from app.occupancy import BuildingOccupancy

        occupancy = BuildingOccupancy(fem=self, model_str=self.occupancy)
        nonstructural_and_contents = occupancy.build()
        self.nonstructural_elements = [
            a for a in nonstructural_and_contents if a.category == "nonstructural"
        ]  # this is pretty bad and arbitrary... as categories can be anythiing.. we shouldn't tie them
        # to attrs.. for now it's fine.
        self.contents = [
            a for a in nonstructural_and_contents if a.category == "contents"
        ]
        # self.build_and_place_slabs() # this doesn't work because there is no constructor
        for asset in self.assets:
            if asset.net_worth:
                asset.net_worth = self.num_frames * asset.net_worth
        return self.assets

    @property
    def total_net_worth(self) -> float:
        return (
            self.elements_net_worth
            + self.nonstructural_net_worth
            + self.contents_net_worth
        )

    @property
    def eigen_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            dict(
                periods=self.periods,
                freqs=self.frequencies,
                values=self.values,
                omegas=self.omegas,
                mass=self.masses,
                weights=self.weights,
                loads=self.uniform_beam_loads,
                loads_kPa=self.uniform_area_loads_kPa,
            ),
            index=range(1, len(self.periods) + 1),
        )

    @property
    def elements_net_worth(self) -> float:
        return sum([a.net_worth for a in self.elements_assets])

    @property
    def nonstructural_net_worth(self) -> float:
        return sum([a.net_worth for a in self.nonstructural_elements])

    @property
    def contents_net_worth(self) -> float:
        return sum([a.net_worth for a in self.contents])

    def structural_elements_breakdown(self) -> pd.DataFrame:
        records = [a.to_dict for a in self.elements]
        df = pd.DataFrame.from_records(records)
        return df

    def nonstructural_elements_breakdown(self) -> pd.DataFrame:
        records = [a.to_dict for a in self.nonstructural_elements]
        df = pd.DataFrame.from_records(records)
        return df

    def contents_breakdown(self) -> pd.DataFrame:
        records = [a.to_dict for a in self.contents]
        df = pd.DataFrame.from_records(records)
        return df

    @property
    def design_space_fig(self):
        fig = Figure()
        trace = Scattergl(x=[1.3], y=[4.2], line=dict(color="black"))
        fig.add_trace(trace)
        trace_text = Scattergl(x=[1.75], y=[1.5], mode="text", text=["RC SMRF"])
        fig.add_trace(trace_text)
        fig.add_shape(
            type="rect",
            x0=1,
            x1=2,
            y0=1,
            y1=3,
            line=dict(
                color="RoyalBlue",
            ),
            # fillcolor='LightSkyBlue',
            # opacity=0.7
        )
        fig.add_shape(
            type="rect",
            x0=1.5,
            x1=1.8,
            y0=1.5,
            y1=2.5,
            line=dict(
                color="limegreen",
            ),
            fillcolor="green",
            opacity=0.2,
        )
        fig.update_layout(
            xaxis_title="design drift %",
            yaxis_title="multiple of Vb-design",
            title="design space",
        )
        return fig

    @property
    def assets_summary(self) -> list[dict]:
        summaries = []
        elements_summary = {
            "category": "Structural elements",
            "total_net_worth": self.elements_net_worth,
        }
        summaries.append(elements_summary)
        elements_summary = {
            "category": "Non structural elements",
            "total_net_worth": self.nonstructural_net_worth,
        }
        summaries.append(elements_summary)
        elements_summary = {
            "category": "Contents",
            "total_net_worth": self.contents_net_worth,
        }
        summaries.append(elements_summary)
        return summaries

    @property
    def elements_str(self) -> str:
        return self.columns_str + self.beams_str

    def to_tcl(self, filepath: Path):
        with open(filepath, "w+") as f:
            f.write(self.model_str)

    def determine_collapse_from_results(self, results: dict) -> bool:
        """
        upper bound on collapse, if any column is in DS==Collapse
        """
        collapse = False
        for st, columns in enumerate(self.columns_by_storey):
            column = columns[0]
            x = results[SummaryEDP.peak_drifts.value][st]
            ds = column._risk.simulate_damage_state(x)
            if ds == column._risk.damage_states[-1]["name"]:
                collapse = True
                break
        return collapse

    def validate(self, *args, **kwargs):
        # TODO:
        # no dangling nodes, at least one element has that ID in i or j,
        # no dangling elements, all element i,j IDs must exist in nodeSet
        if len(set(self.nodeIDs)) != len(self.nodeIDs):
            raise FEMValidationException(
                "Node ids must be unique" + f"{sorted(self.nodeIDs)}"
            )
        if len(set(self.elementIDs)) != len(self.elementIDs):
            raise FEMValidationException(
                "Node ids must be unique" + f"{sorted(self.elementIDs)}"
            )
        if len(self.fixed_nodes) == 0:
            raise FEMValidationException(
                "At least one fixed node must exist!" + f"{self.nodes_str}"
            )

    def get_and_set_eigen_results(self, results_path: Path):
        from app.strana import StructuralAnalysis

        strana = StructuralAnalysis(results_path, fem=self)
        view = strana.modal()
        self.periods = view.periods
        self.frequencies = view.frequencies
        self.omegas = view.omegas
        self.values = view.values
        self.vectors = view.vectors
        self.a0, self.a1 = self._rayleigh()
        return view

    def _rayleigh(self) -> None:
        z = self.damping
        if self.num_modes == 1:
            wi = wj = self.omegas[0]
            a0, a1 = 2 * z * wi * wj / (wi + wj), z * 2 / (wi + wj)
        else:
            c = (0, 1) if self.num_modes > 1 else (0)
            omegas = np.array(self.omegas)
            wi, wj = omegas[np.array(c)]
            a0, a1 = 2 * z * wi * wj / (wi + wj), z * 2 / (wi + wj)
        return float(a0), float(a1)

    def gravity(self, results_path: Path):
        from app.strana import StructuralAnalysis

        strana = StructuralAnalysis(results_path, fem=self)
        view = strana.standalone_gravity()
        return view

    def static(self, results_path: Path, forces_per_storey: list[float]):
        from app.strana import StructuralAnalysis

        strana = StructuralAnalysis(results_path, fem=self)
        view = strana.static(forces_per_storey=forces_per_storey)
        return view

    def pushover(self, results_path: Path, drift: float = 0.05):
        from app.strana import StructuralAnalysis
        from app.utils import AnalysisTypes

        strana = StructuralAnalysis(results_path, fem=self)
        view = strana.pushover(drift=drift)
        self.pushover_abs_path = str(results_path / AnalysisTypes.PUSHOVER.value)
        self._pushover_view = view
        return view

    @property
    def pushover_figs(
        self,
        # results_path: Path,
        # drift: float = 0.05,
        # force=False,
    ):
        # if force or not self._pushover_view:
        #     self.pushover(results_path=results_path, drift=drift)

        df, ndf = self.pushover_dfs
        cols = df[df.columns.difference(["u"])].columns
        fig = df.plot(x="u", y=cols)
        fig.update_layout(
            xaxis_title="roof u (m)", yaxis_title="Vb (kN)", title_text=f"Pushover"
        )
        normalized_fig = ndf.plot(x="u", y=cols)
        normalized_fig.update_layout(
            xaxis_title="roof drift [1]",
            yaxis_title="cs",
            title_text=f"Normalized pushover",
        )
        return fig, normalized_fig

    @property
    def pushover_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # if not self._pushover_view:
        #     raise Exception("You must run a pushover first!")
        view = self._pushover_view
        Vb = view.base_shear()
        Vb = -Vb
        Vb["sum"] = Vb.sum(axis=1)
        roof_disp = view.roof_displacements()
        df = Vb.join(roof_disp)
        cs = Vb / self.weight
        roof_drifts = roof_disp / self.height
        ndf = cs.join(roof_drifts)
        return df, ndf

    @property
    def pushover_stats(self) -> dict:
        df, ndf = self.pushover_dfs
        Vy = df["sum"].max()
        ix = df["sum"].idxmax()
        cs = ndf["sum"].max()
        uy = df.loc[ix]["u"]
        drift_y = ndf.loc[ix]["u"]
        stats = {
            "Vy": f"{Vy:.1f} kN",
            "uy": f"{uy:.3f} [m]",
            "cs": f"{cs:.3f} [1]",
            "drift_y": f"{100*drift_y:.2f} %",
            "c_design": self.extras.get("c_design"),
        }
        return stats

    @property
    def uniform_area_loads_kPa(self) -> list[float]:
        area = self.length**2
        masses_per_storey = np.array(self.masses)
        loads_per_storey = GRAVITY * masses_per_storey / area
        return loads_per_storey

    @property
    def uniform_beam_loads(self) -> list[float]:
        masses_per_storey = np.array(self.masses)
        beam_loads = GRAVITY * masses_per_storey / self.length
        return beam_loads.tolist()

    @property
    def height(self) -> float:
        ys = [n.y for n in self.nodes]
        return max(ys)

    @property
    def length(self) -> float:
        xs = [n.x for n in self.nodes]
        return max(xs)

    @property
    def total_length(self) -> float:
        """not including roof since assets can't really be placed there"""
        ls = [beam.length for beam in self.beams]
        return sum(ls)

    def cytoscape(self) -> tuple[list[dict], list[dict]]:
        ASSET_Y_OFFSET = 25
        SCALE = 100
        fig = [
            {
                "data": {
                    "id": node.id,
                    "label": f"{node.id}",
                },
                "classes": "fixed" if node.fixed else "free",
                "position": {"x": SCALE * node.x, "y": -SCALE * node.y},
            }
            for node in self.nodes
        ]
        fig += [
            {
                "data": {
                    "source": e.i,
                    "target": e.j,
                    "type": e.type,
                    "tag": e.id,
                    "r": 2 * SCALE * e.radius,
                }
            }
            for e in self.elements
        ]
        fig += [
            {
                "data": {
                    "id": ix,
                    "label": content.name,
                    "url": ASSETS_PATH + content.icon if content.icon else "",
                },
                "classes": "contents" if content.icon else "",
                "position": {
                    # "x": (SCALE * (content.x + 2 * np.random.random()))
                    # if content.x is not None
                    # else SCALE * np.random.random() * self.length,
                    "x": SCALE * content.x if content.x is not None else 0,
                    "y": -SCALE * self.leftmost_nodes[content.floor - 1].y
                    - ASSET_Y_OFFSET,
                },
            }
            for ix, content in enumerate(self.contents, len(fig))
            if not content.hidden
        ]
        fig += [
            {
                "data": {
                    "id": iy,
                    "label": asset.name,
                    "url": ASSETS_PATH + asset.icon if asset.icon else "",
                },
                "classes": "nonstructural" if asset.icon else "",
                "position": {
                    "x": SCALE * asset.x if asset.x is not None else 0,
                    # else SCALE * np.random.random() * self.length,
                    "y": -SCALE * self.leftmost_nodes[asset.floor - 1].y
                    - ASSET_Y_OFFSET,
                },
            }
            for iy, asset in enumerate(self.nonstructural_elements, len(fig))
            if not asset.hidden
        ]

        style = [
            {"selector": "edge", "style": {"label": "data(tag)"}},
            {"selector": "edge", "style": {"width": "data(r)"}},  # width of elements
            {"selector": "node", "style": {"label": "data(label)"}},
            {
                "selector": ".fixed",
                "style": {"shape": "rectangle", "background-color": "#444242"},
            },
            {
                "selector": ".contents",
                "style": {
                    "background-image": "data(url)",
                    "background-fit": "cover cover",
                    "width": 50,
                    "height": 50,
                },
            },  # contents
            {
                "selector": ".nonstructural",
                # "style": {"shape": "triangle", "background-color": "#0056ff"},
                "style": {
                    "background-image": "data(url)",
                    "background-fit": "cover cover",
                    "width": 50,
                    "height": 50,
                },
            },
        ]
        return fig, style

    @property
    def assets_pie_fig(self) -> Figure:
        labels = ["structural", "contents", "nonstructural"]
        values = [
            self.elements_net_worth,
            self.contents_net_worth,
            self.nonstructural_net_worth,
        ]
        fig = Figure(data=[Pie(labels=labels, values=values)])
        return fig

    @property
    def model_str(self):
        return str(self)

    @property
    def nodes_str(self):
        return "".join([str(n) for n in self.nodes])

    @property
    def nodeIDs(self):
        return [n.id for n in self.nodes]

    @property
    def elementIDs(self):
        return [e.id for e in self.elements]

    @property
    def elementIDS_as_str(self):
        return " ".join([str(eid) for eid in self.elementIDs])

    @property
    def beams(self):
        return [e for e in self.elements if e.type == ElementTypes.BEAM.value]

    @property
    def beamIDs(self):
        return [b.id for b in self.elements if b.type == ElementTypes.BEAM.value]

    @property
    def beamIDs_str(self):
        return " ".join([str(c) for c in self.beamIDs])

    @property
    def columns(self):
        return [e for e in self.elements if e.type == ElementTypes.COLUMN.value]

    @property
    def columnIDs(self):
        return [c.id for c in self.elements if c.type == ElementTypes.COLUMN.value]

    @property
    def columnIDs_str(self):
        return " ".join([str(c) for c in self.columnIDs])

    @property
    def columns_by_storey(self) -> list[list[ElasticBeamColumn]]:
        st = []
        for iy in range(1, self.num_modes + 1):
            cols = [col for col in self.columns if col.storey == iy]
            st.append(cols)
        return st

    @property
    def columnIDs_by_storey(self) -> list[list]:
        st = []
        for iy in range(1, self.num_modes + 1):
            cols = [col.id for col in self.columns if col.storey == iy]
            st.append(cols)
        return st

    @property
    def beams_by_storey(self):
        st = []
        for iy in range(1, self.num_modes + 1):
            beams = [b for b in self.beams if b.storey == iy]
            st.append(beams)
        return st

    @property
    def beamIDs_by_storey(self):
        st = []
        for iy in range(1, self.num_modes + 1):
            beams = [b.id for b in self.beams if b.storey == iy]
            st.append(beams)
        return st

    @property
    def columns_str(self):
        return "".join([str(c) for c in self.columns])

    @property
    def beams_str(self):
        return "".join([str(b) for b in self.beams])

    @property
    def mass_nodes(self):
        return [n for n in self.nodes if n.mass]

    @property
    def leftmost_nodes(self):
        return self.fixed_nodes[:1] + self.mass_nodes

    @property
    def roof_mass_node(self):
        return [n for n in self.mass_nodes][-1]

    @property
    def fixed_nodes(self):
        return [n for n in self.nodes if n.fixed]

    @property
    def fixedIDs(self):
        return [n.id for n in self.fixed_nodes]

    @property
    def fixedIDs_str(self):
        return " ".join([str(n_id) for n_id in self.fixedIDs])

    @property
    def massIDs(self):
        return [n.id for n in self.mass_nodes]

    @property
    def massIDs_str(self):
        return " ".join([str(n_id) for n_id in self.massIDs])

    @property
    def masses(self):
        return [n.mass for n in self.mass_nodes]

    @property
    def roofID(self):
        return self.massIDs[-1]

    @property
    def weights(self) -> list[float]:
        return [n.mass * GRAVITY for n in self.mass_nodes]

    @property
    def weight(self) -> float:
        return sum(self.weights)

    @property
    def cumulative_weights(self) -> list[float]:
        ws = np.array(list(reversed(self.weights)))
        cum_ws = list(reversed(ws.cumsum().tolist()))
        return cum_ws

    @property
    def mass_dofs(self):
        """needed for static condensation
        WARNING: assumption.
        we can assume that we start at DOF 0 always since
        our mass is in the -x- direction, we need a better implementation...
        """
        return np.cumsum([0] + [n.free_dofs for n in self.mass_nodes[:-1]])

    @property
    def free_dofs(self):
        """all DOF indices for the model"""
        freeDOFs = sum([n.free_dofs for n in self.nodes])
        return list(range(freeDOFs))

    @property
    def fixed_dofs(self):
        """fixed DOF indices"""
        f = np.array(self.fixedIDs)
        ixs = np.stack([f, f + 1, f + 2])
        return ixs

    @property
    def num_modes(self) -> int:
        return len(self.mass_nodes)

    @property
    def column_inertias(self) -> list[float]:
        return [c.Ix for c in self.columns]

    @property
    def column_radii(self) -> list[float]:
        return [c.radius for c in self.columns]

    @property
    def beam_inertias(self) -> list[float]:
        return [c.Ix for c in self.beams]

    @property
    def beam_radii(self) -> list[float]:
        return [c.radius for c in self.beams]

    @property
    def node_recorders(self) -> str:
        """TODO; add -time to .envelope to see the pseudotime that the min/max happened in"""
        s = ""
        # nodeIDs = ""  # per OpenSees docs, it defaults to all nodes in model.
        for edp in OPENSEES_EDPs:
            s += f"recorder Node            -file $abspath/node-{edp}.csv -time -dof 1 2 3 {edp}\n"
            s += f"recorder Node            -file $abspath/mass-{edp}.csv -time -node {self.massIDs_str} -dof 1 {edp}\n"
            s += f"recorder NodeEnvelope    -file $abspath/node-{edp}-envelope.csv -dof 1 2 3 {edp}\n"
            s += f"recorder NodeEnvelope    -file $abspath/mass-{edp}-envelope.csv -node {self.massIDs_str} -dof 1 {edp}\n"
            # -time will prepend a column with the step that node_i achieved envelope at dof_j

        fixed_nodes = self.fixedIDs_str
        for reaction, dof in OPENSEES_REACTION_EDPs:
            s += f"recorder Node            -file $abspath/{reaction}.csv -time -node {fixed_nodes} -dof {dof} reaction\n"
            s += f"recorder NodeEnvelope    -file $abspath/{reaction}-envelope.csv -node {fixed_nodes} -dof {dof} reaction\n"

        s += f"recorder Node            -file $abspath/roof-displacements.csv -time -node {self.roofID} -dof 1 disp\n"
        s += f"recorder NodeEnvelope            -file $abspath/roof-displacements-env.csv -time -node {self.roofID} -dof 1 disp\n"
        s += f"recorder Node            -file $abspath/roof-accels.csv -time -node {self.roofID} -dof 1 accel\n"
        s += f"recorder NodeEnvelope            -file $abspath/roof-accels-env.csv -time -node {self.roofID} -dof 1 accel\n"

        storey_node_ids = self.mass_nodes
        iNodes = "0 " + Node.string_ids_for_list(
            storey_node_ids[:-1]
        )  # 1st and next to last storeys
        jNodes = Node.string_ids_for_list(storey_node_ids)
        s += f"recorder Drift           -file $abspath/drifts.csv -time -iNode {iNodes} -jNode {jNodes} -dof 1 -perpDirn 2\n"
        roofID = self.roofID
        s += f"recorder Drift           -file $abspath/roof-drift.csv -time -iNode {fixed_nodes[0]} -jNode {roofID} -dof 1 -perpDirn 2\n"

        return s

    @property
    def element_recorders(self) -> str:
        # elementIDs = ""  # per OpenSees docs, it defaults to all elements in model.
        # WARNING; when runnning without IDS
        # we get a segfault [1]    39319 segmentation fault  opensees model.tcl
        # IDK why. so bad :(.. I really need to program my own in python numba.
        s = ""
        for edp in OPENSEES_ELEMENT_EDPs:
            for ele_type, ids in [
                ("columns", self.columnIDs_str),
                ("beams", self.beamIDs_str),
            ]:
                s += f"recorder Element         -file $abspath/{ele_type}.csv -ele {ids} -time {edp} \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope.csv -ele {ids} {edp} \n"
        for edp in OPENSEES_ELEMENT_EDPs:
            for ele_type, elems_by_st in [
                ("columns", self.columnIDs_by_storey),
                ("beams", self.beamIDs_by_storey),
            ]:
                for st, elems in enumerate(elems_by_st, 1):
                    ids = " ".join([str(id) for id in elems])
                    s += f"recorder Element         -file $abspath/{ele_type}-{st}.csv -ele {ids} -time {edp} \n"
                    s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope.csv -ele {ids} {edp} \n"

        return s


@dataclass
class PlainFEM(FiniteElementModel):
    """the simplest implementation of the interface"""

    model: str = "PlainFEM"

    def build_and_place_slabs(self) -> list[Asset]:
        from app.assets import RiskModelFactory

        slabs = []
        for floor, beams in enumerate(self.beams_by_storey, 1):
            for beam in beams:
                slab = RiskModelFactory("ConcreteSlabAsset")
                slab.net_worth = 3000 * (2 * beam.length**2)
                #  * beam.radius doesn't work here.. fem is not designed yet!
                slab.floor = floor
                slabs.append(slab)
        self.elements = self.elements + slabs
        return slabs

    def validate(self, *args, **kwargs):
        return super().validate(*args, **kwargs)


@dataclass
class ShearModel(FiniteElementModel):
    """
    disregards beams completely.
    takes lateral stiffness as sum(k_columns)
    """

    _mass_matrix: np.ndarray = None

    def validate(self, *args, **kwargs):
        return super().validate(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.model = self.__class__.__name__
        self.nodes = [self.fixed_nodes[0]] + [
            DiaphragmNode(**n.to_dict) for n in self.mass_nodes
        ]
        columns_by_storey = self.columns_by_storey
        first_bay_columns = [c for c in self.columns if c.bay == 0]
        columns = []
        for cols, prev_col in zip(columns_by_storey, first_bay_columns):
            data = prev_col.to_dict
            Ixs = [c.Ix for c in cols]
            data["Ix"] = sum(Ixs)
            columns.append(ElasticBeamColumn(**data))
        self.num_dofs = len(columns)
        self.elements = columns
        self._mass_matrix = np.diag(self.masses)
        heights = [n.y for n in self.nodes if not n.fixed]
        self._height_matrix = np.diag(heights)
        self.validate()

    def build_and_place_assets(self) -> list[Asset]:
        """no room for assets"""
        return []

    @property
    def storey_stiffnesses(self) -> list:
        ks = []
        for cols in self.columns_by_storey:
            ks.append(sum([c.k for c in cols]))
        return ks

    @property
    def storey_inertias(self) -> list[float]:
        return self.column_inertias

    @property
    def storey_radii(self) -> list[float]:
        return self.column_radii

    def get_and_set_eigen_results(self, results_path: Path):
        view = super().get_and_set_eigen_results(results_path)
        Phi = np.array(view.vectors)
        values = view.values
        N = len(values)
        M = self._mass_matrix
        H = self._height_matrix
        # when vectors are mass-normalized then M_n = v.T @ M @ v = 1, the default solver returns mass normalized vecs.
        ones = np.ones(N)
        m1 = M @ ones
        gamma = Phi.T @ m1
        G = np.diag(gamma)
        # matrix of effective modal masses (s) column_n contains the effective masses for mode_n for each storey (row)
        S = Phi @ M @ G  # as if 'weighing' each col of Phi by the product m_j g_j
        effective_masses = G**2  # since M_n = 1
        gi = np.diag(1.0 / gamma)
        effective_heights = gi @ Phi.T @ H @ M @ ones
        Ones = np.tri(N).T
        # 1 1 1 ... 1
        # 0 1 1 ..  1
        # 0 0 1 ..  1
        # 0 0 0 ... 1
        # this will 'sum' each column bottom-up.
        # V = flip(cumsum(flip(S, 0), 0), 0) # another very bad way of doing it.
        V = Ones @ S
        # now to compute the moments at each storey
        # the stepped matrix looks like this
        # h1 h2      h3 ...  h_n
        # 0  h2-h1   h3-h1   h_n - h1
        # 0   0      h3-h2   h_n - h2..
        # ...
        #   0       0       h_n - h_{n-1}
        h = np.insert(
            np.diagonal(H), 0, 0
        )  # extend with 0s so we can take the diff without losing a row.
        dH = np.array([h - h[ix] for ix, _ in enumerate(h)])  # stepped matrix
        dH = np.triu(dH)[
            :-1, 1:
        ]  # horrible way of just getting the correct triangular-upper array.
        # there must be a better way.
        Mb = dH @ S  # moments of effective-mass

        # for displacement-based design, we need something to multiple D_n(t) against.
        # this is similar as S but without masses!
        # U = Phi @ G
        view.gamma = G
        view.s = S
        view.S = S
        view.inertial_forces = S
        view.effective_masses = effective_masses
        view.M = effective_masses
        view.effective_heights = effective_heights
        view.H = effective_heights
        view.V = V
        view.shears = V
        view.Mb = Mb
        view.overturning_moments = Mb.tolist()
        return view


@dataclass
class RigidBeamFEM(FiniteElementModel):
    def validate(self, *args, **kwargs):
        return super().validate(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.model = self.__class__.__name__
        for beam in self.beams:
            beam.Ix = "1e9"


@dataclass
class BilinFrame(FiniteElementModel):
    def __post_init__(self):
        super().__post_init__()
        self.model = self.__class__.__name__
        if isinstance(self.elements[0], dict):
            self.elements = [BilinBeamColumn(**data) for data in self.elements]
        else:
            self.elements = [BilinBeamColumn(**elem.to_dict) for elem in self.elements]

    @classmethod
    def from_elastic(
        cls,
        fem: FiniteElementModel,
        design_moments: list[float],
        beam_column_ratio: float = 0.75,
        Q: float = 1.0,
    ) -> "BilinFrame":
        """will take design moments and put them into beams and columns"""
        col_moments = np.array(design_moments)
        beam_moments = beam_column_ratio * col_moments
        for cm, cols, bm, beams in zip(
            col_moments, fem.columns_by_storey, beam_moments, fem.beams_by_storey
        ):
            for c in cols:
                c.My = cm
                c.Q = Q
            for b in beams:
                b.My = bm
                b.Q = Q

        instance = cls(**fem.to_dict)
        return instance

    @property
    def elements_str(self) -> str:
        """this is a really dumb limitation of opensees, every E in Domain
        must have a non-null ID.. so we have to do gymnastics here to fill in the sections"""

        import re

        string = "".join([str(e) for e in self.elements])
        expr = re.compile("(?<=forceBeamColumn )\d+")
        ids = expr.findall(string)
        aggs = {}
        elastics = {}
        plastics = {}
        ix = 1
        for id in ids:
            aggs[f"agg{id}"] = ix
            ix += 1
            elastics[f"elastic{id}"] = ix
            ix += 1
            plastics[f"plastic{id}"] = ix
            ix += 1
        matcher = aggs | elastics | plastics
        formatted = string % matcher
        return formatted

    @property
    def element_recorders(self) -> str:
        s = super().element_recorders
        for ele_type, ids in [
            ("columns", self.columnIDs_str),
            ("beams", self.beamIDs_str),
        ]:
            s += f"recorder Element         -file $abspath/{ele_type}-a.csv -ele {ids} -time section 1 force \n"
            s += f"recorder Element         -file $abspath/{ele_type}-b.csv -ele {ids} -time section 5 force \n"
            s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope-a.csv -ele {ids} section 1 force \n"
            s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope-b.csv -ele {ids} section 5 force \n"

        for ele_type, elems_by_st in [
            ("columns", self.columnIDs_by_storey),
            ("beams", self.beamIDs_by_storey),
        ]:
            for st, elems in enumerate(elems_by_st, 1):
                ids = " ".join([str(id) for id in elems])
                s += f"recorder Element         -file $abspath/{ele_type}-{st}-a.csv -ele {ids} -time section 1 force \n"
                s += f"recorder Element         -file $abspath/{ele_type}-{st}-b.csv -ele {ids} -time section 5 force \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope-a.csv -ele {ids} section 1 force \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope-b.csv -ele {ids} section 5 force \n"
        return s


@dataclass
class IMKFrame(FiniteElementModel):
    done: bool = False

    @property
    def rayleigh_damping_string(self) -> str:
        s = f"region 4 -eleRange {self.elementIDS_as_str} -rayleigh 0. 0. {self.a1} 0.\n"
        s += f"region 5 -node {self.massIDs_str} -rayleigh {self.a0} 0. 0. 0.\n"
        return s

    @property
    def fixedIDs(self):
        return [n.id for n in self.nodes if n.base]

    @property
    def nodes_str(self) -> str:
        s = super().nodes_str
        for mass_node in self.mass_nodes:
            st = mass_node.storey
            st_nodes = [
                n for n in self.nodes if n.storey == st and not n.zerolen and not n.mass
            ]
            for st_node in st_nodes:
                s += f"equalDOF {mass_node.id} {st_node.id} 1\n"
        return s

    @property
    def springs(self):
        return [
            e
            for e in self.elements
            if e.type == ElementTypes.SPRING_BEAM.value
            or e.type == ElementTypes.SPRING_COLUMN.value
        ]

    @property
    def springs_beams(self):
        return [e for e in self.elements if e.type == ElementTypes.SPRING_BEAM.value]

    @property
    def spring_beams_ids(self):
        return [c.id for c in self.springs_beams]

    @property
    def spring_beams_ids_str(self):
        return " ".join([str(i) for i in self.spring_beams_ids])

    @property
    def springs_columns(self):
        return [e for e in self.elements if e.type == ElementTypes.SPRING_COLUMN.value]

    @property
    def spring_columns_ids(self):
        return [c.id for c in self.springs_columns]

    @property
    def spring_columns_ids_str(self):
        return " ".join([str(i) for i in self.spring_columns_ids])

    @property
    def spring_ids_str(self):
        return self.spring_columns_ids_str + " " + self.spring_beams_ids_str

    @property
    def springs_str(self) -> str:
        return " ".join([str(c) for c in self.springs])

    @property
    def elements_str(self) -> str:
        s = super().elements_str
        s += self.springs_str
        return s

    def __post_init__(self):
        self.model = self.__class__.__name__
        super().__post_init__()
        if not self.done:
            nodes = []
            next_id = self.nodes[-1].id + 1
            # create nodes in counterclockwise manner
            nodes_by_id = {}
            for node in self.nodes:
                if node.fixed:
                    orientations = ["col_down"]
                elif node.storey == self.num_storeys:
                    if node.column == 1:
                        orientations = ["col_up", "beam_left"]
                    elif node.column == self.num_cols:
                        orientations = ["col_up", "beam_right"]
                    else:
                        orientations = ["col_up", "beam_left", "beam_right"]
                elif node.column == 1:
                    orientations = ["col_up", "beam_left", "col_down"]
                elif node.column == self.num_cols:
                    orientations = ["col_up", "col_down", "beam_right"]
                else:
                    orientations = ["col_up", "beam_left", "col_down", "beam_right"]
                nodes_by_id[node.id] = {}
                for o in orientations:
                    n = Node(
                        id=next_id,
                        x=node.x,
                        y=node.y,
                        column=node.column,
                        bay=node.bay,
                        free_dofs=1,
                        storey=node.storey,
                        floor=node.floor,
                        zerolen=node.id,
                        orientation=o,
                        base=node.fixed,
                    )
                    nodes_by_id[node.id][o] = next_id
                    next_id += 1
                    nodes.append(n)

            self.nodes += nodes
            elements = []
            elem_id = 1
            recorder_ixs = {
                ElementTypes.SPRING_COLUMN.value: 1,
                ElementTypes.SPRING_BEAM.value: 1,
            }
            for elem in self.elements:
                i, j, bay, st, fl = elem.i, elem.j, elem.bay, elem.storey, elem.floor
                if elem.type == ElementTypes.BEAM.value:
                    imk_i, imk_j = (
                        nodes_by_id[i].pop("beam_left"),
                        nodes_by_id[j].pop("beam_right"),
                    )
                    type = ElementTypes.SPRING_BEAM.value
                else:
                    imk_i, imk_j = (
                        nodes_by_id[i].pop("col_down"),
                        nodes_by_id[j].pop("col_up"),
                    )
                    type = ElementTypes.SPRING_COLUMN.value

                data_imk1 = dict(
                    i=i,
                    j=imk_i,
                    type=type,
                    id=elem_id,
                    recorder_ix=recorder_ixs[type],
                )
                recorder_ixs[type] = recorder_ixs[type] + 1
                d1 = {**elem.to_dict, **data_imk1}
                imk1 = IMKSpring.from_bilin(**d1)
                elem_id += 1
                elements.append(imk1)
                Ic = imk1.Ic if elem.type == ElementTypes.COLUMN.value else 1000
                bc = BeamColumn(
                    id=elem_id,
                    radius=imk1.radius,
                    i=imk_i,
                    j=imk_j,
                    Ix=Ic,
                    E=elem.E,
                    type=elem.type,
                    storey=st,
                    bay=bay,
                    floor=fl,
                )
                elem_id += 1
                elements.append(bc)
                data_imk2 = dict(
                    i=imk_j,
                    j=j,
                    type=type,
                    id=elem_id,
                    recorder_ix=recorder_ixs[type],
                )
                recorder_ixs[type] = recorder_ixs[type] + 1
                d2 = {**elem.to_dict, **data_imk2}
                imk2 = IMKSpring.from_bilin(**d2)
                elements.append(imk2)
                elem_id += 1
            self.elements = elements
            self.done = True

    @property
    def element_recorders(self) -> str:
        s = f"region 1 -ele {self.spring_columns_ids_str}\n"
        s += f"recorder Element         -file $abspath/columns-M.csv -time      -region 1 -dof 3 force\n"
        s += f"recorder EnvelopeElement -file $abspath/columns-M-envelope.csv   -region 1 -dof 3 force\n"
        s += f"recorder Element         -file $abspath/columns-rot.csv -time    -region 1 -dof 2 deformation\n"
        s += f"recorder EnvelopeElement -file $abspath/columns-rot-envelope.csv -region 1 -dof 2 deformation\n"
        s += f"region 2 -ele {self.spring_beams_ids_str}\n"
        s += f"recorder Element         -file $abspath/beams-M.csv -time      -region 2 -dof 3 force\n"
        s += f"recorder EnvelopeElement -file $abspath/beams-M-envelope.csv   -region 2 -dof 3 force\n"
        s += f"recorder Element         -file $abspath/beams-rot.csv -time    -region 2 -dof 2 deformation\n"
        s += f"recorder EnvelopeElement -file $abspath/beams-rot-envelope.csv -region 2 -dof 2 deformation\n"
        return s

    def determine_collapse_from_results(self, results: dict):
        """
        uses the following criteria
        - residual drifts
        - dynamical instability
        - shear failure (Elwood drift)
        - too much damage everywhere, impossible to restore without demolishing
        """
        return False


class FEMFactory:
    DEFAULT = PlainFEM.__name__
    options = {
        PlainFEM.__name__: PlainFEM,
        RigidBeamFEM.__name__: RigidBeamFEM,
        ShearModel.__name__: ShearModel,
        BilinFrame.__name__: BilinFrame,
        IMKFrame.__name__: IMKFrame,
    }

    def __new__(cls, **data) -> FiniteElementModel:
        model = data.get("model", cls.DEFAULT)
        return cls.options[model](**data)

    @classmethod
    def add(cls, name, seed):
        cls.options[name] = seed

    @classmethod
    def models(cls) -> list[dict[str, str]]:
        return [{"label": name, "value": name} for name, _ in cls.options.items()]


class ElementFactory:
    DEFAULT = BilinBeamColumn.__name__
    options = {
        BeamColumn.__name__: BeamColumn,
        ElasticBeamColumn.__name__: ElasticBeamColumn,
        BilinBeamColumn.__name__: BilinBeamColumn,
        IMKSpring.__name__: IMKSpring,
    }

    def __new__(cls, **data) -> FiniteElementModel:
        model = data.get("model", cls.DEFAULT)
        return cls.options[model](**data)

    @classmethod
    def add(cls, name, seed):
        cls.options[name] = seed

    @classmethod
    def models(cls) -> list[dict[str, str]]:
        return [{"label": name, "value": name} for name, _ in cls.options.items()]
