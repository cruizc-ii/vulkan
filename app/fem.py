from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from app.assets import (
    AssetFactory,
    Asset,
)
from app.utils import (
    OPENSEES_ELEMENT_EDPs,
    OPENSEES_EDPs,
    OPENSEES_REACTION_EDPs,
)
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh, ishermitian
import pandas as pd
from pathlib import Path
from app.utils import GRAVITY, SummaryEDP, YamlMixin, ASSETS_PATH
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line, Pie
from app.concrete import DesignException, RectangularConcreteColumn
from app.elements import (
    FE,
    ConcreteElasticFoundation,
    Node,
    DiaphragmNode,
    BeamColumn,
    BilinBeamColumn,
    IMKSpring,
    ElasticBeamColumn,
    FEMValidationException,
    ElementTypes,
    ConcreteElasticSlab,
)


@dataclass
class FiniteElementModel(ABC, YamlMixin):
    nodes: list[Node]
    elements: list[FE]
    damping: float = 0.05
    model: str = "ABC"
    num_frames: int = 2
    num_storeys: float | None = None
    num_floors: float | None = None
    num_bays: float | None = None
    num_cols: float | None = None
    total_area: float | None = None
    depth: float | None = None
    occupancy: str | None = None
    masses: list[float] | None = None
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
    chopra_fundamental_period_plus1sigma: float | None = None
    miranda_fundamental_period: float | None = None
    cdmx_fundamental_period: float | None = None
    uniform_beam_loads_by_mass: list[float] | None = None

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
            self.nonstructural_elements = self.build_nonstructural_elements()

        if len(self.contents) > 0 and isinstance(self.contents[0], dict):
            self.contents = [AssetFactory(**data) for data in self.contents]
        else:
            self.contents = self.build_contents()

        self.model = self.__class__.__name__

        if self.pushover_abs_path:
            from app.strana import StructuralResultView

            self._pushover_view = StructuralResultView(
                abs_folder=self.pushover_abs_path
            )
        self.validate()

    @classmethod
    def from_spec(
        cls,
        spec: "BuildingSpecification",  # noqa: F821
        *args,
        fem: "FiniteElementModel" | None = None,  # noqa: F821
        **kwargs,
    ) -> "FiniteElementModel":
        nodes = (
            fem.nodes
            if fem is not None
            else [Node(id=id, **info) for id, info in spec.nodes.items()]
        )
        elements = (
            fem.elements
            if fem is not None
            else ElasticBeamColumn.from_adjacency(spec._adjacency)
        )
        # kwargs.pop("elements", [])
        # kwargs.pop("nodes", [])
        # fem_kwargs = fem.to_dict if fem is not None else {}
        # this is an antipattern, so many errors here.
        # how to make this better?
        _fem = cls(
            *args,
            nodes=nodes,
            elements=elements,
            masses=fem.masses if fem is not None else [],
            damping=spec.damping,
            occupancy=spec.occupancy,
            num_frames=spec.num_frames,
            num_cols=spec.num_cols,
            num_bays=spec.num_bays,
            num_floors=spec.num_floors,
            num_storeys=spec.num_storeys,
            total_area=spec.total_area,
            depth=spec.depth,
            chopra_fundamental_period_plus1sigma=spec.chopra_fundamental_period_plus1sigma,
            miranda_fundamental_period=spec.miranda_fundamental_period,
            cdmx_fundamental_period=spec.cdmx_fundamental_period,
            **kwargs,
            # **fem_kwargs,
        )
        print(_fem.masses, _fem.length)
        return _fem

    @property
    def period(self) -> float:
        # this should break when we call it without running modal first.
        return self.periods[0]

    def collapse_drift_estimation(self, ndf: pd.DataFrame) -> float:
        SHEAR_TOLERANCE = 0.01
        V_c = ndf["sum"].values[-1]
        u_c = ndf["u"].values[-1]
        if np.abs(V_c) < SHEAR_TOLERANCE:
            _df = ndf.iloc[20:]  # do not take the initial ones
            _df = _df.where(_df["sum"] < SHEAR_TOLERANCE).dropna()
            u_c = _df["u"].values[0]
        return u_c

    def pushover_stats(self) -> dict:
        df = self.pushover_df()
        ndf = self.normalized_pushover_df()
        Vy, uy = self.yield_estimation(df)
        drift_c = self.collapse_drift_estimation(ndf)
        drift_y = uy / self.height
        ductility = drift_c / drift_y
        cs = Vy / self.weight
        c_design = self.extras.get("c_design", 1)
        c_design_error = (cs - c_design) / c_design
        Vy_design = c_design * self.weight
        Vy_error = (Vy - Vy_design) / Vy_design
        period = self.periods[0] if len(self.periods) > 0 else 0
        chosen_period_formulae = self.cdmx_fundamental_period
        period_error = (
            (self.periods[0] - chosen_period_formulae) / chosen_period_formulae
            if len(self.periods) > 0
            else ""
        )
        stats = {
            "net worth [$]": self.total_net_worth,
            "elements net worth [$]": self.elements_net_worth,
            "nonstructural net worth [$]": self.nonstructural_net_worth,
            "contents net worth [$]": self.contents_net_worth,
            "Vy [kN]": Vy,
            "uy [m]": uy,
            "cs [1]": cs,
            "ductility": ductility,
            "drift_y [%]": 100 * drift_y,
            "c_design [1]": self.extras.get("c_design"),
            "period [s]": f"{period:.2f} s",
            "Vy": f"{Vy:.1f} kN",
            "Vy_design": f"{Vy_design:.1f} kN",
            "Vy_error": f"{100*Vy_error:.2f} %",
            "uy": f"{uy:.3f} [m]",
            "cs": f"{cs:.3f} [1]",
            "Sa_y_g": f"{cs:.3f} [g]",
            "drift_y": f"{100*drift_y:.2f} %",
            "drift_y [1]": drift_y,
            "c_design": f"{c_design:.3f} [1]",
            "design_error": f"{100*c_design_error:.2f} %",
            "miranda period [s]": f"{chosen_period_formulae:.2f} s",
            "period_error": f"{100*period_error:.1f} %",
            "_pushover_x": df["u"].to_list(),
            "_pushover_y": df["sum"].to_list(),
            "_norm_pushover_x": ndf["u"].to_list(),
            "_norm_pushover_y": ndf["sum"].to_list(),
        }
        return stats

    @property
    def elements_assets(self) -> list["Asset"]:
        # here we count only the elements that are indeed Assets
        # for instance, the elastic column used in the IMK ensemble is not an Asset
        return [ele for ele in self.elements if Asset in inspect.getmro(ele.__class__)]

    @property
    def assets(self) -> list["Asset"]:
        return self.elements_assets + self.nonstructural_elements + self.contents

    @property
    def slabs(self):
        return [s for s in self.elements if s.type == ElementTypes.SLAB.value]

    def build_and_place_slabs_and_foundation(
        self, Q: int = 1
    ) -> tuple[list[Asset], Asset]:
        slabs = []
        for floor, beams in enumerate(self.beams_by_storey, 1):
            for beam in beams:
                id = (
                    beam.id * 10000000
                )  # hack to avoid having duplicate ids, each beam is tied to a slab via an ID.
                slab = ConcreteElasticSlab(id=id, length=beam.length, floor=floor, Q=Q)
                slabs.append(slab)
        for asset in slabs:
            if asset.net_worth:
                asset.net_worth = (self.num_frames - 1) * asset.net_worth
        foundation = ConcreteElasticFoundation(
            id=id * 2, length=self.length, Q=Q, floor=1
        )
        return slabs, foundation

    def build_nonstructural_elements(self) -> list[Asset]:
        from app.occupancy import BuildingOccupancy

        occupancy = BuildingOccupancy(fem=self, model_str=self.occupancy)
        nonstructural_and_contents = occupancy.build()
        nonstructural = [
            a for a in nonstructural_and_contents if a.category == "nonstructural"
        ]  # categories can be any string but we should ideally filter by ENUM
        # collocation algorithm already takes care of spatial distribution.
        # for asset in nonstructural:
        #     if asset.net_worth:
        #         asset.net_worth = (self.num_frames - 1) * asset.net_worth
        return nonstructural

    def build_contents(self) -> list[Asset]:
        from app.occupancy import BuildingOccupancy

        occupancy = BuildingOccupancy(fem=self, model_str=self.occupancy)
        nonstructural_and_contents = occupancy.build()
        contents = [
            a for a in nonstructural_and_contents if a.category == "contents"
        ]  # categories can be any string but we should ideally filter by ENUM
        # collocation algorithm already takes care of spatial distribution.
        # for asset in contents:
        #     if asset.net_worth:
        #         asset.net_worth = self.num_frames * asset.net_worth
        return contents

    @property
    def readable_total_net_worth(self) -> str:
        total = self.total_net_worth
        s = "$ "
        if total >= 1000:
            s += f"{total/1000:.2f} M"
        else:
            s += f"{total:.0f} k"
        return s + " USD"

    @property
    def total_net_worth(self) -> float:
        return (
            self.elements_net_worth
            + self.nonstructural_net_worth
            + self.contents_net_worth
        )

    @property
    def cost_per_unit_area(self) -> float:
        return self.total_net_worth / self.total_area

    @property
    def eigen_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        eigen_table = pd.DataFrame(
            dict(
                periods=self.periods,
                freqs=self.frequencies,
                omegas=self.omegas,
            ),
            index=range(1, len(self.periods) + 1),
        )
        storey_table = pd.DataFrame(
            dict(
                mass=self.masses,
                weights=self.weights,
                loads_kN_per_m=self.uniform_beam_loads,
                loads_kPa=self.uniform_area_loads_kPa,
            ),
            index=range(1, len(self.periods) + 1),
        )
        return eigen_table, storey_table

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

    def column_beam_ratios(self, key: str = "My") -> pd.DataFrame:
        df = self.structural_elements_breakdown()
        df = df[df["type"].str.contains("spring_column|spring_beam")]
        xs, ys = sorted(list(set(df.x.values))), sorted(list(set(df.y.values)))
        n_cols, n_floors = len(xs), len(ys)
        M = np.full((3 * n_floors, 3 * n_cols), np.nan)
        for fl, y in enumerate(ys):
            for col, x in enumerate(xs):
                springs = df[(df.x == x) & (df.y == y)]
                up_spring = springs[springs.up == True]
                right_spring = springs[springs.right == True]
                down_spring = springs[springs.down == True]
                left_spring = springs[springs.left == True]
                up, right, down, left = 0.0, 0.0, 0.0, 0.0
                if not up_spring.empty:
                    up = up_spring[key].values[0]
                if not down_spring.empty:
                    down = down_spring[key].values[0]
                if not left_spring.empty:
                    left = left_spring[key].values[0]
                if not right_spring.empty:
                    right = right_spring[key].values[0]
                cols = up + down
                beams = left + right
                ratio = cols / beams if beams != 0 else np.nan
                # x coord=floors,  ycoord= cols
                M[3 * fl, 3 * col + 1] = up
                M[3 * fl + 2, 3 * col + 1] = down
                M[3 * fl + 1, 3 * col + 2] = left
                M[3 * fl + 1, 3 * col] = right
                M[3 * fl + 1, 3 * col + 1] = ratio

        M = np.flip(M)
        return pd.DataFrame(M)

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

    def to_tcl(self, filepath: Path) -> None:
        with open(filepath, "w+") as f:
            f.write(self.model_str)

    def determine_collapse_from_results(
        self, results: dict, view_handler
    ) -> "CollapseTypes":
        """
        upper bound on collapse, if any column is in DS==Collapse
        collapse_elwood = determine_collapse_elwood()
        collapse_residual =
        return collapse_damage or collapse_residual or collapse_elwood
        """
        from app.utils import CollapseTypes

        collapse = CollapseTypes.NONE.value
        for st, columns in enumerate(self.columns_by_storey):
            column = columns[0]
            x = results[SummaryEDP.peak_drifts.value][st]
            ds = column._risk.simulate_damage_state(x)
            if ds == column._risk.damage_states[-1]["name"]:
                collapse = CollapseTypes.INSTABILITY.value
                break
        return collapse

    def validate(self, *args, **kwargs):
        # TODO: no dangling nodes, at least one element has that ID in i or j,
        # TODO: no dangling elements, all element i,j IDs must exist in nodeSet
        if len(set(self.nodeIDs)) != len(self.nodeIDs):
            raise FEMValidationException(
                "Node ids must be unique" + f"{sorted(self.nodeIDs)}"
            )
        if len(set(self.elementIDs)) != len(self.elementIDs):
            raise FEMValidationException(
                "Element ids must be unique" + f"{sorted(self.elementIDs)}"
            )
        if len(self.fixed_nodes) == 0:
            raise FEMValidationException(
                "At least one fixed node must exist!" + f"{self.nodes_str}"
            )

    def get_and_set_eigen_results(self, results_path: Path):
        from app.strana import StructuralAnalysis

        strana = StructuralAnalysis(results_path=results_path, fem=self)
        view = strana.modal()
        self.periods = view.periods
        self.frequencies = view.frequencies
        self.omegas = view.omegas
        self.values = view.values
        self.vectors = view.vectors
        self.a0, self.a1 = self._rayleigh()
        return view

    def _rayleigh(self) -> tuple[float, float]:
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

    def pushover(self, results_path: Path, drift: float = 0.05, mode: int | None = 1):
        from app.strana import StructuralAnalysis
        from app.utils import AnalysisTypes

        vectors = None
        if mode is not None:
            if self.vectors is None:
                self.get_and_set_eigen_results(results_path=results_path)
            vecs = np.array(self.vectors).T
            vectors = vecs[mode - 1]
        strana = StructuralAnalysis(results_path, fem=self)
        view = strana.pushover(drift=drift, modal_vectors=vectors)
        self.pushover_abs_path = str(results_path / AnalysisTypes.PUSHOVER.value)
        self._pushover_view = view
        return view

    def pushover_figs(
        self,
        # results_path: Path,
        # drift: float = 0.05,
        # force=False,
    ):
        # if force or not self._pushover_view:
        #     self.pushover(results_path=results_path, drift=drift)

        df = self.pushover_df()
        ndf = self.normalized_pushover_df()
        cols = df[df.columns.difference(["u"])].columns
        fig = df.plot(x="u", y=cols)
        fig.update_layout(
            xaxis_title="roof u [m]",
            yaxis_title="Vb [kN]",
            title_text=f"1st mode pushover",
        )
        normalized_fig = ndf.plot(x="u", y=cols)
        normalized_fig.update_layout(
            xaxis_title="roof drift [1]",
            yaxis_title="cs",
            title_text=f"Normalized 1st mode pushover",
            xaxis_range=[0.0, 0.05],
        )
        return fig, normalized_fig

    def pushover_df(self) -> pd.DataFrame:
        view = self._pushover_view
        Vb = -view.base_shear()
        Vb["sum"] = Vb.sum(axis=1)
        roof_disp = view.roof_displacements()
        df = Vb.join(roof_disp)
        return df

    def normalized_pushover_df(self) -> pd.DataFrame:
        df = self.pushover_df()
        h, W = self.height, self.weight
        df["u"] = df["u"] / h
        cols = df.columns[~df.columns.isin(["u"])]
        df[cols] = df[cols] / W
        return df

    def pushover_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        view = self._pushover_view
        columns: list[IMKSpring] = self.springs_columns[::2]  # take the top spring
        bottom_columns = columns[: self.num_cols]
        # print(view.view_column_design_axials())
        # print(view.view_column_axials(envelope=True))
        axial_envelopes = view.view_column_design_axials()[: self.num_cols]
        VRs = [
            c.aci_shear_capacity(axial_force=P)
            for c, P in zip(bottom_columns, axial_envelopes)
        ]
        # VRs2 = [c.ntc_shear_capacity() for c in bottom_columns]
        VRs2 = [c.Vcr for c in bottom_columns]
        # print(VRs2)
        VR2 = sum(VRs2)
        VR = sum(VRs)
        # VRs = [c.VR for c in bottom_columns]
        # VR = sum(VRs)
        # print(VR)
        Vb = view.base_shear()
        VR = [VR for _ in Vb.index]
        Vb = -Vb
        Vb["sum"] = Vb.sum(axis=1)
        roof_disp = view.roof_displacements()
        Vb["VR"] = VR
        Vb["VR_ntc"] = VR2
        df = Vb.join(roof_disp)
        cs = Vb / self.weight
        roof_drifts = roof_disp / self.height
        ndf = cs.join(roof_drifts)
        return df, ndf

    @staticmethod
    def yield_estimation(df: pd.DataFrame) -> tuple[float, float]:
        _YIELD_TANGENT_PCT = 0.40  # curr_tangent < PCT*initial tangent => yield point.
        df = df.reset_index()
        df2 = df.diff(1)
        df2["fp"] = (
            df2["sum"] / df2["u"]
        )  # an idea might be to use a better approximation of derivative as (f(x+e)-f(x-e))/2e. will need an applymap.
        K = df2["fp"].iloc[:10].mean()
        tol = -0.05
        possible_pts = np.where(df2.diff(1, axis=0) < -tol)[
            0
        ]  # where next tangent is smaller than previous by a small negative tolerance
        df3 = df2.iloc[possible_pts]
        indices = df3["fp"][df3["fp"] < _YIELD_TANGENT_PCT * K].index
        if len(indices) == 0:
            return df["sum"].max(), 0
        yield_point = indices[0]
        Vy, uy = df[["sum", "u"]].iloc[yield_point]
        return Vy, uy

    @property
    def uniform_area_loads_kPa(self) -> list[float]:
        area = self.length**2
        pressures_per_storey = np.array(self.weights) / area
        return pressures_per_storey.tolist()

    @property
    def uniform_beam_loads(self) -> list[float]:
        if self.length == 0:
            beam_loads = [0.0 for _ in self.masses]
        else:
            beam_loads = GRAVITY * np.array(self.masses) / self.length
            beam_loads = beam_loads.tolist()
        return beam_loads

    @property
    def height(self) -> float:
        ys = [n.y for n in self.nodes]
        return max(ys)

    @property
    def length(self) -> float:
        xs = [n.x for n in self.nodes]
        return max(xs)

    @property
    def width(self) -> float:
        return self.length

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
            if e.type not in [ElementTypes.SLAB.value, ElementTypes.FOUNDATION.value]
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
    def beams_by_storey(self) -> list[list[ElasticBeamColumn]]:
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

    # @property
    # def masses(self):
    #     return [n.mass for n in self.mass_nodes]

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
    def storey_stiffnesses(self) -> list[float]:
        ks = []
        for cols in self.columns_by_storey:
            cols_k = [c.get_and_set_k() for c in cols]
            ks.append(sum(cols_k))
        return ks

    @property
    def storey_inertias(self) -> list[float]:
        Ix = []
        for cols in self.columns_by_storey:
            cols_k = [c.Ix for c in cols]
            Ix.append(sum(cols_k))
        return Ix

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
        s += f"recorder NodeEnvelope    -file $abspath/roof-displacements-env.csv -time -node {self.roofID} -dof 1 disp\n"
        s += f"recorder Node            -file $abspath/roof-accels.csv -time -node {self.roofID} -dof 1 accel\n"
        s += f"recorder NodeEnvelope    -file $abspath/roof-accels-env.csv -time -node {self.roofID} -dof 1 accel\n"

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
    model: str = "PlainFEM"


@dataclass
class ShearModel(FiniteElementModel):
    """
    can run opensees to get periods!
    disregards beams completely.
    takes lateral stiffness as sum(k_columns)
    """

    _mass_matrix: np.ndarray | None = None
    _height_matrix: np.ndarray | None = None
    _stifness_matrix: np.ndarray | None = None
    _inertias: np.ndarray | None = None

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

        if self._inertias is not None:
            for i, (cols, prev_col) in enumerate(
                zip(columns_by_storey, first_bay_columns)
            ):
                data = prev_col.to_dict
                data["Ix"] = self._inertias[i]
                data["radius"] = None
                columns.append(ElasticBeamColumn(**data))
        else:
            for cols, prev_col in zip(columns_by_storey, first_bay_columns):
                data = prev_col.to_dict
                Ixs = [c.Ix for c in cols]
                data["Ix"] = sum(Ixs)
                data["radius"] = None
                columns.append(ElasticBeamColumn(**data))

        self.num_dofs = len(columns)
        self.elements = columns

        self._mass_matrix = np.diag(self.masses)
        heights = [n.y for n in self.nodes if not n.fixed]
        self._height_matrix = np.diag(heights)

        ks = np.array(self.storey_stiffnesses)
        upper = -np.diagflat(ks[1:], 1)
        lower = -np.diagflat(ks[1:], -1)
        ks2 = np.roll(np.copy(ks), -1)
        ks2[-1] = 0
        self._stifness_matrix = np.diag(ks) + np.diag(ks2) + upper + lower
        K, M = self._stifness_matrix, self._mass_matrix
        isher = ishermitian(K)
        if not isher:
            raise DesignException("Stiffness matrix is not positive definite")
        vals, vecs = eigh(K, M)
        # vals, vecs = vals[::-1], vecs[::-1]
        omegas = np.sqrt(vals)
        freqs = omegas / 2 / np.pi
        Ts = 1.0 / freqs
        self.periods = Ts
        self.frequencies = freqs
        self.vectors = vecs
        self.omegas = omegas
        self.validate()

    def build_and_place_assets(self) -> list[Asset]:
        """no room for assets"""
        return []

    @property
    def storey_radii(self) -> list[float]:
        return self.column_radii

    def get_and_set_eigen_results(
        self, results_path: Path
    ) -> "StructuralResultView":  # noqa: F821
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
        view.inertial_forces = S
        view.shears = V
        view.overturning_moments = Mb
        view.effective_masses = effective_masses
        view.effective_heights = effective_heights
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
        # if isinstance(self.elements[0], dict):
        #     self.elements = [BilinBeamColumn(**data) for data in self.elements]
        # else:
        #     self.elements = [BilinBeamColumn(**elem.to_dict) for elem in self.elements]

    @classmethod
    def from_elastic(
        cls,
        fem: FiniteElementModel,
        design_moments: list[float],
        beam_column_ratio: float = 1.0 / 1.5,
        Q: float = 1.0,
    ) -> "BilinFrame":
        """
        will take design moments per storey and put them into beams and columns
        """
        col_moments = np.array(design_moments)
        beam_moments = beam_column_ratio * col_moments
        for cm, cols, bm, beams in zip(
            col_moments, fem.columns_by_storey, beam_moments, fem.beams_by_storey
        ):
            for c in cols:
                # c.My = cm/fem.num_cols
                c.model = "BilinBeamColumn"
                c.My = cm
                c.Q = Q
            for b in beams:
                # b.My = bm/fem.num_cols
                b.model = "BilinBeamColumn"
                b.My = bm
                b.Q = Q

        instance = cls(**fem.to_dict)
        return instance

    @property
    def elements_str(self) -> str:
        """this is a really dumb limitation of opensees, every E in Domain
        must have a non-null ID.. so we have to do gymnastics here to fill in the sections
        """

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
    def springs_columns_by_storey(self):
        st = []
        for iy in range(1, self.num_modes + 1):
            cols = [col for col in self.springs_columns if col.storey == iy]
            st.append(cols)
        return st

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
            node_coordinates_by_id = {}
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
                    node_coordinates_by_id[node.id] = dict(x=node.x, y=node.y)
                    next_id += 1
                    nodes.append(n)

            self.nodes += nodes
            elements = []
            elem_id = 1
            recorder_ixs = {
                ElementTypes.SPRING_COLUMN.value: 1,
                ElementTypes.SPRING_BEAM.value: 1,
            }
            beams_and_columns = [
                e
                for e in self.elements
                if (
                    e.type == ElementTypes.BEAM.value
                    or e.type == ElementTypes.COLUMN.value
                )
            ]
            for elem in beams_and_columns:
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

                x, y = node_coordinates_by_id[i]["x"], node_coordinates_by_id[i]["y"]
                up, left, down, right = (
                    False,
                    elem.type == ElementTypes.BEAM.value,
                    elem.type == ElementTypes.COLUMN.value,
                    False,
                )
                data_imk1 = dict(
                    i=i,
                    j=imk_i,
                    x=x,
                    y=y,
                    down=down,
                    left=left,
                    type=type,
                    id=elem_id,
                    recorder_ix=recorder_ixs[type],
                    net_worth=0,
                )
                recorder_ixs[type] = recorder_ixs[type] + 1
                d1 = {**elem.to_dict, **data_imk1}
                imk1 = IMKSpring.from_bilin(**d1)
                imk1.net_worth = (2 * self.num_frames - 1) * imk1.net_worth
                # TR152_Ibarra-Krawinkler-2005.pdf p.298
                Ks = imk1.Ks
                Kmem = elem.Kbc
                # Kbc = (Kmem * Ks) / (Ks - 2 * Kmem)
                n = 10
                Kbc = (n + 2) / n * Kmem
                EI_bc = imk1.length * Kbc / 6
                elem_id += 1
                elements.append(imk1)
                Ic = imk1.Ic if elem.type == ElementTypes.COLUMN.value else 1e5
                bc = BeamColumn(
                    id=elem_id,
                    radius=imk1.radius,
                    i=imk_i,
                    j=imk_j,
                    # Ix=elem.Ix,
                    # E=elem.E,
                    Ix=EI_bc,
                    E=1.0,
                    type=elem.type,
                    storey=st,
                    bay=bay,
                    floor=fl,
                )
                elem_id += 1
                elements.append(bc)
                x, y = node_coordinates_by_id[j]["x"], node_coordinates_by_id[j]["y"]
                up, left, down, right = (
                    elem.type == ElementTypes.COLUMN.value,
                    False,
                    False,
                    elem.type == ElementTypes.BEAM.value,
                )
                data_imk2 = dict(
                    i=imk_j,
                    j=j,
                    x=x,
                    y=y,
                    up=up,
                    right=right,
                    type=type,
                    id=elem_id,
                    recorder_ix=recorder_ixs[type],
                    net_worth=0,
                )
                recorder_ixs[type] = recorder_ixs[type] + 1
                d2 = {**elem.to_dict, **data_imk2}
                imk2 = IMKSpring.from_bilin(**d2)
                imk2.net_worth = (2 * self.num_frames - 1) * imk2.net_worth
                elements.append(imk2)
                elem_id += 1
            slabs = self.slabs
            foundation = [
                s for s in self.elements if s.type == ElementTypes.FOUNDATION.value
            ]
            self.elements = elements + slabs + foundation
            self.done = True

    @property
    def element_recorders(self) -> str:
        s = super().element_recorders
        s += f"region 1 -ele {self.spring_columns_ids_str}\n"
        s += f"recorder Element         -file $abspath/columns-M.csv -time      -region 1 -dof 3 force\n"
        s += f"recorder EnvelopeElement -file $abspath/columns-M-envelope.csv   -region 1 -dof 3 force\n"
        s += f"recorder Element         -file $abspath/columns-rot.csv -time    -region 1 deformation\n"
        s += f"recorder EnvelopeElement -file $abspath/columns-rot-envelope.csv -region 1 deformation\n"
        s += f"region 2 -ele {self.spring_beams_ids_str}\n"
        s += f"recorder Element         -file $abspath/beams-M.csv -time      -region 2 -dof 3 force\n"
        s += f"recorder EnvelopeElement -file $abspath/beams-M-envelope.csv   -region 2 -dof 3 force\n"
        s += f"recorder Element         -file $abspath/beams-rot.csv -time    -region 2 deformation\n"
        s += f"recorder EnvelopeElement -file $abspath/beams-rot-envelope.csv -region 2 deformation\n"
        return s

    def determine_collapse_from_results(
        self, results: dict, view_handler: "StructuralResultView"
    ) -> "CollapseTypes":
        """
        simplification of the actual problem
        1. constant axial force (P = dead weight / num_columns)
        2. uses a single column from each storey (when they could be built with different steel ratios, but we assume they are more or less the same and we are thinking rigid diaphragms..)
        """
        from app.utils import CollapseTypes

        collapse_type = self.determine_intra_analysis_collapse(
            results, view_handler=view_handler
        )
        if collapse_type == CollapseTypes.NONE.value:
            collapse_type = self.determine_extra_analysis_collapse(
                results, view_handler=view_handler
            )
        return collapse_type

    def determine_intra_analysis_collapse(
        self, results: dict, view_handler: "StructuralResultView"
    ) -> "CollapseTypes":
        """
        read collapse.csv written by dynamic_integrator.tcl
        if that file exists we have intra-analysis collapse.
        """
        from app.utils import CollapseTypes

        collapse_type = CollapseTypes.NONE.value
        try:
            file = view_handler.read_collapse_file()
            if file is not None:
                collapse_type = CollapseTypes.INSTABILITY.value
        except FileNotFoundError as e:
            print(e)
        return collapse_type

    def determine_extra_analysis_collapse(
        self, results: dict, view_handler: "StructuralResultView"
    ) -> "CollapseTypes":
        from app.utils import CollapseTypes

        return CollapseTypes.NONE.value

        shears = view_handler.view_column_design_shears()
        axials = view_handler.view_column_design_axials()
        columns: list[IMKSpring] = self.springs_columns[::2]  # take the top spring
        # capacities = [
        #     column.elwood_shear_capacity(shear, axial)
        #     for column, shear, axial in zip(columns, shears, axials)
        # ]
        capacities = [
            column.ntc_shear_capacity(shear, axial)
            for column, shear, axial in zip(columns, shears, axials)
        ]
        shears = np.array(shears)
        capacities = np.array(capacities)
        print(shears > capacities)
        # drifts_by_storey = view_handler.view_peak_drifts().to_list()
        # drifts = [d for d in drifts_by_storey for _ in range(self.num_cols)]
        # collapses = [d > capacity for d, capacity in zip(drifts, capacities)]
        collapses = [d > capacity for d, capacity in zip(shears, capacities)]
        shear_collapse = (
            CollapseTypes.SHEAR.value if any(collapses) else CollapseTypes.NONE.value
        )
        return shear_collapse


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
    # DEFAULT = ElasticBeamColumn.__name__
    options = {
        BeamColumn.__name__: BeamColumn,
        ConcreteElasticSlab.__name__: ConcreteElasticSlab,
        ConcreteElasticFoundation.__name__: ConcreteElasticFoundation,
        ElasticBeamColumn.__name__: ElasticBeamColumn,
        BilinBeamColumn.__name__: BilinBeamColumn,
        IMKSpring.__name__: IMKSpring,
    }

    def __new__(cls, **data) -> FiniteElementModel:
        model = data.get("model")
        return cls.options[model](**data)

    @classmethod
    def add(cls, name, seed):
        cls.options[name] = seed

    @classmethod
    def models(cls) -> list[dict[str, str]]:
        return [{"label": name, "value": name} for name, _ in cls.options.items()]
