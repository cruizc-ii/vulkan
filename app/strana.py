from __future__ import annotations
from app.fem import Node, ElasticBeamColumn, FiniteElementModel
from app.hazard import Hazard, Record
from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import Union, Optional
from app.utils import (
    EDP,
    GRAVITY,
    AnalysisTypes,
    NamedYamlMixin,
    OPENSEES_EDPs,
    OPENSEES_ELEMENT_EDPs,
    OPENSEES_REACTION_EDPs,
    IDAResultsDataFrame,
    SummaryEDP,
    YamlMixin,
)
from dataclasses import dataclass, field
from plotly.graph_objects import Figure
import numpy as np
from numpy import flip, cumsum
import pandas as pd
from pathlib import Path
import os
import subprocess
from shortuuid import uuid
from app.codes import BuildingCode
from app.utils import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
STRANA_DIR = MODELS_DIR / "strana_models"

pd.set_option("plotting.backend", "plotly")


@dataclass
class StructuralResultView(YamlMixin):
    abs_folder: str
    values: Optional[list] = None
    vectors: Optional[list] = None
    periods: Optional[list] = None
    omegas: Optional[list] = None
    frequencies: Optional[list] = None
    peak_drifts: Optional[list] = None
    peak_floor_accels: Optional[list] = None
    record: Record = None
    scale: float = None

    _DEFAULT_NAME = "results.yml"
    _K_STATIC_NAME = "K-static.csv"
    _cache_modal_results: dict = None
    _init: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.abs_folder, Path):
            self._path = self.abs_folder
            self.abs_folder = str(self.abs_folder.resolve())
        else:
            self._path = Path(self.abs_folder)

    def __lazy_init__(self) -> None:
        if self._init:
            return
        if isinstance(self.record, dict):
            self.record = Record(**self.record)
        self._init = True

    def to_file(self) -> None:
        filepath = self._path / self._DEFAULT_NAME
        return super().to_file(filepath)

    @classmethod
    def from_file(cls, path: Path):
        filepath = path / cls._DEFAULT_NAME
        return super().from_file(filepath)

    def view_result_by_edp_and_node(self, edp: EDP, node: int) -> pd.DataFrame:
        fns = {
            EDP.rotations_env.value: self.view_rotations_envelope,
            # EDP.disp_env.value: self.view_displacements_envelope
        }
        result = fns[edp](node=node)
        return result

    def _read_eigen_values(self) -> pd.DataFrame:
        values = pd.read_csv(self._path / "eigen_values.csv", sep="\s+", header=None)
        return values

    def _read_eigen_vectors(self) -> pd.DataFrame:
        vectors = pd.read_csv(self._path / "eigen_vectors.csv", sep="\s+", header=None)
        return vectors

    def view_modal_results(self) -> dict[str, list]:
        """
        what we need is the modal expansion of the effective earthquake forces m\iota. on what follows
        everything will be done via matrix calculus. each column will correspond to a mode
        each row is an active inertial DOF. we seek S_ij which expresses
        the effective masses acting at each DOF for each mode.
        """
        if not self._cache_modal_results:
            values = self._read_eigen_values()
            values = values.to_numpy().flatten()
            vectors = self._read_eigen_vectors()
            vectors = vectors.to_numpy().T

            omegas = np.sqrt(values)
            periods = 2 * np.pi / omegas
            frequencies = 1.0 / periods

            self._cache_modal_results = {
                "values": values.tolist(),
                "vectors": vectors.tolist(),
                "omegas": omegas.tolist(),
                "periods": periods.tolist(),
                "frequencies": frequencies.tolist(),
            }
        return self._cache_modal_results

    def get_and_set_modal_results(self) -> dict[str, list]:
        results = self.view_modal_results()
        self.values = results["values"]
        self.vectors = results["vectors"]
        self.omegas = results["omegas"]
        self.periods = results["periods"]
        self.frequencies = results["frequencies"]
        return results

    def _read_envelope(self, filename: Union[Path, str], **kwargs) -> pd.DataFrame:
        """return absmax() response"""
        path = self._path / filename
        return pd.read_csv(path, header=None, sep="\s+", **kwargs)
        #  skiprows=[0, 1])

    def _read_timehistory(self, filename: Union[Path, str], names=None) -> pd.DataFrame:
        """return absmax() response"""
        path = self._path / filename
        return pd.read_csv(path, header=None, sep="\s+", index_col=0, names=names)

    def reactions_env(self) -> dict[str, DataFrame]:
        results = {}
        for reaction, _ in OPENSEES_REACTION_EDPs:
            envelope_name = f"{reaction}-envelope.csv"
            results[reaction] = self._read_envelope(envelope_name).iloc[-1]
        return results

    def reactions_timehistory(self) -> dict[str, DataFrame]:
        results = {}
        for reaction, _ in OPENSEES_REACTION_EDPs:
            name = f"{reaction}.csv"
            results[reaction] = self._read_timehistory(name)
        return results

    def reactions(self) -> dict[str, DataFrame]:
        results = {}
        for edp, _ in OPENSEES_REACTION_EDPs:
            name = f"{edp}.csv"
            res = self._read_timehistory(name).iloc[-1]
            results[edp] = res.values.flatten()
        return results

    def base_shear(self) -> DataFrame:
        Vbs = self._read_timehistory("base-shear.csv")
        return Vbs

    def peak_base_shear(self) -> DataFrame:
        env = self._read_envelope("base-shear-env.csv")
        return env.iloc[-1]

    def view_displacements_envelope(self, node: int, dof: int) -> DataFrame:
        """
        is there a better way to do this? does a Recorder have a notion of roof, x, y, z..
        where does that complexity live?
        this is very imperative.
        """
        env = self._read_envelope("node-disp-envelope.csv")
        col = (node - 1) * 3 + dof - 1
        disp = env[col].values.sum()
        return disp

    def view_mass_disp_env(self) -> DataFrame:
        env = self._read_envelope("mass-disp-envelope.csv")
        return env.iloc[-1]

    def peak_roof_disp(self) -> float:
        disp = self.view_mass_disp_env()
        return float(disp.values[-1])

    def view_mass_displacements(self) -> DataFrame:
        disp = self._read_timehistory("mass-displacements.csv")
        return disp

    def roof_displacements(self) -> float:
        disp = self._read_timehistory("roof-displacements.csv", names=["u"])
        return disp

    def view_rotations_envelope(self, node: int) -> DataFrame:
        return self.view_displacements_envelope(node, 3)

    def _view_column_forces(self, envelope=False) -> DataFrame:
        if not envelope:
            filename = "columns.csv"
            forces = self._read_timehistory(filename)
        else:
            filename = "columns-envelope.csv"
            forces = self._read_envelope(filename)
        return forces

    def view_column_forces(self, dof: int, envelope: bool = False) -> DataFrame:
        all_forces = self._view_column_forces(envelope=envelope)
        if envelope:
            dofs = [c for c in all_forces.columns if (c - dof + 1) % 3 == 0]
        else:
            dofs = [c - 1 for c in all_forces.columns if (c - dof) % 3 == 0]
        dof_forces = all_forces.iloc[:, dofs]
        return dof_forces

    def view_column_axials(self, envelope: bool = False) -> DataFrame:
        return self.view_column_forces(1, envelope=envelope)

    def view_column_shears(self, envelope: bool = False) -> DataFrame:
        return self.view_column_forces(2, envelope=envelope)

    def view_column_moments(self, envelope: bool = False) -> DataFrame:
        return self.view_column_forces(3, envelope=envelope)

    def view_column_design_axials(self) -> DataFrame:
        axials = self.view_column_axials(envelope=True)
        max_abs_both_ends = axials.iloc[-1]
        elems = iter(axials.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def view_column_design_shears(self) -> DataFrame:
        shears = self.view_column_shears(envelope=True)
        max_abs_both_ends = shears.iloc[-1]
        elems = iter(shears.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def view_column_design_moments(self) -> DataFrame:
        moments = self.view_column_moments(envelope=True)
        max_abs_both_ends = moments.iloc[-1]
        elems = iter(moments.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def _view_beam_forces(self, envelope: bool = False) -> DataFrame:
        if not envelope:
            filename = "beams.csv"
            forces = self._read_timehistory(filename)
        else:
            filename = "beams-envelope.csv"
            forces = self._read_envelope(filename)
        return forces

    def view_beam_forces(self, dof: int, envelope: bool = False) -> DataFrame:
        all_forces = self._view_beam_forces(envelope=envelope)
        if envelope:
            dofs = [c for c in all_forces.columns if (c - dof + 1) % 3 == 0]
        else:
            dofs = [c - 1 for c in all_forces.columns if (c - dof) % 3 == 0]
        dof_forces = all_forces.iloc[:, dofs]
        return dof_forces

    def view_beam_axials(self, envelope: bool = False) -> DataFrame:
        return self.view_beam_forces(1, envelope=envelope)

    def view_beam_shears(self, envelope: bool = False) -> DataFrame:
        return self.view_beam_forces(2, envelope=envelope)

    def view_beam_moments(self, envelope: bool = False) -> DataFrame:
        return self.view_beam_forces(3, envelope=envelope)

    def view_beam_design_axials(self) -> DataFrame:
        axials = self.view_beam_axials(envelope=True)
        max_abs_both_ends = axials.iloc[-1]
        elems = iter(axials.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def view_beam_design_shears(self) -> DataFrame:
        shears = self.view_beam_shears(envelope=True)
        max_abs_both_ends = shears.iloc[-1]
        elems = iter(shears.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def view_beam_design_moments(self) -> DataFrame:
        moments = self.view_beam_moments(envelope=True)
        max_abs_both_ends = moments.iloc[-1]
        elems = iter(moments.columns)
        max_any_end = [max(max_abs_both_ends[list(ij)]) for ij in zip(elems, elems)]
        return max_any_end

    def view_node_reactions_envelope(self, node: int, dof: int) -> DataFrame:
        filename = "node-reaction-envelope.csv"
        env = self._read_envelope(filename)
        col = (node - 1) * 3 + dof - 1
        reaction = env[col].values.sum()
        return reaction

    def view_moments_envelope(
        self,
    ) -> DataFrame:
        filename = "node-reaction-envelope.csv"
        env = self._read_envelope(filename)
        moments = env.iloc[:, [c for c in env.columns if (c + 1) % 3 == 0]]
        return moments

    def view_axials(self) -> DataFrame:
        filename = "node-reaction.csv"
        env = self._read_timehistory(filename)
        axials = env.iloc[:, [c for c in env.columns if (c + 1) % 4 == 0]]
        return axials

    def view_shears(self) -> DataFrame:
        filename = "node-reaction.csv"
        env = self._read_timehistory(filename)
        shears = env.iloc[:, [c for c in env.columns if (c + 1) % 2 == 0]]
        return shears

    def view_moments(self) -> DataFrame:
        filename = "node-reaction.csv"
        env = self._read_timehistory(filename)
        moments = env.iloc[:, [c for c in env.columns if (c + 1) % 3 == 0]]
        return moments

    def view_drifts(self) -> DataFrame:
        filename = "drifts.csv"
        return self._read_timehistory(filename)

    def view_floor_accels(self) -> DataFrame:
        self.__lazy_init__()
        filename = "mass-accel.csv"
        storey_accels = self._read_timehistory(filename)
        ground_accel = self.scale * self.record._df.to_frame()
        # accels = storey_accels.add(ground_accel, axis='index', fill_value=0).fillna(0)
        accels = storey_accels.merge(
            ground_accel, how="outer", left_index=True, right_index=True
        )
        accels = accels.interpolate("cubic").fillna(0)
        return accels

    def view_floor_accels_envelope(self) -> DataFrame:
        filename = "mass-accel-envelope.csv"
        return self._read_envelope(filename)

    def view_floor_vels_envelope(self) -> DataFrame:
        filename = "mass-vel-envelope.csv"
        return self._read_envelope(filename)

    def view_node_accels(self) -> Figure:
        filename = "node-accel.csv"
        return self._read_timehistory(filename)

    def drifts_plot(self) -> Figure:
        fig = self.view_drifts().plot()
        fig.update_layout(xaxis_title="t (s)", yaxis_title="drifts per storey [1]")
        return fig

    def moments_plot(self) -> Figure:
        fig = self.view_moments().plot()
        return fig

    def floor_accels_plot(self) -> Figure:
        fig = self.view_floor_accels().plot()
        fig.update_layout(xaxis_title="t (s)", yaxis_title="accels per floor m/s/s")
        return fig

    def floor_accels_plot_in_g(self) -> Figure:
        accels = self.view_floor_accels()
        accels = accels / GRAVITY
        fig = accels.plot()
        fig.update_layout(xaxis_title="t (s)", yaxis_title="accels per floor (g)")
        return fig

    def normalized_floor_accels_plot(self) -> Figure:
        fig = self.view_normalized_absolute_floor_accels().plot()
        fig.update_layout(xaxis_title="t (s)", yaxis_title="PFA/PGA")
        return fig

    @property
    def timehistory_figures(self) -> list[dcc.Graph]:
        """
        TODO; this is interesting.. a better approach is to
        define DRIFTS.csv NODE-ACCEL.csv as EDPs.
        filenames <-> EDP and we could just
        use a mapper
        """
        figs = []
        figs.append(self.drifts_plot())
        # figs.append(self.moments_plot())
        figs.append(self.floor_accels_plot())
        figs.append(self.floor_accels_plot_in_g())
        figs.append(self.normalized_floor_accels_plot())
        return [dcc.Graph(figure=f) for f in figs]

    def view_peak_drifts(self) -> DataFrame:
        abs_drifts = self.view_drifts().abs()
        return abs_drifts.max()

    def view_peak_floor_vels(self) -> DataFrame:
        vels = self.view_floor_vels_envelope()
        return vels.iloc[-1]

    def view_absolute_floor_accels(self) -> DataFrame:
        df = self.view_floor_accels()
        # sum column 0 to all other columns... ugh how to do this better?
        for col in df.columns:
            if col != 0:
                df[col] = df[col] + df[0]
        return df

    def view_normalized_absolute_floor_accels(self) -> DataFrame:
        df = self.view_absolute_floor_accels()
        normalized_df = df / (self.scale * self.record.pfa)
        return normalized_df

    def view_peak_absolute_floor_accels(self) -> DataFrame:
        df = self.view_absolute_floor_accels()
        # take the max abs out of each one
        envelopes = df.abs().values.max(axis=0)
        return envelopes.tolist()

    def view_peak_absolute_floor_accelerations_in_g(self) -> DataFrame:
        df = self.view_absolute_floor_accels()
        envs = df.abs().values.max(axis=0)
        envs = envs / GRAVITY
        return envs.tolist()

    def view_timehistory_summary(self) -> dict:
        accels = self.view_peak_absolute_floor_accelerations_in_g()
        results = {
            "path": self.abs_folder,  # DONT REMOVE. frontend relies on this to load results.
            SummaryEDP.peak_drifts.value: self.view_peak_drifts().to_list(),
            SummaryEDP.peak_floor_accels.value: accels,
            SummaryEDP.peak_floor_vels.value: self.view_peak_floor_vels().to_list(),
            "accel": accels,
        }
        return results

    def get_and_set_timehistory_summary(self) -> dict:
        results = self.view_timehistory_summary()
        self.peak_floor_accels = results[SummaryEDP.peak_floor_accels.value]
        self.peak_drifts = results[SummaryEDP.peak_drifts.value]
        self.peak_floor_vels = results[SummaryEDP.peak_floor_vels.value]
        return results


@dataclass
class Recorder:
    path: Path
    fem: "FiniteElementModel"
    model_str: str = None
    view: StructuralResultView = None

    def __post_init__(self) -> None:
        os.makedirs(self.abspath, exist_ok=True)
        self.view = StructuralResultView(self.abspath)

    @property
    def abspath(self) -> str:
        return str(self.path.resolve())

    @property
    def tcl_string(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.fem) + self.recorders

    """
    improvements:
    it might be better to register a recorder for every string appended
    so when I load strana again.. i have access to what it supposedly saved to db
    and don't have to write paths manually
    """

    @property
    def node_recorders(self) -> str:
        """TODO; add -time to .envelope to see the pseudotime that the min/max happened in"""
        # nodeIDs = ""  # per OpenSees docs, it defaults to all nodes in model.
        s = f"set abspath {self.abspath}\n"
        for edp in OPENSEES_EDPs:
            s += f"recorder Node            -file $abspath/node-{edp}.csv -time -dof 1 2 3 {edp}\n"
            s += f"recorder Node            -file $abspath/mass-{edp}.csv -time -node {self.fem.massIDs_str} -dof 1 {edp}\n"
            s += f"recorder NodeEnvelope    -file $abspath/node-{edp}-envelope.csv -dof 1 2 3 {edp}\n"
            s += f"recorder NodeEnvelope    -file $abspath/mass-{edp}-envelope.csv -node {self.fem.massIDs_str} -dof 1 {edp}\n"
            # -time will prepend a column with the step that node_i achieved envelope at dof_j

        fixed_nodes = self.fem.fixedIDs_str
        for reaction, dof in OPENSEES_REACTION_EDPs:
            s += f"recorder Node            -file $abspath/{reaction}.csv -time -node {fixed_nodes} -dof {dof} reaction\n"
            s += f"recorder NodeEnvelope    -file $abspath/{reaction}-envelope.csv -node {fixed_nodes} -dof {dof} reaction\n"

        s += f"recorder Node            -file $abspath/roof-displacements.csv -time -node {self.fem.roofID} -dof 1 disp\n"
        s += f"recorder NodeEnvelope            -file $abspath/roof-displacements-env.csv -time -node {self.fem.roofID} -dof 1 disp\n"
        s += f"recorder Node            -file $abspath/roof-accels.csv -time -node {self.fem.roofID} -dof 1 accel\n"
        s += f"recorder NodeEnvelope            -file $abspath/roof-accels-env.csv -time -node {self.fem.roofID} -dof 1 accel\n"

        storey_node_ids = self.fem.mass_nodes
        iNodes = "0 " + Node.string_ids_for_list(
            storey_node_ids[:-1]
        )  # 1st and next to last storeys
        jNodes = Node.string_ids_for_list(storey_node_ids)
        s += f"recorder Drift           -file $abspath/drifts.csv -time -iNode {iNodes} -jNode {jNodes} -dof 1 -perpDirn 2\n"
        roofID = self.fem.roofID
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
                ("columns", self.fem.columnIDs_str),
                ("beams", self.fem.beamIDs_str),
            ]:
                s += f"recorder Element         -file $abspath/{ele_type}.csv -ele {ids} -time {edp} \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope.csv -ele {ids} {edp} \n"
        for edp in OPENSEES_ELEMENT_EDPs:
            for ele_type, elems_by_st in [
                ("columns", self.fem.columnIDs_by_storey),
                ("beams", self.fem.beamIDs_by_storey),
            ]:
                for st, elems in enumerate(elems_by_st, 1):
                    ids = " ".join([str(id) for id in elems])
                    s += f"recorder Element         -file $abspath/{ele_type}-{st}.csv -ele {ids} -time {edp} \n"
                    s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope.csv -ele {ids} {edp} \n"

        return s

    @property
    def recorders(self) -> str:
        s = self.node_recorders
        s += self.element_recorders
        return s

    @property
    def elastic_static_solvers(self) -> str:
        string = "constraints Transformation \n"
        string += "numberer RCM \n"
        string += "system BandGeneral \n"
        string += "test NormDispIncr 1.0e-05 25 5 \n"
        string += "algorithm Newton \n"
        string += "integrator LoadControl 0.1 1 \n"
        string += "analysis Static \n"
        string += "initialize \n"
        string += "analyze 10 \n"
        string += "remove recorders \n"
        string += "loadConst -time 0.0 \n"
        return string


@dataclass
class GravityRecorderMixin(Recorder):
    def __str__(self) -> str:
        s = super().__str__()
        s += self.gravity_str
        return s

    @property
    def gravity_str(self) -> str:
        analysis_str = "pattern Plain 1 Linear {\n"
        for beams, beam_load in zip(
            self.fem.beams_by_storey, self.fem.uniform_beam_loads
        ):
            beam_ids = ElasticBeamColumn.string_ids_for_list(beams)
            analysis_str += (
                f"eleLoad -ele {beam_ids} -type beamUniform {-beam_load:.1f} \n"
            )
        analysis_str += "}\n"
        analysis_str += self.elastic_static_solvers
        return analysis_str


@dataclass
class StaticRecorder(GravityRecorderMixin):
    forces_per_storey: list[float] = None

    # deformation.. is this curvature?
    # recorder Element -file ${gravity_results}/col_moments_th.out -time -ele 11 plasticDeformation

    def __str__(self) -> str:
        s = str(self.fem) + self.gravity_str
        s += self.static_str + self.recorders
        s += self.elastic_static_solvers
        return s

    @property
    def static_str(self) -> str:
        from itertools import zip_longest

        analysis_str = "pattern Plain 2 Linear {\n"
        for node, fx in zip_longest(
            self.fem.mass_nodes,
            self.forces_per_storey,
            fillvalue=self.forces_per_storey[0],
        ):
            analysis_str += f"load {node.id} {fx} 0.0 0.0\n"
        analysis_str += "}\n"
        return analysis_str


@dataclass
class PushoverRecorder(GravityRecorderMixin):
    drift: float = 0.05
    steps: int = 2000
    tol: str = 1.0e-5

    def __str__(self) -> str:
        s = str(self.fem)
        s += self.gravity_str
        s += self.pushover_str + self.recorders
        s += self.extra_recorders
        s += self.pushover_solvers
        return s

    @property
    def element_recorders(self) -> str:
        s = super().element_recorders
        for ele_type, ids in [
            ("columns", self.fem.columnIDs_str),
            ("beams", self.fem.beamIDs_str),
        ]:
            s += f"recorder Element         -file $abspath/{ele_type}-a.csv -ele {ids} -time section 1 force \n"
            s += f"recorder Element         -file $abspath/{ele_type}-b.csv -ele {ids} -time section 5 force \n"
            s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope-a.csv -ele {ids} section 1 force \n"
            s += f"recorder EnvelopeElement -file $abspath/{ele_type}-envelope-b.csv -ele {ids} section 5 force \n"
        for ele_type, elems_by_st in [
            ("columns", self.fem.columnIDs_by_storey),
            ("beams", self.fem.beamIDs_by_storey),
        ]:
            for st, elems in enumerate(elems_by_st, 1):
                ids = " ".join([str(id) for id in elems])
                s += f"recorder Element         -file $abspath/{ele_type}-{st}-a.csv -ele {ids} -time section 1 force \n"
                s += f"recorder Element         -file $abspath/{ele_type}-{st}-b.csv -ele {ids} -time section 5 force \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope-a.csv -ele {ids} section 1 force \n"
                s += f"recorder EnvelopeElement -file $abspath/{ele_type}-{st}-envelope-b.csv -ele {ids} section 5 force \n"

        return s

    @property
    def pushover_str(self) -> str:
        analysis_str = "pattern Plain 2 Linear {\n"
        for load, node in enumerate(self.fem.mass_nodes, 1):
            analysis_str += f"load {node.id} -{load} 0.0 0.0\n"
        analysis_str += "}\n"
        return analysis_str

    @property
    def extra_recorders(self) -> str:
        fixed_nodes = self.fem.fixedIDs_str
        s = ""
        s += f"recorder NodeEnvelope -file $abspath/base-shear-env.csv -node {fixed_nodes} -dof 1 reaction\n"
        s += f"recorder Node -file $abspath/base-shear.csv -time -node {fixed_nodes} -dof 1 reaction\n"
        return s

    @property
    def pushover_solvers(self) -> str:
        maxU = self.drift * self.fem.height
        dU = maxU / self.steps
        tol = self.tol * np.sqrt(len(self.fem.nodes))
        string = "constraints Transformation \n"
        string += "numberer RCM \n"
        string += "system BandGeneral \n"
        string += f"test NormDispIncr {self.tol} 100\n"
        string += "algorithm NewtonLineSearch \n"
        string += f"integrator DisplacementControl {self.fem.roofID} 1 {dU:.10f}\n"
        string += "analysis Static \n"
        string += f"set maxU {maxU}\n"
        string += "set disp 0.0\n"
        string += "set ok 0\n"
        string += """
while {$ok == 0 && $disp < $maxU} {
    set ok [analyze 1]
    set fail 0
    while {$ok != 0} {
        incr fail
        test NormDispIncr %s 1000
        set ok [analyze 1]
        test NormDispIncr %s 100
        if {$fail > 1} {
            puts "pushover failed"
            return 0
        }
    }
    set disp [nodeDisp %d 1]
}
puts "pushover successful"
        """ % (
            tol,
            tol,
            self.fem.roofID,
        )
        return string


@dataclass
class KRecorder(Recorder):
    def __str__(self) -> str:
        s = super().__str__()
        s += self.stiffness_matrix_solvers
        return s

    def view_stiffness_matrix(self):
        K = pd.read_csv(self.path / self.view._K_STATIC_NAME, sep="\s+", header=None)
        return K

    @property
    def stiffness_matrix_solvers(self) -> str:
        string = "constraints Transformation \n"
        string += "numberer Plain \n"
        string += "system FullGeneral \n"
        string += "test NormDispIncr 1.0e-05 25 5 \n"
        string += "algorithm Newton \n"
        string += "integrator LoadControl 0.1 1 \n"
        string += "analysis Static \n"
        string += "initialize \n"
        string += "analyze 1 \n"
        string += f"printA -file {self.path / self.view._K_STATIC_NAME}\n"
        string += "analyze 9 \n"
        string += "remove recorders \n"
        string += "loadConst -time 0.0 \n"
        # string += "wipeAnalysis \n"
        return string


@dataclass
class EigenRecorder(Recorder):
    _cache: dict = None

    def __str__(self) -> str:
        """add -time to envelope to get"""
        s = str(self.fem)
        s += (
            # f"set eigenvalues [eigen -generalized -fullGenLapack {self.fem.modes}]\n"
            # set eigenvalues [eigen -generalized -genBandArpack 5]
            f"set eigenvalues [eigen {self.fem.num_modes}]\n"
        )
        eigen_values_path = (self.path / "eigen_values.csv").resolve()
        eigen_vectors_path = (self.path / "eigen_vectors.csv").resolve()
        s += f'set eigen_values_file [open {eigen_values_path} "w"]\n'
        s += "puts $eigen_values_file $eigenvalues \n"
        s += "close $eigen_values_file \n"
        s += f'set eigen_vectors_file [open {eigen_vectors_path} "w"]\n'
        modes = " ".join([str(i) for i in range(1, self.fem.num_modes + 1)])
        massIDS = " ".join([str(i) for i in self.fem.massIDs])
        s += "set storeys {%s}\n" % modes
        s += "set massNodes {%s}\n" % massIDS
        s += "foreach mode $storeys {\n"
        s += "  foreach m $massNodes {\n"
        s += (
            "      lappend eigenvector($mode) [lindex [nodeEigenvector $m $mode 1] 0]\n"
        )
        s += "  }\n"
        s += "  puts $eigen_vectors_file $eigenvector($mode)\n"
        s += "}\n"
        return s


@dataclass
class TimehistoryRecorder(GravityRecorderMixin):
    record: Record = None
    a0: float = None
    a1: float = None
    scale: float = None
    gravity_loads: bool = True
    EXTRA_FREE_VIBRATION_SECONDS: float = 10.0

    def __post_init__(self):
        os.makedirs(self.abspath, exist_ok=True)
        self.view = StructuralResultView(
            self.abspath, record=self.record, scale=self.scale
        )

    def __str__(self) -> str:
        """https://portwooddigital.com/2021/02/28/norms-and-tolerance/"""
        s = str(self.fem)
        num_nodes = len(self.fem.nodes)
        tol = 1e-5 * num_nodes
        if self.gravity_loads:
            s += self.gravity_str
        s += self.recorders
        s += f'set timeSeries "Series -dt {self.record.dt} -filePath {self.record.abspath} -factor {self.scale}"\n'
        s += "pattern UniformExcitation 2 1 -accel $timeSeries\n"
        # todo figure out correct a0, a1 placement for NONLIN analyses. where Ke changes.
        # s += f"rayleigh {self.a0} {self.a1} 0 0\n"
        s += f"rayleigh {self.a0} 0 {self.a1} 0\n"
        # s += f"rayleigh {self.a0} 0 0 {self.a1}\n"
        s += self.elastic_dynamic_solvers
        # s += f"analyze {self.record.steps} {self.record.dt}\n"
        s += """
set converged 0
set time [getTime]
set duration %s
while {$converged == 0 && $time <= $duration} {
    set tol %s
    test NormUnbalance $tol 100
    set time [getTime]
    set converged [analyze 1 0.01]
    set loops 0
    while {$converged !=0} {
        incr loops
        set tol %s
        test NormUnbalance $tol 100
        set converged [analyze 10 0.001]
        if {$loops > 10} {
            puts "Analysis did not converge"
            break
        }
    }
}
        """ % (
            self.record.duration + self.EXTRA_FREE_VIBRATION_SECONDS,
            tol,
            tol,
        )
        return s

    @property
    def elastic_dynamic_solvers(
        self,
    ) -> str:
        """TODO; set error to 1.0e-5 * sqrt(DOFS)"""
        s = "constraints Transformation\n"
        s += "numberer RCM\n"
        s += "test NormUnbalance 1.0e-5 100\n"  # NormDispIncr if our elements are very stiff
        s += "algorithm KrylovNewton\n"
        s += "integrator Newmark 0.5 0.25\n"
        s += "system BandGeneral\n"
        s += "analysis Transient\n"
        return s


@dataclass
class StructuralAnalysis:
    results_path: Path
    fem: FiniteElementModel

    def __post_init__(self):
        self.K_path = self.results_path / AnalysisTypes.K.value
        self.modal_path = self.results_path / AnalysisTypes.MODAL.value
        self.gravity_path = self.results_path / AnalysisTypes.GRAVITY.value
        self.static_path = self.results_path / AnalysisTypes.STATIC.value
        self.pushover_path = self.results_path / AnalysisTypes.PUSHOVER.value
        self.timehistory_path = self.results_path / AnalysisTypes.TIMEHISTORY.value

        self.K_recorder = KRecorder(self.K_path, fem=self.fem)
        self.modal_recorder = EigenRecorder(self.modal_path, fem=self.fem)

    def run(self, recorder: Recorder) -> StructuralResultView:
        results_path = recorder.path / "model.tcl"
        with open(results_path, "w") as f:
            f.write(recorder.tcl_string)
        os.chmod(results_path, 0o777)
        result = subprocess.call(str(results_path.resolve()), shell=True)
        # print(f"run result {result}")
        return recorder.view

    def get_stiffness_matrix(self) -> np.ndarray:
        _ = self.run(recorder=self.K_recorder)
        return self.K_recorder.view_stiffness_matrix()

    @property
    def Ke(self) -> np.ndarray:
        return self.get_stiffness_matrix()

    @property
    def K_static(self) -> np.ndarray:
        """
        statically condensed matrix.
        WARNING; this will not work in general, if 'mass_dofs' is incorrectly defined.
        tries to condense Ke statically using info in FEM
        ks = ktt - kto * koo^-1 * kot
        """
        from numpy.linalg import inv

        Ke = self.Ke
        ixs = self.fem.mass_dofs
        ktt = Ke.loc[ixs, ixs]
        free_dofs = self.fem.free_dofs
        rest = set(free_dofs) - set(ixs)
        koo = Ke.loc[rest, rest]
        kto = Ke.loc[ixs, rest]
        kot = Ke.loc[rest, ixs]
        ktt = ktt.to_numpy()
        koo = koo.to_numpy()
        kto = kto.to_numpy()
        kot = kot.to_numpy()
        Ks = ktt - (kto @ inv(koo)) @ kot
        return Ks

    @property
    def Ks(self) -> np.ndarray:
        return self.K_static

    def modal(self, *args, **kwargs) -> StructuralResultView:
        """
        -fullGenLapack does NOT return normalized by mass vectors.. which is a shame.
        by default opensees does '-generalized -genbandArpack' with returns vecs normalized by mass
        however it can only process n-1 dofs.
        so one must add fictitious masses to other dofs to get around this limitation
        e.g. -mass 1000 1e-9 0.
        this makes things more convoluted.
        """
        # TODO; use a better combination.. the one that minimizes weighted error.
        view = self.run(recorder=self.modal_recorder)
        view.get_and_set_modal_results()
        return view

    def standalone_gravity(self) -> StructuralResultView:
        recorder = GravityRecorderMixin(self.gravity_path, fem=self.fem)
        return self.run(recorder=recorder)

    def static(self, forces_per_storey: list[float]) -> StructuralResultView:
        recorder = StaticRecorder(
            self.static_path, fem=self.fem, forces_per_storey=forces_per_storey
        )
        return self.run(recorder=recorder)

    def pushover(self, drift: float = 0.05) -> Recorder:
        recorder = PushoverRecorder(self.pushover_path, fem=self.fem, drift=drift)
        return self.run(recorder=recorder)

    def timehistory(
        self,
        record: Record,
        scale=None,
        a0: float = None,
        a1: float = None,
        gravity_loads=True,
    ) -> "StructuralResultView":
        if a0 is None or a1 is None:
            self.fem.get_and_set_eigen_results(self.modal_path)
            a0, a1 = self.fem.a0, self.fem.a1
        if not scale:
            scale = record.scale

        recorder = TimehistoryRecorder(
            path=self.timehistory_path,
            fem=self.fem,
            record=record,
            a0=a0,
            a1=a1,
            scale=scale,
            gravity_loads=gravity_loads,
        )
        view = self.run(recorder=recorder)
        return view


class HazardNotFoundException(FileNotFoundError):
    pass


class SpecNotFoundException(FileNotFoundError):
    pass


@dataclass
class IDA(NamedYamlMixin):
    name: str
    hazard_abspath: str
    design_abspath: str
    start: float = 0.1
    stop: float = 1.0
    step: float = 0.1
    results: dict = None
    standard: bool = False

    _hazard: Hazard = None
    _design = None
    _intensities: np.ndarray = None

    _COLLAPSE_DRIFT: float = 0.2

    def __post_init__(self):
        from app.design import ReinforcedConcreteFrame

        try:
            if self.hazard_abspath and not self._hazard:
                self._hazard: Hazard = Hazard.from_file(self.hazard_abspath)
        except FileNotFoundError as e:
            raise HazardNotFoundException
        try:
            if self.design_abspath and not self._design:
                self._design = ReinforcedConcreteFrame.from_file(self.design_abspath)
        except FileNotFoundError as e:
            raise SpecNotFoundException

        if self.standard:
            self._intensities = self._hazard.intensities_for_idas()
        else:
            linspace = np.arange(self.start, self.stop + self.step / 2, self.step)
            self._intensities = (
                linspace,
                linspace + self.step / 2,
                linspace - self.step / 2,
            )

    def run(
        self,
        results_dir: Path = None,
        run_id: str = None,
        fem_ix: int = -1,
        period_ix: int = 0,
    ) -> IDAResultsDataFrame:
        if not run_id:
            run_id = str(uuid())

        fem = self._design.fems[fem_ix]
        modal_view = fem.get_and_set_eigen_results(results_dir)
        period = modal_view.periods[period_ix]

        dataframe_records = []
        num_records = len(self._hazard.records)
        num_intensities = len(self._intensities[0])
        counter = 0
        for rix, record in enumerate(self._hazard.records, start=1):
            collapse = False
            for iix, (intensity, sup, inf) in enumerate(
                zip(*self._intensities), start=1
            ):
                counter += 1
                complete_pct = 100 * counter / (num_records * num_intensities)
                print(
                    f"record {rix} of {num_records} at intensity {iix} of {num_intensities} ({intensity:.2f}g) -- {complete_pct:.0f} % done."
                )
                intensity_str_precision = f"{intensity:.8f}"
                outdir = results_dir / run_id / record.name / intensity_str_precision
                results_to_meters = 1.0 / 100
                scale_factor = results_to_meters * record.get_scale_factor(
                    period=period, intensity=intensity
                )
                rate_inf, rate_sup = self._hazard._curve.interpolate_rate_for_values(
                    [inf, sup]
                )
                freq = rate_inf - rate_sup
                # if collapse:
                #     dataframe_records.append(row)
                #     continue
                strana = StructuralAnalysis(outdir, fem=fem)
                th_view = strana.timehistory(record=record, scale=scale_factor)
                results = th_view.get_and_set_timehistory_summary()
                # frequency of exceedance of bin i.
                collapse = self._design.fem.determine_collapse_from_results(results)
                row = {
                    "record": record.name,
                    "intensity_str": intensity_str_precision,
                    "intensity": intensity,
                    "sup": sup,
                    "inf": inf,
                    "freq": freq,
                    "collapse": collapse,
                    **results,
                }
                dataframe_records.append(row)
                th_view.to_file()

        results_df = pd.DataFrame.from_records(dataframe_records)
        results_df.to_csv(STRANA_DIR / f"{run_id}.csv")
        self.results = results_df.to_dict(orient="records")
        return results_df

    def view_ida_curves(self) -> Figure:
        """
        classic Vamvatsikos ida curves
        peak_drift in any storey vs Sa(g)(T_1, z_n%)
        """
        import plotly.express as px

        df = pd.DataFrame.from_records(self.results)
        # df2 = df.pivot(
        #     index="intensity", columns="record", values=SummaryEDP.peak_drifts.value
        # )

        def select_peak_drift(drifts_by_storey):
            if np.any(np.isnan(drifts_by_storey)):
                return np.nan
            return max(drifts_by_storey)

        df["peak_drifts"] = 100 * df[SummaryEDP.peak_drifts.value].apply(
            select_peak_drift
        )
        df["symbol"] = df["collapse"].apply(lambda c: "x" if c else "circle")
        fig = px.line(
            df,
            y="intensity",
            x="peak_drifts",
            color="record",
            symbol="symbol",
            symbol_sequence=["circle", "x"],
            markers=True,
        )
        # seems like figures are 100% of parent container
        fig.update_layout(
            yaxis_title="accel (g)",
            xaxis_title="peak drift any storey (%)",
            title="IDA curves",
            # marginal_x="histogram", marginal_y="rug",
            # yaxis_range=[0.0, 0.3],
            # width=1100,
            # height=600,
            # autosize=True,
            # responsive=True,
        )
        return fig


@dataclass
class RSA(StructuralAnalysis):
    code: BuildingCode

    def get_design_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """nxn where each column is f_n = s_i a_i"""
        view = self.fem.get_and_set_eigen_results(self.results_path)
        S = view.S
        pseudo_accels = self.code.get_Sas(self.fem.periods)
        As = np.array(pseudo_accels)
        n = len(As)
        As = np.eye(n) * As
        F = S.dot(As)  # s_i a_i
        cs = pseudo_accels[0] / GRAVITY
        return F, view.V, cs

    def srss(self) -> tuple[list, list, list]:
        moments = []
        forces, shears, cs = self.get_design_forces()
        for f in forces:
            recorder = self.static(f)
            moments.append(recorder.view_column_design_moments())
        M = np.array(moments).T
        peak_moments = np.sqrt(np.sum(M ** 2, axis=1))
        peak_shears = np.sqrt(np.sum(shears ** 2, axis=1))
        return peak_moments.tolist(), peak_shears.tolist(), float(cs)