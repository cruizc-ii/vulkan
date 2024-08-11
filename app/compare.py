from __future__ import annotations
from abc import ABC, abstractmethod

from app.assets import (
    AssetFactory,
    RiskAsset,
    RiskModelFactory,
    Asset,
)
from app.loss import LOSS_DIR, LossAggregator
from app.utils import (
    OPENSEES_ELEMENT_EDPs,
    OPENSEES_EDPs,
    OPENSEES_REACTION_EDPs,
    NamedYamlMixin,
)
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from app.utils import GRAVITY, SummaryEDP, YamlMixin, ASSETS_PATH, RESULTS_DIR
from app.hazard import Hazard
from app.strana import HazardNotFoundException
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line, Pie
from app.concrete import RectangularConcreteColumn
from app.design import ReinforcedConcreteFrame, FiniteElementModel
from app.strana import IDA, STRANA_DIR
from human_id import generate_id as uuid

colors = px.colors.qualitative.Bold
colors = px.colors.sequential.Greys_r
from plotly.colors import n_colors

colors = n_colors("rgb(0, 0, 0)", "rgb(200, 200, 200)", 2, colortype="rgb")


@dataclass
class CompareInterface(ABC, NamedYamlMixin):
    name: str
    hazard_abspath: str | None = None
    _hazard: Hazard | None = None
    comparisons: list[DesignComparison] = field(default_factory=list)

    @abstractmethod
    def run(self):
        # must setup summarization entities, run strana then loss
        pass


@dataclass
class IDACompare(CompareInterface):
    def __post_init__(self):
        try:
            if self.hazard_abspath is not None and not self._hazard:
                self._hazard: Hazard = Hazard.from_file(self.hazard_abspath)
        except FileNotFoundError:
            raise HazardNotFoundException

        self.comparisons = [
            DesignComparison(
                **{**comparison_dict, "hazard_abspath": self.hazard_abspath}
            )
            for comparison_dict in self.comparisons
        ]

    @staticmethod
    def prettify_fig(fig: Figure):
        fig.update_layout(plot_bgcolor="white")
        fig.update_yaxes(
            showgrid=True,
            ticks="inside",
            mirror=True,
            gridwidth=1,
            showline=True,
            linecolor="black",
            zerolinewidth=0.0,
            linewidth=1,
            zerolinecolor="lightgray",
        )
        fig.update_xaxes(
            showgrid=True,
            ticks="inside",
            mirror=True,
            gridwidth=1,
            linecolor="black",
            zerolinewidth=0.0,
            linewidth=1,
            zerolinecolor="lightgray",
        )
        return fig

    @property
    def pushover_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.pushover_trace
            trace.marker.color = color
            fig.add_trace(trace)
            Vy, uy = comp.summary.get("Vy [kN]", 0), comp.summary.get("uy [m]", 0)
            fig.add_trace(
                Scattergl(
                    x=[uy],
                    y=[Vy],
                    marker_symbol="circle",
                    showlegend=False,
                    marker_size=8,
                    marker=dict(color=color),
                )
            )
        fig.update_layout(
            yaxis_title="V [kN]",
            xaxis_title="u [m]",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def normalized_pushover_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.normalized_pushover_trace
            trace.marker.color = color
            fig.add_trace(trace)
            Vy, drift_y = comp.summary.get("Vy [kN]", 0), comp.summary.get(
                "drift_y [1]", 0
            )
            W = float(comp.summary.get("weight", 1))
            fig.add_trace(
                Scattergl(
                    x=[drift_y],
                    y=[Vy / W],
                    marker_symbol="circle",
                    showlegend=False,
                    marker_size=8,
                    marker=dict(color=color),
                )
            )
        fig.update_layout(
            yaxis_title="cs=V/W [1]",
            xaxis_title="y=u/H [1]",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def ida_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.ida_trace
            trace.marker.color = color
            fig.add_trace(trace)
        fig.update_layout(
            yaxis_title="Sa [g]",
            xaxis_title="drift [1]",
            xaxis_range=[0, 0.10],
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def normalized_ida_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.normalized_ida_trace
            trace.marker.color = color
            fig.add_trace(trace)
        fig.update_layout(
            yaxis_title="Sa/Sa_design [1]",
            xaxis_title="drift/drift_yield [1]",
            xaxis_range=[0.0, 10.0],
            yaxis_range=[0.0, 8.0],
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def rate_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.rate_trace
            trace.marker.color = color
            fig.add_trace(trace)
        fig.update_layout(
            yaxis_title="v($) 1/yr",
            xaxis_title="Loss $",
            # xaxis_type="log",
            # xaxis_range=[-2, 3],
            yaxis_type="log",
            yaxis_range=[-4, 0],
            # autosize=True,
            # responsive=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def risk_figs(self):
        fig = Figure()
        for comp, color in zip(self.comparisons, colors):
            trace = comp.risk_trace
            trace.marker.color = color
            fig.add_trace(trace)
        fig.update_layout(
            yaxis_title="Cost $",
            xaxis_title="discount factor [1]",
            # title="Risk",
            # xaxis_type="log",
            # yaxis_type="log",
            # xaxis_range=[-2, 3],
            # yaxis_range=[-5, 0],
            # autosize=True,
            # responsive=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7),
        )
        fig = self.prettify_fig(fig)
        return fig

    @property
    def summary_df(self) -> list[dict]:
        df = pd.DataFrame(self.summary)
        df.style.format({"E": "{:.1f}"})
        return df

    @property
    def summary(self) -> list[dict]:
        return [c.summary for c in self.comparisons]

    @property
    def design_abspaths(self):
        return [
            c.design_abspath for c in self.comparisons if c.design_abspath is not None
        ]

    def add_design(self, design_abspath: str) -> bool:
        if design_abspath in self.design_abspaths:
            return False
        try:
            comparison = DesignComparison(design_abspath=design_abspath)
        except HazardNotFoundException as e:
            print(e)
            raise e
        self.comparisons.append(comparison)
        return True

    def remove_design(self, design_abspath: str) -> bool:
        if design_abspath not in self.design_abspaths:
            return False
        dix = self.design_abspaths.index(design_abspath)
        self.comparisons = [d for ix, d in enumerate(self.comparisons) if ix != dix]
        return True

    def run(
        self,
        strana=False,
        loss=False,
    ) -> bool:
        for comp in self.comparisons:
            comp.run(name=self.name, strana=strana, loss=loss)

        return True


@dataclass
class DesignComparison(YamlMixin):
    summary: dict = field(default_factory=dict)
    hazard_abspath: str | None = None
    _hazard_model: Hazard | None = None
    design_abspath: str | None = None
    _design_model: ReinforcedConcreteFrame | None = None
    strana_abspath: str | None = None
    _strana_model: IDA | None = None
    loss_abspath: str | None = None
    _loss_model: LossAggregator | None = None

    def __post_init__(self):
        try:
            if self.hazard_abspath is not None and not self._hazard_model:
                self._hazard_model: Hazard = Hazard.from_file(self.hazard_abspath)
        except FileNotFoundError:
            raise HazardNotFoundException
        if self.design_abspath:
            try:
                self._design_model = ReinforcedConcreteFrame.from_file(
                    self.design_abspath
                )
            except FileNotFoundError as e:
                print("err loading design path", e)
                raise e

    @property
    def pushover_trace(self) -> Figure:
        x, y = self.summary.get("_pushover_x", []), self.summary.get("_pushover_y", [])
        ductility = self.summary.get("ductility", 0)
        name = self.summary.get(f"design name", "") + f"  µ={ductility:.1f}"
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def normalized_pushover_trace(self) -> Figure:
        x, y = self.summary.get("_norm_pushover_x", []), self.summary.get(
            "_norm_pushover_y", []
        )
        ductility = self.summary.get("ductility", 0)
        name = self.summary.get(f"design name", "") + f"  µ = {ductility:.1f}"
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def ida_trace(self) -> Figure:
        x, y = self.summary.get("_ida_x", []), self.summary.get("_ida_y", [])
        x, y = [0] + x, [0] + y
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def normalized_ida_trace(self) -> Figure:
        x, y = self.summary.get("_norm_ida_x", []), self.summary.get("_norm_ida_y", [])
        x, y = [0] + x, [0] + y
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def rate_trace(self) -> Figure:
        x, y = self.summary.get("_rate_x", []), self.summary.get("_rate_y", [])
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def risk_trace(self) -> Figure:
        xs = np.linspace(0.01, 0.12, 100)
        discount_factor = xs
        initial_cost = self.summary.get("net worth $", 0)
        aal: float = self.summary.get("AAL $", 0) or 0
        name = self.summary.get("design name", "")
        ys = initial_cost + aal / discount_factor
        trace = Scattergl(x=xs, y=ys, name=name)
        return trace

    @property
    def summary_df(self) -> pd.DataFrame:
        # this does not work because some keys are themselves arrays, confusing the constructor
        # df = pd.DataFrame.from_dict(self.summary, orient="columns")
        d = {k: v for k, v in self.summary.items() if not k.startswith("_")}
        df = pd.DataFrame([d])
        df.style.format({"E": "{:.2f}"})
        return df

    def get_summary(self) -> dict:
        design_summary = self._design_model.summary if self._design_model else {}
        fem_summary = (
            self._design_model.fem.pushover_stats()
            if self._design_model and self._design_model.fem
            else {}
        )
        strana_summary = self._strana_model.summary if self._strana_model else {}
        loss_summary = self._loss_model.summary if self._loss_model else {}
        d = {**design_summary, **fem_summary, **strana_summary, **loss_summary}
        return YamlMixin.numpy_dict_factory(d)

    def run(
        self,
        name: str | None = None,
        strana: bool = False,
        loss: bool = False,
        standard: bool = True,
    ):
        name = name if name else str(uuid())
        ida_name = f"{name}-{self._design_model.name}-{self._hazard_model.name}"
        if strana:
            strana_abspath = str((STRANA_DIR / f"{ida_name}.yml").resolve())
            ida = IDA(
                name=ida_name,
                hazard_abspath=self.hazard_abspath,
                design_abspath=self.design_abspath,
                standard=standard,
            )
            ida.run_parallel(results_dir=RESULTS_DIR)
            ida.to_file(STRANA_DIR)
            self._strana_model = ida
            self.strana_abspath = strana_abspath
        else:
            ida = IDA.from_file(self.strana_abspath)
            self._strana_model = ida
            strana_abspath = self.strana_abspath

        if loss and self._strana_model is not None:
            loss_name = ida_name
            agg = LossAggregator(name=loss_name, ida_model_path=strana_abspath)
            loss_abspath = str((LOSS_DIR / f"{ida_name}.yml").resolve())
            self._loss_model = agg
            agg.run()
            agg.to_file(LOSS_DIR)
            self.loss_abspath = loss_abspath

        self.summary = {**self.summary, **self.get_summary()}
        return True
