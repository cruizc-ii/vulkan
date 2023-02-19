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
from pathlib import Path
from app.utils import GRAVITY, SummaryEDP, YamlMixin, ASSETS_PATH, RESULTS_DIR
from app.hazard import Hazard
from app.strana import HazardNotFoundException
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line, Pie
from app.concrete import RectangularConcreteColumn
from app.design import ReinforcedConcreteFrame, FiniteElementModel
from app.strana import IDA, STRANA_DIR
from shortuuid import uuid


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

    @property
    def pushover_figs(self):
        fig = Figure()
        for comp in self.comparisons:
            fig.add_trace(comp.pushover_trace)
        fig.update_layout(
            yaxis_title="Vy [kN]",
            xaxis_title="u [m]",
            title="Pushover curves",
            # autosize=True,
            # responsive=True,
        )
        return fig

    @property
    def normalized_pushover_figs(self):
        fig = Figure()
        for comp in self.comparisons:
            fig.add_trace(comp.normalized_pushover_trace)
        fig.update_layout(
            yaxis_title="cs=Vy/W [1]",
            xaxis_title="y=u/H [1]",
            title="Normalized pushover curves",
            # autosize=True,
            # responsive=True,
        )
        return fig

    @property
    def ida_figs(self):
        fig = Figure()
        for comp in self.comparisons:
            fig.add_trace(comp.ida_trace)
        fig.update_layout(
            yaxis_title="Sa [g]",
            xaxis_title="drift [1]",
            title="IDA curves",
            # autosize=True,
            # responsive=True,
        )
        return fig

    @property
    def normalized_ida_figs(self):
        fig = Figure()
        for comp in self.comparisons:
            fig.add_trace(comp.normalized_ida_trace)
        fig.update_layout(
            yaxis_title="Sa/Sa_design [1]",
            xaxis_title="drift [1]",
            title="Normalized ida curves",
            # autosize=True,
            # responsive=True,
        )
        return fig

    @property
    def summary_df(self) -> list[dict]:
        df = pd.DataFrame(self.summary)
        df.style.format({"E": "{:.2f}"})
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

    @property
    def pushover_trace(self) -> Figure:
        x, y = self.summary.get("_pushover_x", []), self.summary.get("_pushover_y", [])
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def normalized_pushover_trace(self) -> Figure:
        x, y = self.summary.get("_norm_pushover_x", []), self.summary.get(
            "_norm_pushover_y", []
        )
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def ida_trace(self) -> Figure:
        x, y = self.summary.get("_ida_x", []), self.summary.get("_ida_y", [])
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
        return trace

    @property
    def normalized_ida_trace(self) -> Figure:
        x, y = self.summary.get("_norm_ida_x", []), self.summary.get("_norm_ida_y", [])
        name = self.summary.get("design name", "")
        trace = Scattergl(x=x, y=y, name=name)
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
            self._design_model.fem.summary
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
        if self.design_abspath:
            try:
                self._design_model = ReinforcedConcreteFrame.from_file(
                    self.design_abspath
                )
            except FileNotFoundError as e:
                print("err loading design path", e)
                raise e
        else:
            return False

        name = name if name else str(uuid())
        ida_name = f"{name}-{self._design_model.name}-{self._hazard_model.name}"
        strana_abspath = str((STRANA_DIR / f"{ida_name}.yml").resolve())
        if strana:
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

        if loss and self._strana_model is not None:
            loss_name = ida_name
            agg = LossAggregator(name=loss_name, ida_model_path=strana_abspath)
            loss_abspath = str((LOSS_DIR / f"{ida_name}.yml").resolve())
            self._loss_model = agg
            agg.to_file(LOSS_DIR)
            self.loss_abspath = loss_abspath

        self.summary = self.get_summary()
        return True
