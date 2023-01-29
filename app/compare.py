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
    NamedYamlMixin,
)
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
from app.utils import GRAVITY, SummaryEDP, YamlMixin, ASSETS_PATH
from app.hazard import Hazard
from app.strana import HazardNotFoundException
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line, Pie
from app.concrete import RectangularConcreteColumn
from app.design import ReinforcedConcreteFrame, FiniteElementModel
from app.strana import IDA


@dataclass
class CompareInterface(ABC, NamedYamlMixin):
    # name: str | None = None
    name: str
    hazard_abspath: str | None = None
    design_abspaths: list[str] = field(default_factory=list)
    strana_abspaths: list[str | None] = field(default_factory=list)
    _design_models: list[ReinforcedConcreteFrame] = field(default_factory=list)
    _strana_models: list[IDA] = field(default_factory=list)
    _hazard: Hazard | None = None

    @abstractmethod
    def run(self):
        # must setup comparison objects such as graphs, figs,
        pass


class IDACompare(CompareInterface):
    def __post_init__(self):
        try:
            if self.hazard_abspath is not None and not self._hazard:
                self._hazard: Hazard = Hazard.from_file(self.hazard_abspath)
        except FileNotFoundError:
            raise HazardNotFoundException
        # try:
        #     if self.design_abspath is not None and not self._design:
        #         self._design = ReinforcedConcreteFrame.from_file(self.design_abspath)
        # except FileNotFoundError:
        #     raise SpecNotFoundException
        # if len(self.records) > 0 and isinstance(self.records[0], dict):
        #     self.records = [Record(**data) for data in self.records]
        # if isinstance(self.curve, dict):
        #     self._curve = HazardCurveFactory(**self.curve)
        # if isinstance(self.curve, str):
        #     self._curve = HazardCurveFactory(name=self.curve)

    def add_design(self, design_abspath: str) -> bool:
        from app.design import ReinforcedConcreteFrame

        if design_abspath in self.design_abspaths:
            return False
        try:
            design = ReinforcedConcreteFrame.from_file(design_abspath)
        except FileNotFoundError as e:
            print(e)
            raise e
        self._design_models.append(design)
        return True

    def remove_design(self, design_abspath: str) -> bool:
        if design_abspath not in self.design_abspaths:
            return False
        # self.records = [r for r in self.records if r.path != record_path]
        return True

    def run(self):
        return
