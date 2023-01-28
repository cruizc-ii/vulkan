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
from app.design import ReinforcedConcreteFrame


@dataclass
class CompareInterface(ABC, NamedYamlMixin):
    # name: str | None = None
    name: str
    hazard_abspath: str | None = None
    design_abspaths: list[str] = field(default_factory=list)
    strana_abspaths: list[str | None] = field(default_factory=list)
    # _design_models: list[FiniteElementModel]
    # _strana_models: list[]
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

    def run(self):
        return
