from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from app.hazard import RECORDS_DIR

DESIGN_SPECTRA_PATH = RECORDS_DIR / "design_spectra"


class BuildingCode(ABC):
    @abstractmethod
    def get_Sa(self, period, **kwargs) -> float:
        pass

    @abstractmethod
    def get_Sas(self, periods: list[float], **kwargs) -> list[float]:
        pass


@dataclass
class CDMXBuildingCode(BuildingCode):
    Q: float = 1
    _SPECTRA_NAME = "EspectrosDis145.txt"

    def __post_init__(self):
        spectra_path = DESIGN_SPECTRA_PATH / self._SPECTRA_NAME
        self.spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
        self.R0 = 2.0 if self.Q >= 3 else 1.75
        self.k1 = 0.8
        self.k2 = 0
        self.R = self.k1 * self.R0 + self.k2
        self.Qprime = self.Q

    def get_ordinate_for_period(self, series, period):
        s = series.reindex([period], method="nearest")
        ordinate = s.values.flatten()[0]
        return ordinate

    def get_series_for_Q(self):
        """
        returns the series that corresponds to Q.
        if Q < 1.0 it returns series/Q.
        """
        df = self.spectra
        try:
            cols = [float(c.strip("ED=()Q=")) for c in df.columns[2:]]
            ix = [ix for ix, c in enumerate(cols) if np.isclose(c, self.Q, 0.049)][0]
            series = df.iloc[:, ix + 2]
        except IndexError:
            # Q < 1.0
            col = "Elástico"
            series = df[col] / self.Qprime

        return series

    def get_Sa(self, period, **kwargs) -> float:
        series = self.get_series_for_Q()
        Sa_gals = self.get_ordinate_for_period(series, period)
        Sa = float(Sa_gals / 100)  # to m/s2
        return Sa

    def get_Sas(self, periods: list[float], **kwargs) -> list[float]:
        series = self.get_series_for_Q()
        Sa_gals = np.array(
            [self.get_ordinate_for_period(series, period) for period in periods]
        )
        Sa = Sa_gals / 100  # to m/s2
        return Sa.tolist()
