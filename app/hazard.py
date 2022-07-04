from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import os
import subprocess
import pathlib
from posixpath import basename
import pandas as pd
from pandas.core import series
import pandera as pa
import numpy as np
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
import plotly.express as px
from app.utils import NamedYamlMixin, YamlMixin, UploadComponent
from typing import Optional
import yaml

ROOT_DIR = pathlib.Path(__file__).parent.parent
RECORDS_DIR = ROOT_DIR / "records"
CONFIG_FILE = RECORDS_DIR / "record-parameters.yml"
SDOF_API_PATH = ROOT_DIR / "api" / "sdof" / "elastoplastic-sdof-api.tcl"
SA_SPECTRA_DIR = RECORDS_DIR / "Sa_spectra"
SD_SPECTRA_DIR = RECORDS_DIR / "Sd_spectra"
SV_SPECTRA_DIR = RECORDS_DIR / "Sv_spectra"
ACCEL_SPECTRA_DIR = RECORDS_DIR / "accel_spectra"
VEL_SPECTRA_DIR = RECORDS_DIR / "vel_spectra"

TimeHistoryDataFrame = pd.DataFrame
TimeHistorySeries = pd.Series

try:
    with open(CONFIG_FILE) as f:
        config_file = yaml.safe_load(f)
except FileNotFoundError as e:
    print(e)

dataframe_schema = pa.DataFrameSchema(
    columns={".*": pa.Column(pa.Float32, regex=True, coerce=True)},
    index=pa.Index(pa.Float32, coerce=True),
)

series_schema = pa.SeriesSchema(
    pa.Float64,
    index=pa.Index(pa.Float64, coerce=True),
    nullable=False,
)


class RecordKind(Enum):
    ACCEL = "acc"
    VEL = "vel"
    DISP = "disp"
    FOURIER = "fourier"


@dataclass
class Spectra:
    path: str
    scale: float = 1.0
    _df: pd.DataFrame = None

    def validate(self):
        dataframe_schema.validate(self._df)

    @classmethod
    def from_csv(cls, path: str, scale: float = None, **csv_kwargs):
        df = pd.read_csv(
            path, header=None, names=["Sa"], index_col=0, sep=" ", **csv_kwargs
        )
        instance = cls(path=path, scale=scale, _df=df)
        return instance

    def __post_init__(self):
        self._df = pd.read_csv(
            self.path, header=None, names=["Sa"], index_col=0, sep=" "
        )
        self.validate()
        self._df = self.scale * self._df
        self.maxValue = np.max(self._df.values)
        self.minValue = np.min(self._df.values)
        self.maxAbs = np.max((np.abs(self.maxValue), np.abs(self.minValue)))

    def get_ordinate_for_period(self, period: float) -> float:
        self._df.loc[period] = np.NaN
        self._df.sort_index(inplace=True)
        self._df.interpolate(inplace=True)
        ordinate = self._df.loc[period]
        return ordinate

    def get_ordinates_for_periods(self, periods: list[float]) -> list[float]:
        return [self.get_ordinate_for_period(p) for p in periods]


@dataclass
class Record:
    path: str
    dt: float = 0.01
    scale: float = 1.0
    name: str | None = None
    kind: RecordKind = field(default=RecordKind.ACCEL.value)
    _spectra: pd.DataFrame = None
    _df: pd.DataFrame = None

    def validate(self):
        series_schema.validate(self._df)

    def __post_init__(self):
        self.name = os.path.basename(Path(self.path).name)
        self.dt = config_file.get(self.name, {}).get("dt", self.dt)
        self.scale = config_file.get(self.name, {}).get("scale", self.scale)
        self._df = pd.read_csv(self.path, header=None).squeeze("columns")
        self.validate()
        n = len(self._df)
        self.duration = (n - 1) * self.dt
        self.steps = len(self._df)
        time = np.linspace(0, self.duration, n)
        self._df.index = time
        self._df = self.scale * self._df
        self.maxValue = np.max(self._df.values)
        self.minValue = np.min(self._df.values)
        self.maxAbs = np.max((np.abs(self.maxValue), np.abs(self.minValue)))
        try:
            spectra_path = SA_SPECTRA_DIR / self.name
            spectra = pd.read_csv(
                spectra_path, header=None, names=["Sa"], index_col=0, sep=" "
            )
            dataframe_schema.validate(spectra)
            self._spectra = spectra
        except FileNotFoundError as e:
            spectra_outpath = SA_SPECTRA_DIR.resolve()
            run_string = f"{SDOF_API_PATH} name={self.name} record={self.path} outputdir={spectra_outpath} spectra=true Sa=t damping=0.05 dt={self.dt} scale={self.scale}"
            failure = subprocess.call(run_string, shell=True)
            if not failure:
                spectra = pd.read_csv(
                    spectra_path, header=None, names=["Sa"], index_col=0, sep=" "
                )
                dataframe_schema.validate(spectra)
                self._spectra = spectra
            else:
                raise e
            self.scale = config_file.get(self.name, {}).get("scale", 1.0)

    @property
    def pfa(self) -> float:
        """this is bad.. we should search for the lowest T. we are assuming loc[0]"""
        accel = config_file.get(self.name, {}).get("pfa")
        if accel is None:
            accel: float = self._spectra.iloc[0]["Sa"]
        return accel

    @property
    def summary(self):
        return {"maxAbs": self.maxAbs, "duration": self.duration}

    @property
    def abspath(self):
        if not isinstance(self.path, str):
            return str(self.path.resolve())
        return self.path

    @property
    def figure(self):
        df = self._df
        fig = px.line(df, x=df.index, y=df.values)
        fig.update_traces(line=dict(width=1.0, color="Black"))
        fig.update_layout(
            xaxis_title="s",
            yaxis_title="acc (gals)",
            title_text=f"{self.name}",
        )
        return fig

    @property
    def spectra(self):
        df = self._spectra
        x, y = df.index, df.Sa
        fig = px.line(df, x=x, y=y)
        fig.update_traces(line=dict(width=1.0, color="Black"))
        fig.update_layout(
            xaxis_title="T (s)", yaxis_title="Sa (gals)", title_text=f"Sa 5% spectra"
        )
        return fig

    def get_scale_factor(self, period, intensity) -> float:
        df = self._spectra
        gals = intensity * 981
        df = gals / df
        try:
            factor = df.loc[period].values[0]
        except KeyError:
            df.loc[period] = None
            df = df.sort_index()
            df = df.interpolate()
            factor = df.loc[period].values[0]
        return float(factor)

    def get_natural_scale_factor(self, period):
        """
        returns a numerical value in (g) such that
        the scale for which to multiply the record is 0.01
        (unity) but in meters
        """
        df = self._spectra
        try:
            spectral_value = df.loc[period].values[0]
        except KeyError:
            df.loc[period] = None
            df = df.sort_index()
            df = df.interpolate()
            spectral_value = df.loc[period].values[0]

        natural_factor = spectral_value / 981
        return natural_factor


@dataclass
class Hazard(NamedYamlMixin):
    name: str
    records: list[Record] = field(default_factory=list)
    kind: str = RecordKind.ACCEL.value
    curve: Optional[dict] = None
    _curve: Optional["HazardCurveFactory"] = None

    def __post_init__(self):
        if len(self.records) > 0 and isinstance(self.records[0], dict):
            self.records = [Record(**data) for data in self.records]
        if isinstance(self.curve, dict):
            self._curve = HazardCurveFactory(**self.curve)

    def get_simulated_timehistory(self, num_simulations: int = 10) -> TimeHistorySeries:
        return self._curve.get_simulated_timehistory(num_simulations)

    def add_record(self, record: Record) -> bool:
        paths = [r.path for r in self.records]
        if record.path in paths:
            print(paths)
            print("record path in paths")
            return False
        self.records.append(record)
        return True

    def remove_record(self, record_path: str) -> bool:
        self.records = [r for r in self.records if r.path != record_path]
        return True

    def intensities_for_idas(self):
        return self._curve.intensities_for_idas()

    @property
    def rate_figure(self):
        return self._curve.figure


@dataclass
class HazardCurveFactory(ABC, YamlMixin):
    """
    implement x and y as arrays.
    x = Sa (usually but can be any intensity)
    y = 1/yr
    call super().__post_init__() to build the _df
    """

    name: str
    x: Optional[list] = None
    y: Optional[list] = None
    _df: Optional[pd.DataFrame] = None
    _IDA_LINSPACE_BINS: int = 10

    def __post_init__(self):
        if self._df is None:
            if self.y and self.x:
                self._df = pd.DataFrame(dict(a=self.y), index=self.x)
                dataframe_schema.validate(self._df)
        else:
            dataframe_schema.validate(self._df)
            self.x, self.y = self._df.x.to_list(), self._df.y.to_list()

    @property
    @abstractmethod
    def html(self):
        """
        a set of input groups corresponding to the params needed
        in the class __post_init__
        """
        pass

    @property
    def figure(self):
        fig = Figure()
        trace = Scattergl(x=self.x, y=self.y, marker=dict(color="LightSkyBlue"))
        fig.add_trace(trace)
        fig.update_layout(
            xaxis_title="Sa",
            yaxis_title="1/yr",
            title_text="Rate of exceedance.",
            xaxis_type="log",
            yaxis_type="log",
        )
        return fig

    def interpolate_rate_for_values(self, values: list[float]) -> list[float]:
        df = self._df
        merged = df.index.union(values)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear", limit_direction="both", limit=1)
        sas = df["a"].loc[values].to_list()
        return sas

    def samples(self, n: int = None, range_space: np.ndarray = None) -> list[float]:
        """
        range_space is the range of the CDF that we try to sample,
        if None then we sample uniformly
        """
        if not n:
            n = self._IDA_LINSPACE_BINS
        df = self._df
        v0 = df["a"].iloc[0]
        self.v0 = v0
        df["S"] = df / v0  # normalize to get survival function
        if range_space is None:
            range_space = np.random.uniform(
                0, 1, n
            )  # from 0 to 1 (we are sampling using the range of the CDF)

        df["cdf"] = 1 - df["S"]
        df.index.name = "Sa"
        df = df.reset_index()
        df = df.set_index(df.cdf)
        idx1 = df.index
        merged = idx1.union(range_space)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear").dropna(axis=0, how="any")
        sas = df["Sa"].loc[range_space].to_list()
        return sas

    def get_simulated_timehistory(self, num_simulations: int = 10) -> TimeHistorySeries:
        sas = self.samples(n=num_simulations)
        df = self._df
        ix = df.index
        merged = ix.union(sas)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear").dropna(axis=0, how="any")
        years_random_linspace = np.random.rand(num_simulations)
        years = -1 * np.log(1 - years_random_linspace) / self.v0
        years = years.cumsum()
        # years = 1./df['a'].loc[sas]
        # years = years.values.cumsum()
        df = TimeHistorySeries(sas, index=years)
        return df

    def intensities_for_idas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, stop = self._df.index.min(), self._df.index.max()
        linspace, step = np.linspace(
            start=start, stop=stop, num=self._IDA_LINSPACE_BINS, retstep=True
        )
        sups = linspace + step / 2
        infs = linspace - step / 2
        return linspace, sups, infs

    # def intensities_for_idas(self) -> np.ndarray:
    #     unif = np.random.uniform(0, 1, self._IDA_LINSPACE_SAMPLE)
    #     skewed = np.array([0.9, 0.95, 0.99, 0.9999])
    #     range_space = np.concatenate([unif, skewed])
    #     sas = self.samples(range_space=range_space)
    #     # sas += [1.0, 1.5, 2.0, 3.0]
    #     sas = sorted(sas)
    #     return sas

    @staticmethod
    def rate_of_exceedance(
        array: np.ndarray, years=None, name: str = "values", method: str = "average"
    ) -> pd.DataFrame:
        """
        array must have repeated values otherwise this wont work
        returns a normalized dataframe with the rate of exceedance
        converts array into dataframe -> cdf -> 1-v/v0 = CDF
        """
        df = pd.DataFrame(pd.Series(array, name=name).dropna())
        df["cdf"] = df.rank(method="average", pct=True)
        df.sort_values(name, inplace=True)
        events = df[name].size
        if not years:
            years = events
        df["v"] = (1 - df.cdf) * events / years
        df["pdf"] = pd.Series(array)
        return df


@dataclass
class ParetoCurve(HazardCurveFactory):
    """
    specify html input fields with metadata attr
    """

    name: str = "pareto"
    v0: float = field(default=2.0, metadata={"input": True, "unit": "1/yr"})
    a0: float = field(default=0.05, metadata={"input": True, "unit": "(g)"})
    r: float = field(default=2.0, metadata={"input": True, "unit": ""})
    amax: float = field(default=3.0, metadata={"input": True, "unit": "(g)"})
    _DEFAULT_POINTS = 200

    def __post_init__(self) -> None:
        if not self.x:
            x = np.linspace(self.a0, self.amax, self._DEFAULT_POINTS)
        if not self.y:
            if isinstance(x, list):
                x = np.array(x)
            y = self.v0 * (self.a0 / x) ** self.r
            self.x = x.tolist()
            self.y = y.tolist()
        super().__post_init__()

    @property
    def html(self):
        pass

    #     return html.Div(
    #         [
    #             html.H4("Parameters", className="mt-2"),
    #         ]
    #         + [
    #             dbc.InputGroup(
    #                 [
    #                     dbc.InputGroupAddon(f"{f.name}  ", addon_type="prepend"),
    #                     dbc.Input(
    #                         id={"type": "hazard-model-input-args", "index": ix},
    #                         value=f"{getattr(self, f.name)}",
    #                         key=f"{f.name}",
    #                     ),
    #                     dbc.InputGroupAddon(
    #                         f" {f.metadata.get('unit')}", addon_type="append"
    #                     ),
    #                 ]
    #             )
    #             for ix, f in enumerate(fields(self))
    #             if f.metadata.get("input")
    #         ]
    #     )

    def get_simulated_timehistory(
        self, num_simulations: int = 10, decimals: int = 4
    ) -> "TimeHistorySeries":
        # simulating WITHOUT decimals leads to continuous values.. which rank(pct=True) will return a correct CDF
        # otherwise we would need to groupby().agg(count) to count the freq of each value (since if we are rounding there WILL be duplicates)
        # let's sample from Reals for now..
        years_random_linspace = np.random.rand(num_simulations)
        Sa_random_linspace = np.random.rand(num_simulations)
        years = -1 * np.log(1 - years_random_linspace) / self.v0
        years = years.cumsum()
        Sa = self.a0 / np.sqrt(1 - Sa_random_linspace)
        # Sa = np.round(Sa, decimals)
        df = TimeHistorySeries(Sa, index=years)
        # df.to_csv('./accel-sim.csv')
        return df


@dataclass
class UserDefinedCurve(HazardCurveFactory):
    name: str = "user"


#     @property
#     def html(self):
#         return html.Div(
#             [
#                 dbc.InputGroup(
#                     [
#                         dbc.InputGroupAddon("Normalize Sa   ", addon_type="prepend"),
#                         dbc.Input(
#                             id="normalize-accels-input",
#                             placeholder="1.0",
#                             value=1,
#                             type="number",
#                         ),
#                     ]
#                 ),
#                 UploadComponent(
#                     id="upload-hazard-file", msg="Upload a comma-separated file a,v(a)"
#                 ),
#             ]
#         )


class HazardCurveFactory:
    models = {"pareto": ParetoCurve, "user": UserDefinedCurve}

    def __new__(cls, **data) -> HazardCurveFactory:
        name = data["name"]
        return cls.models[name](**data)

    @classmethod
    def add(cls, name, seed):
        cls.models[name] = seed

    @classmethod
    def options(cls):
        return list(cls.models.keys())
