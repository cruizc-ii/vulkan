from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import os
import subprocess
import pathlib
from posixpath import basename
from pandas import DataFrame, Series, read_csv
import pandera as pa
import numpy as np
from enum import Enum
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
import plotly.express as px
from app.utils import NamedYamlMixin, YamlMixin, UploadComponent
import yaml
import streamlit
from typing_extensions import TypeAlias
from app.utils import GRAVITY


ROOT_DIR = pathlib.Path(__file__).parent.parent
RECORDS_DIR = ROOT_DIR / "records"
CONFIG_FILE = RECORDS_DIR / "record-parameters.yml"
SDOF_API_PATH = ROOT_DIR / "api" / "opensees-elastoplastic-sdof-api.tcl"
SA_SPECTRA_DIR = RECORDS_DIR / "Sa_spectra"
SD_SPECTRA_DIR = RECORDS_DIR / "Sd_spectra"
SV_SPECTRA_DIR = RECORDS_DIR / "Sv_spectra"
ACCEL_SPECTRA_DIR = RECORDS_DIR / "accel_spectra"
VEL_SPECTRA_DIR = RECORDS_DIR / "vel_spectra"

TimeHistoryDataFrame: TypeAlias = DataFrame
TimeHistorySeries: TypeAlias = Series

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
    _df: TimeHistoryDataFrame = field(default_factory=TimeHistoryDataFrame)

    def validate(self):
        dataframe_schema.validate(self._df)

    @classmethod
    def from_csv(cls, path: str, scale: float = 1.0, **csv_kwargs):
        df = read_csv(
            path, header=None, names=["Sa"], index_col=0, sep=" ", **csv_kwargs
        )
        instance = cls(path=path, scale=scale, _df=df)
        return instance

    def __post_init__(self):
        self._df = read_csv(self.path, header=None, names=["Sa"], index_col=0, sep=" ")
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
    kind: str = field(default=RecordKind.ACCEL.value)
    _spectra: DataFrame = None
    _df: DataFrame = None

    def validate(self):
        series_schema.validate(self._df)

    def __post_init__(self):
        self.name = os.path.basename(Path(self.path).name)
        self.dt = config_file.get(self.name, {}).get("dt", self.dt)
        self.scale = config_file.get(self.name, {}).get("scale", self.scale)
        self._df = read_csv(self.path, header=None).squeeze("columns")
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
            spectra = read_csv(
                spectra_path, header=None, names=["Sa"], index_col=0, sep=" "
            )
            dataframe_schema.validate(spectra)
            self._spectra = spectra
        except FileNotFoundError as e:
            spectra_outpath = SA_SPECTRA_DIR.resolve()
            run_string = f"{SDOF_API_PATH} name={self.name} record={self.path} outputdir={spectra_outpath} spectra=true Sa=t damping=0.05 dt={self.dt} scale={self.scale}"
            failure = subprocess.call(run_string, shell=True)
            if not failure:
                spectra = read_csv(
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

    def figure(self, normalize_g: bool=False):
        df = self._df
        x, y = df.index, df.values
        y = y / 100 # from gals to m/s/s
        y = y / GRAVITY if normalize_g else y
        fig = px.line(df, x=x, y=y)
        fig.update_traces(line=dict(width=1.0, color="Black"))
        fig.update_layout(
            xaxis_title="s",
            yaxis_title="a (g)" if normalize_g else "a (m/s/s)",
            title_text=f"{self.name}",
        )
        return fig

    def spectra(self, normalize_g: bool=False):
        df = self._spectra
        x, y = df.index, df.Sa
        y = y / 100 # from gals to m/s/s
        y = y / GRAVITY if normalize_g else y
        fig = px.line(df, x=x, y=y)
        fig.update_traces(line=dict(width=1.0, color="Black"))
        fig.update_layout(
            xaxis_title="T (s)",
            yaxis_title="a (g)" if normalize_g else "a (m/s/s)",
            title_text=f"Sa 5% spectra"
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
    curve: str | dict | None = None
    _curve: "HazardCurve" | None = None

    def __post_init__(self):
        if len(self.records) > 0 and isinstance(self.records[0], dict):
            self.records = [Record(**data) for data in self.records]
        if isinstance(self.curve, dict):
            self._curve = HazardCurveFactory(**self.curve)
        if isinstance(self.curve, str):
            self._curve = HazardCurveFactory(name=self.curve)

    def simulate_intensities(self, n: int = 10) -> "TimeHistorySeries":
        return self._curve.simulate_intensities(n)

    def add_record(self, record: Record) -> bool:
        paths = [r.path for r in self.records]
        if record.path in paths:
            return False
        self.records.append(record)
        return True

    def remove_record(self, record_path: str) -> bool:
        self.records = [r for r in self.records if r.path != record_path]
        return True

    def intensities_for_idas(self):
        return self._curve.intensities_for_idas()

    @property
    def record_names(self) -> list[str]:
        return [r.name for r in self.records if r.name]

    def rate_figure(self, normalize_g:bool=True, logx:bool=True, logy:bool=True):
        return self._curve.figure(normalize_g=normalize_g, logx=logx, logy=logy)


@dataclass
class HazardCurve(ABC, YamlMixin):
    """
    describes annual rates of exceedance of intensity 'a'
    x = intensity (usually Sa in g)
    y = 1/unit_of_time usually 1/yr
    ALWAYS call super().__post_init__() to build the _df
    """

    name: str
    x: list | None = None
    y: list | None = None
    v0: float | None = None
    _df: DataFrame = field(default_factory=DataFrame)
    _IDA_LINSPACE_BINS: int = 10

    def __str__(self) -> str:
        return str(self._df)

    def __post_init__(self):
        if self._df.size == 0:
            if self.y and self.x:
                self.v0 = self.y[0]
                self._df = DataFrame(dict(y=self.y), index=self.x)
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

    def figure(self, normalize_g:bool=True, logx=True, logy=True):
        fig = Figure()
        x, y = self.x, self.y
        x, y = np.array(x), np.array(y)
        y = y if normalize_g else y * GRAVITY
        trace = Scattergl(x=x, y=y, marker=dict(color="LightSkyBlue"))
        fig.add_trace(trace)
        fig.update_layout(
            xaxis_title="Sa (g)" if normalize_g else 'Sa m/s/s',
            yaxis_title="1/yr",
            title_text="annual rate of exceedance.",
            # xaxis_type="log",
            # yaxis_type="log",
        )
        if logx:
            fig.update_xaxes(type="log")
        if logy:
            fig.update_yaxes(type="log")
        return fig

    def interpolate_rate_for_values(self, values: list[float]) -> list[float]:
        df: DataFrame = self._df
        merged = df.index.union(values)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear", limit_direction="both", limit=1)
        sas = df["y"].loc[values].to_list()
        return sas

    def samples(
        self, n: int = _IDA_LINSPACE_BINS, range_space: np.ndarray = None
    ) -> list[float]:
        """
        range_space is the range of the CDF that we try to sample,
        if None then we sample using Uni[0, 1] using n samples
        """
        df = self._df
        df["S"] = df / self.v0  # normalize v(a) to get survival function
        range_space = np.random.uniform(0, 1, n) if range_space is None else range_space
        df["cdf"] = 1 - df["S"]
        df.index.name = "Sa"
        df = df.reset_index()
        df = df.set_index(df["cdf"])
        idx1 = df.index
        merged = idx1.union(range_space)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear").dropna(axis=0, how="any")
        sas = df["Sa"].loc[range_space].to_list()
        return sas

    def simulate_intensities(self, n: int = 10) -> "TimeHistorySeries":
        sas = self.samples(n=n)
        df = self._df
        ix = df.index
        merged = ix.union(sas)
        df = df.reindex(merged)
        df = df.sort_index()
        df = df.interpolate(method="linear").dropna(axis=0, how="any")
        years_random_linspace = np.random.rand(n)
        years = -1 * np.log(1 - years_random_linspace) / self.v0
        years = years.cumsum()
        # years = 1./df['a'].loc[sas]
        # years = years.values.cumsum()
        df = TimeHistorySeries(sas, index=years)
        return df

    def intensities_for_idas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df: DataFrame = self._df
        start, stop = df.index.min(), df.index.max()
        linspace, step = np.linspace(
            start=start, stop=stop, num=self._IDA_LINSPACE_BINS, retstep=True
        )
        sups = linspace + step / 2
        infs = linspace - step / 2
        return linspace, sups, infs

    @staticmethod
    def rate_of_exceedance(
        array: np.ndarray,
        years: bool | float | None = None,
        name: str = "values",
        method: str = "average",
    ) -> DataFrame:
        """
        array MUST have repeated values otherwise this wont work ..??
        returns a normalized dataframe with the rate of exceedance
        converts array into dataframe -> cdf -> 1-v/v0 = CDF
        """
        df = DataFrame(Series(array, name=name).dropna())
        df["cdf"] = df.rank(method="average", pct=True)
        df.sort_values(name, inplace=True)
        # num_events = df[name].size
        num_events = len(array)
        years = num_events if not years else years
        df["v"] = (1 - df.cdf) * num_events / years
        df["pdf"] = Series(array)
        return df


@dataclass
class ParetoCurve(HazardCurve):
    name: str = "pareto"
    v0: float = field(
        default=2.0,
        metadata={"input": True, "units": "1/yr", "help": "this is help text"},
    )
    a0: float = field(default=0.05, metadata={"input": True, "units": "(g)"})
    r: float = field(default=2.0, metadata={"input": True, "units": ""})
    amax: float = field(default=3.0, metadata={"input": True, "units": "(g)"})
    _DEFAULT_POINTS: int = 100

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

    def html(self, st: "streamlit") -> None:
        input_fields = [f for f in fields(self) if f.metadata.get("input")]
        for f in input_fields:
            meta = f.metadata
            help = meta.get("help")
            units = meta.get("units")
            name = f.name + f" {units}"
            # min_value = meta.get("min_value")
            # step = meta.get("step")
            # max_value = meta.get("max_value")
            # format = meta.get("format")
            changed = st.number_input(
                name,
                # min_value=min_value,
                # step=step or 1,
                # max_value=max_value,
                # format=format or "%d",
                help=help,
                value=getattr(self, f.name) or f.default,
            )
            # print(f.name, changed, getattr(self, f.name))
            if changed != getattr(self, f.name):
                # st.write(f.name, changed)
                setattr(self, f.name, changed)

    def simulate_intensities(
        self, n: int = 10, decimals: int = 4
    ) -> "TimeHistorySeries":
        # simulating WITHOUT rounding decimals leads to continuous values for which rank(pct=True) will return a correct CDF.
        # otherwise we would need to groupby().agg(count) to count the freq of each value (since if we are rounding there WILL be duplicates)
        years_random_linspace = np.random.rand(n)
        Sa_random_linspace = np.random.rand(n)
        years = -1 * np.log(1 - years_random_linspace) / self.v0
        years = years.cumsum()
        Sa = self.a0 / np.sqrt(1 - Sa_random_linspace)
        # Sa = np.round(Sa, decimals)
        df = TimeHistorySeries(Sa, index=years)
        # df.to_csv('./accel-sim.csv')
        return df


@dataclass
class UserDefinedCurve(HazardCurve):
    name: str = "user"

    def html(self, st: streamlit) -> None:
        """
        TODO: return a file uploader for streamlit
        """
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
        return


class HazardCurveFactory:
    models = {"pareto": ParetoCurve, "user": UserDefinedCurve}

    def __new__(cls, **data) -> HazardCurve:
        name = data["name"]
        return cls.models[name](**data)

    @classmethod
    def add(cls, name, seed):
        cls.models[name] = seed

    @classmethod
    def options(cls):
        return list(cls.models.keys())
