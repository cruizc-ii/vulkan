from __future__ import annotations
from app.strana import IDA, SummaryEDP
from random import randrange, uniform
import numpy as np
from dataclasses import dataclass, asdict, field
import pandas as pd
from shortuuid import uuid
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line
from app.assets import Asset, LossResultsDataFrame
from app.hazard import HazardCurve, TimeHistorySeries
from app.utils import EDP, NamedYamlMixin, IDAResultsDataFrame, YamlMixin
from pathlib import Path
import plotly.express as px
from app.utils import ROOT_DIR

LOSS_DIR = ROOT_DIR / "models" / "loss"
LOSS_CSV_RESULTS_DIR = ROOT_DIR / "models" / "loss_csvs"
RATE_CSV_RESULTS_DIR = ROOT_DIR / "models" / "rate_csvs"
# EXPECTED_LOSS_CSV_RESULTS_DIR = ROOT_DIR / "models" / "expected_loss_csvs"
# ACCELS_CSV_RESULTS_DIR = ROOT_DIR / "models" / "accels_csvs"
# SAMPLED_STRANA_CSV_RESULTS_DIR = ROOT_DIR / "models" / "sampled_strana_results_csvs"
# EXPECTED_STRANA_CSV_RESULTS_DIR = ROOT_DIR / "models" / "expected_strana_results_csvs"


@dataclass
class Loss:
    name: str = None
    src: str = None
    net_worth: float = None
    average_annual_loss: float = None
    average_annual_loss_pct: float = None
    expected_loss: float = None
    expected_loss_pct: float = None
    std_loss: float = None
    _loss_df: LossResultsDataFrame = None
    _ida_results_df: IDAResultsDataFrame = None
    rate_src: str = None
    _rate_df: pd.DataFrame = None
    _RATE_NUM_BINS: int = 20
    scatter_src: str = None
    _scatter_df: pd.DataFrame = None

    def __post_init__(self):
        if self.src:
            self._loss_df = pd.read_csv(self.src)
            self._loss_df = self._loss_df.set_index(["intensity", "freq"])
        if self.rate_src:
            self._rate_df = pd.read_csv(self.rate_src, index_col=0)

    def save_df(self, df: pd.DataFrame, folder: Path, name: str = "") -> str:
        simID = uuid()
        src = str((folder / f"{name}-{simID}.csv").resolve())
        df.to_csv(src)
        return src

    def stats(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.average_annual_loss,
            self.expected_loss,
            self.std_loss,
            # self.sum_losses,
            self.expected_loss_pct,
            self.average_annual_loss_pct,
            self.net_worth,
        )

    def _get_and_set_loss_statistics(self):
        df = self._loss_df.copy(deep=True)
        freq = df.index.get_level_values("freq")
        lambda0 = sum(freq)
        df["aal"] = df["mean"] * freq.values
        aal = float(df["aal"].sum())
        mean = aal / lambda0
        self.expected_loss = mean
        self.average_annual_loss = aal
        self.average_annual_loss_pct = self.average_annual_loss / self.net_worth
        self.expected_loss_pct = self.expected_loss / self.net_worth
        return self.stats()

    def _compute_rate_losses(self) -> pd.DataFrame:
        self._loss_linspace = np.linspace(0, self.net_worth, self._RATE_NUM_BINS)
        df = self._loss_df.copy(deep=True)
        df = df[df.columns.difference(["mean", "aal", "std"])]
        record_columns = df.columns
        df["num"] = df[record_columns].count(axis=1).values
        columns = df.columns
        freq = df.index.get_level_values("freq")
        for loss in self._loss_linspace:
            df[loss] = (
                df[df[record_columns] > loss].count(axis=1).values
                * freq.values
                / df["num"]
            )
        self._rate_df = df[df.columns.difference(columns)].sum()
        self.rate_src = self.save_df(
            self._rate_df, RATE_CSV_RESULTS_DIR, name=self._csv_name
        )
        return self._rate_df

    def expected_loss_and_variance_fig(self, normalization: float = 1.0):
        df = self._loss_df
        df = df * 1.0 / normalization
        df["std"] = df[df.columns.difference(["mean"])].std(axis=1, ddof=0)
        df = df.reset_index()
        fig = px.line(df, x="intensity", y=["mean", "std"], markers=True)
        fig.update_layout(
            xaxis_title="accel (g)",
            yaxis_title="Loss $",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            title_font_size=24,
        )
        return fig

    def rate_fig(self, normalization: float = 1.0, log_axes: bool = True):
        df = self._rate_df
        df.index = df.index * 1.0 / normalization
        fig = px.line(df, markers=True)
        if log_axes:
            fig.update_layout(
                xaxis_type="log",
                yaxis_type="log",
            )
        fig.update_layout(
            # xaxis_range=[-0.3, df.values.max()],
            # yaxis_range=[-0.1, 0.15],
            xaxis_title="$",
            yaxis_title="v(L)",
        )
        return fig

    def aggregated_expected_loss_and_variance_fig(self, df: pd.DataFrame):
        # start, stop, num = df.index.min(), df.index.max(), 10
        # binspace = np.linspace(start=start, stop=stop, num=num)
        # bins = pd.cut(df.index, binspace)
        # df = df.groupby(bins).agg(["mean", "std"]).fillna(0)
        # df.columns = [
        #     "_".join([str(c) for c in col]) if type(col) is tuple else col
        #     for col in df.columns.values
        # ]  # this is because groupy and .agg makes them a multiindex and plotly does not support plotting this..
        # df.index = df.index.astype(
        #     "str"
        # )  # sometimes they are ints so we have to convert them to strings..
        fig = df.plot()
        fig.update_layout(
            xaxis_title="accel (g)",
            yaxis_title="Loss $",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            title_font_size=24,
            # xaxis_type="log",
            # yaxis_type="log",
        )
        return fig

    def deaggregate_rate_of_exceedance(
        self, df: pd.DataFrame, key: str
    ) -> pd.DataFrame:
        keys = df[key].unique()
        rates = {}
        for k in keys:
            srcs = df[df[key].isin([k])]["src"].values
            sum_df = None
            for src in srcs:
                adf = pd.read_csv(src)
                adf = adf.set_index(["intensity", "freq"])
                if sum_df is None:
                    sum_df = adf
                else:
                    sum_df = sum_df.add(adf)
            rates[k] = self._rate_of_exceedance_for_loss_df(sum_df)
        vdf = pd.DataFrame(rates, index=rates[k].index)
        return vdf

    def _rate_of_exceedance_for_loss_df(self, df: LossResultsDataFrame) -> pd.DataFrame:
        """
        loss results df has either a column called freq or index
        and has columns for the loss of each record at each intensity
        """
        _loss_linspace = np.linspace(0, self.net_worth, self._RATE_NUM_BINS)
        df = df[df.columns.difference(["mean", "aal", "std"])]
        columns = df.columns
        freq = df.index.get_level_values("freq")
        for loss in _loss_linspace:
            df[loss] = (
                df[df[columns] > loss].count(axis=1).values
                * freq.values
                / df[columns].count(axis=1)
            )
        rate_df = df[df.columns.difference(columns)].sum()
        return rate_df

    def multiple_rates_of_exceedance_fig(self, df: pd.DataFrame, key: str):
        vdf = self.deaggregate_rate_of_exceedance(df, key)
        fig = vdf.plot()
        fig.update_layout(
            xaxis_title="$",
            yaxis_title="v(L)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            xaxis_type="log",
            yaxis_type="log",
            # title_font_size=24,
            # xaxis_range=[0, 5],
            # margin=dict(l=0, r=0, b=0),
            # title_text=f"rates of exceedance",
        )
        return fig

    # fig = Figure()
    # for col in df.columns:
    #         arr = df[col]
    #         v_df = HazardCurve.rate_of_exceedance(array=arr, years=years, name=col)[
    #             [col, "v"]
    #         ]
    #         trace = Scattergl(
    #             x=v_df[col],
    #             y=v_df.v,
    #             name=col,
    #         )
    #         fig.add_trace(trace)

    #     return fig


@dataclass
class LossModel(YamlMixin, Loss):
    _asset: Asset = None
    category: str = None
    floor: str = None

    def __post_init__(self):
        super().__post_init__()
        if self._asset is not None:
            self.category = self._asset.category
            self.floor = self._asset.floor
            self.net_worth = self._asset.net_worth
            self.name = self._asset.name
            self._compute_losses()
            self._get_and_set_loss_statistics()
            self._compute_rate_losses()

    def _compute_losses(self) -> LossResultsDataFrame:
        self._csv_name = f"{self.name}-{self.floor}"
        df = self._ida_results_df
        df["loss"] = self._asset.dollars(strana_results_df=self._ida_results_df)
        df["name"] = self._asset.name
        df["category"] = self._asset.category
        df["floor"] = self._asset.floor
        self._scatter_df = df.copy(deep=True)
        df = df.pivot(index=["intensity", "freq"], columns="record", values="loss")
        df["mean"] = df.mean(axis=1)
        self._loss_df = df
        self.src = self.save_df(
            self._loss_df, LOSS_CSV_RESULTS_DIR, name=self._csv_name
        )


class IDANotFoundException(FileNotFoundError):
    pass


@dataclass
class LossAggregator(NamedYamlMixin, Loss):
    ida_model_path: str = None
    loss_models: list["LossModel"] = None
    _assets: list[Asset] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        try:
            self._ida = IDA.from_file(self.ida_model_path)
            self.net_worth = self._ida._design.fem.total_net_worth
            self._hazard = self._ida._hazard
            self._assets = self._ida._design.fem.assets
            self._ida_results_df = pd.DataFrame.from_dict(self._ida.results)
            self._csv_name = "total"
            self._scatter_csv_name = "scatter"
            self._loss_linspace = np.linspace(0, self.net_worth, self._RATE_NUM_BINS)
        except FileNotFoundError:
            raise IDANotFoundException

        if self.scatter_src:
            self._scatter_df = pd.read_csv(self.scatter_src, index_col=0)

    def run(self) -> None:
        self._create_loss_models()
        self._compute_losses()
        self._get_and_set_scatter_df()
        self._get_and_set_loss_statistics()

    def _create_loss_models(self) -> list[LossModel]:
        self.loss_models = [
            LossModel(
                _asset=asset,
                _ida_results_df=self._ida_results_df,
            )
            for asset in self._assets
            if asset
        ]
        return self.loss_models

    def _compute_losses(self) -> LossResultsDataFrame:
        loss_dfs = [lm._loss_df for lm in self.loss_models]
        self._loss_df = self.sum_columns_for_similar_dfs(loss_dfs)
        self.src = self.save_df(
            self._loss_df, LOSS_CSV_RESULTS_DIR, name=self._csv_name
        )
        self._compute_rate_losses()
        return self._loss_df

    def _get_and_set_scatter_df(self) -> pd.DataFrame:
        df = self._concat_scatter_dfs()
        self.scatter_src = self.save_df(
            df, LOSS_CSV_RESULTS_DIR, name=self._scatter_csv_name
        )
        self._scatter_df = df

    def _concat_scatter_dfs(self) -> pd.DataFrame:
        scatter_dfs = [lm._scatter_df for lm in self.loss_models]
        df = scatter_dfs[0]
        for _df in scatter_dfs[1:]:
            df = pd.concat([df, _df])
        return df

    def sum_columns_for_similar_dfs(
        self, dfs: list[pd.DataFrame], columns=True
    ) -> pd.DataFrame:
        """algebraically adds dataframes that have the same schema along all their columns, ignoring index"""
        df = dfs[0]
        # if index in df.columns:
        #     df = df.set_index(index)
        for _df in dfs[1:]:
            # if index in df.columns:
            #     _df = _df.set_index(index)
            df = df.add(_df)
        # df = df.reset_index()
        return df

    def aggregate_src_df(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        we need to load the df for each asset
        first, get all csvs from src and put the numpy array of losses in a col
        now groupby the key and construct a new dataframe
        where each row is the
        """
        df["asset"] = df[["src"]].applymap(
            lambda src: pd.read_csv(src, usecols=["mean"], squeeze=True).to_numpy()
        )
        grouped = df.groupby(key)[["asset"]].agg(np.sum)
        data = {
            **grouped.to_dict()["asset"],
        }
        df = pd.DataFrame(data, index=self._loss_df.index.get_level_values("intensity"))
        df["total"] = df.sum(axis=1)
        return df

    @staticmethod
    def filter_src_df(
        df: pd.DataFrame,
        category_filter: list[str],
        name_filter: list[str],
        storey_filter: list[str],
    ) -> pd.DataFrame:
        df = df[
            (df["category"].isin(category_filter))
            & (df["name"].isin(name_filter))
            & (df["floor"].isin(storey_filter))
        ]
        return df

    @property
    def summary(self) -> dict:
        return {
            "AAL $": self.average_annual_loss,
            "EL $": self.expected_loss,
            "std L $": self.std_loss,
            # self.sum_losses,
            "EL %": self.expected_loss_pct,
            "aal %": self.average_annual_loss_pct,
            "net worth $": self.net_worth,
        }

    @property
    def asset_records(self) -> list[dict]:
        return [a.to_dict for a in self._assets]

    @property
    def assets_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.asset_records)

    @property
    def loss_models_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_models)

    def scatter_fig(
        self, category_filter=None, name_filter=None, floor_filter=None
    ) -> Figure:
        df = self._scatter_df
        df = self.filter_src_df(df, category_filter, name_filter, floor_filter)
        fig = px.scatter(
            df,
            x="intensity",
            y="loss",
            color="name",
            # size='loss',
            # hover_data=['petal_width']
        )
        return fig

    # def _init_dfs(self):
    #     """
    #     samples from the strana results df using the simulated accels from the hazard and interpolating
    #     constructs an expected_results_df sampling from uniform linspace and interpolating
    #     """
    #     df = self._ida_results_df.sample(frac=1)  # shuffle the dataframe.
    #     df = df.drop_duplicates(
    #         "intensity_str"
    #     )  # effectively samples records for a given intensity, dropping the duplicate rows
    #     df = df.set_index("intensity")
    #     df2 = df.copy(deep=True)
    #     accels = self._accels_series.values
    #     merged_index = df.index.union(accels).sort_values()
    #     df = df.sort_index().reindex(merged_index, method="nearest")
    #     self._sampled_strana_results_df = df.loc[accels]
    #     self.save_df(self._sampled_strana_results_df, SAMPLED_STRANA_CSV_RESULTS_DIR)

    #     low, high = min(df.index), max(df.index)
    #     uniform_linspace = np.linspace(
    #         start=low, stop=high, num=self._NUM_SAMPLES_EXPECTED_LOSS
    #     )
    #     uniform_merged_index = df2.index.union(uniform_linspace).sort_values()
    #     df2 = df2.sort_index().reindex(uniform_merged_index, method="nearest")
    #     self._expected_strana_results_df = df2.loc[uniform_linspace]
    #     self.save_df(self._expected_strana_results_df, EXPECTED_STRANA_CSV_RESULTS_DIR)

    # def _simulate_accels(self) -> TimeHistorySeries:
    #     self._accels_series = self._hazard.get_simulated_timehistory(
    #         num_simulations=self.num_simulations
    #     )
    #     self.num_years = float(self._accels_series.index[-1])
    #     self.accels_src = self.save_df(self._accels_series, ACCELS_CSV_RESULTS_DIR)
    #     return self._accels_series
