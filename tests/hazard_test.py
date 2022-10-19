from unittest import TestCase
import pandas as pd
from app.hazard import Record, Hazard, ParetoCurve, HazardCurveFactory, HazardCurve
from .test import RECORDS_DIR, HAZARD_FIXTURES_PATH
import numpy as np


class TestRecords(TestCase):
    maxDiff = None

    def test_from_path(self):
        r = Record(RECORDS_DIR / "98_SCT190985NS.csv")
        self.assertIsInstance(r._df, pd.Series)
        self.assertIsInstance(r._spectra, pd.DataFrame)


class TestHazardCurve(TestCase):
    maxDiff = None

    def test_pareto(self):
        v0, a0 = 2.0, 0.05
        n = 100_000
        # n = 10
        # n = 1000
        # n = 10_000
        hazard: HazardCurve = HazardCurveFactory(name="pareto", v0=v0, a0=a0)
        self.assertIsInstance(hazard, ParetoCurve)
        self.assertEqual(hazard.v0, 2)
        self.assertEqual(hazard.a0, 0.05)
        series = hazard.samples(n=n)
        ts = hazard.simulate_intensities(n=n)
        self.assertTrue(len(ts), n)
        years = ts.index[-1]
        roe = HazardCurve.rate_of_exceedance(series, years=years, name="Sa")
        index = abs(roe.Sa - a0).idxmin()
        empirical_v0 = roe.loc[index].v
        # non-deterministic test, delta is too strict.. what is the correct?
        self.assertAlmostEquals(empirical_v0, v0, delta=2.0 * np.pi / np.sqrt(n))


class TestHazard(TestCase):
    maxDiff = None

    def test_from_file(self):
        hazard = Hazard.from_file(HAZARD_FIXTURES_PATH / "test-hazard-from-file.yml")
        self.assertTrue(all([isinstance(r, Record) for r in hazard.records]))
        self.assertIsInstance(hazard._curve, HazardCurve)

    def test_to_file(self):
        abspath = str((RECORDS_DIR / "98_SCT190985NS.csv").resolve())
        record = Record(path=abspath)
        name = "hazard-from-instance-to-file"
        hazard = Hazard(name=name, records=[record], curve="pareto")
        hazard.to_file(HAZARD_FIXTURES_PATH)
        from_file = Hazard.from_file(HAZARD_FIXTURES_PATH / f"{name}.yml")
        self.assertDictEqual(hazard.to_dict, from_file.to_dict)


class TestRecord(TestCase):
    maxDiff = None

    def test_get_scale_factor(self):
        record_abspath = str((RECORDS_DIR / "98_SCT190985NS.csv").resolve())
        rec = Record(path=record_abspath)
        scale = rec.get_scale_factor(period=1.5, intensity=1.0)
        self.assertAlmostEqual(scale, 4.13, 0)

        scale = rec.get_scale_factor(period=0.21, intensity=0.2)
        self.assertAlmostEqual(scale, 1.6, delta=0.1)

        scale = rec.get_scale_factor(period=1.0, intensity=0.2)
        self.assertAlmostEqual(scale, 1.2, delta=0.1)

        # this is the one I care about
        scale = rec.get_scale_factor(period=1.0, intensity=0.4)
        self.assertAlmostEqual(scale, 2.40, delta=0.1)

    def test_get_peak_accel(self):
        """it should try to find it in the config file or compute it"""
        record_abspath = str((RECORDS_DIR / "98_SCT190985NS.csv").resolve())
        rec = Record(path=record_abspath)
        self.assertAlmostEqual(rec.pfa, 103.31, 0)

        record_abspath = str((RECORDS_DIR / "43_TL140995NS.csv").resolve())
        rec = Record(path=record_abspath)
        self.assertAlmostEqual(rec.pfa, 30.21, 0)
