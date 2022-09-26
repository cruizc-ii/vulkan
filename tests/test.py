from pathlib import Path

TEST_ROOT = Path(__file__).parent
ROOT_DIR = TEST_ROOT.parent
API_PATH = ROOT_DIR / "app"
SDOF_API_PATH = API_PATH / "sdof/elastoplastic-sdof-api.tcl"
DESIGN_MODELS_PATH = TEST_ROOT.parent / "models" / "design"
DESIGN_FIXTURES_PATH = TEST_ROOT / "design_fixtures"
RESULTS_DIR = TEST_ROOT / "results"
FEM_FIXTURES_PATH = TEST_ROOT / "fem_fixtures"
RECORDS_DIR = TEST_ROOT.parent / "records"
HAZARD_FIXTURES_PATH = TEST_ROOT / "hazard_fixtures"
