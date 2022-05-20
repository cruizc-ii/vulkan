import streamlit as st
from app.design import ReinforcedConcreteFrame
from app.utils import find_files
from pathlib import Path

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DESIGN_DIR = MODELS_DIR / "design"

buildings = find_files(DESIGN_DIR)
buildings
with st.sidebar:
    left, right = st.columns(2)
    prev = left.button("back")
    next = right.button("next")
    if prev:
        "success"
    st.header("sidebar")
    file = st.selectbox("select a building", options=buildings)
    design = ReinforcedConcreteFrame.from_file(DESIGN_DIR / file)
    design
    design.to_json
