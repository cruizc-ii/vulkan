import streamlit as st
from app.design import ReinforcedConcreteFrame
from app.utils import find_files
from pathlib import Path
from st_cytoscape import cytoscape
import time
import shutil
from app.utils import ROOT_DIR, DESIGN_DIR, MODELS_DIR


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


padding_top = 2

st.markdown(
    f"""
    <style>
        .appview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)  # remove unnecessary padding at top

if "module" not in st.session_state:
    st.session_state.module = 1


def switch_module(delta: int):
    st.session_state.module = st.session_state.module + delta


title = (
    "design",
    "hazard",
    "structural analysis",
    "loss",
)


with st.sidebar:
    st.header(title[st.session_state.module - 1])
    if st.session_state.module == 1:
        buildings = find_files(DESIGN_DIR)
        file = st.selectbox("select a building", options=buildings)
        if file:
            design = ReinforcedConcreteFrame.from_file(DESIGN_DIR / file)
        else:
            design = ReinforcedConcreteFrame(
                name="default",
                storeys=[3.5, 3.0],
                bays=[5.0],
            )

        name = st.text_input(
            "give it a name",
            value=file or "",
        )
        storeys_input = st.text_input(
            "storeys",
            value=",".join([str(s) for s in design.storeys]) or "3",
            help="heights in meters separated by comma",
        )
        storeys = [float(s) for s in storeys_input.split(",")]
        bays_input = st.text_input(
            "bays",
            value=",".join([str(s) for s in design.bays]) or "5",
            help="widths in meters separated by comma",
        )
        bay = [float(s) for s in bays_input.split(",")]

        left, right = st.columns(2)
        delete = left.button("🗑️", help="delete this building")
        if delete:
            with st.spinner("deleting building"):
                time.sleep(2)
                design.delete(DESIGN_DIR)
                design = None
                st.success("design successful")
        run_design = right.button("design", help="run a design")
        params_missing = not name
        if run_design and params_missing:
            st.error("params missing")
        elif run_design:
            with st.spinner("running design"):
                design = ReinforcedConcreteFrame(
                    name=name,
                    storeys=storeys,
                    bays=bay,
                )
                time.sleep(2)
                design.to_file(DESIGN_DIR)
            st.success("design successful")
            design.to_json

        # elems, style = design.fem.cytoscape()

left, right = st.columns(2)
if st.session_state.module != 1:
    prev = left.button("back", on_click=switch_module, args=[-1])
if st.session_state.module != 4:
    next = right.button("next", on_click=switch_module, args=[1])

layout = {"name": "preset"}  # this layout respects nodes' (x,y)
# selected = cytoscape(elems, style, key="graph", layout=layout)
design.to_json

# selected = cytoscape(
#     elements=elements,
#     stylesheet=stylesheet,
#     layout=layout,
#     user_panning_enabled=False,
#     user_zooming_enabled=False,
# )
