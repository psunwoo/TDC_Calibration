# streamlit_app.py

!pip install steamlit
import os
import sys
import importlib

# To import from other py files
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.append(current_dir)

import TDL_Class
importlib.reload(TDL_Class)


from TDL_Class import *
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("🔍 TDL Partial Order Reconstruction GUI")

# --- Session state to store the TDL instance ---
if "tdl" not in st.session_state:
    st.session_state.tdl = TDL()

# --- File upload for physical bin data ---
st.sidebar.header("📥 Load Trial Data")
file = st.sidebar.file_uploader("Upload a .npy or .txt file containing physical bins", type=["npy", "txt"])

if file:
    if file.name.endswith(".npy"):
        physical_bins = np.load(file)
    else:
        physical_bins = np.loadtxt(file, dtype=int)
    st.session_state.physical_bins = physical_bins
    st.success("Data loaded successfully!")

    st.write("### Raw Tapped Physical Bins:")
    st.code(physical_bins)

    trial_num = st.number_input("Select Trial Number", min_value=0, value=st.session_state.tdl.trial)
    run_feed = st.button("✅ Feed Physical Bins into TDL")

    if run_feed:
        st.session_state.tdl.feed_physical_nums(physical_bins, trial_num)
        st.success(f"Trial {trial_num} data fed!")

        st.write("### Current TDL State")
        st.json({
            "Trial": st.session_state.tdl.trial,
            "Start Bin": st.session_state.tdl.start_num,
            "End Bin": st.session_state.tdl.end_num,
            "Cells": list(st.session_state.tdl.cells.keys())
        })

# --- Perform POR ---
if st.button("🚀 Perform Initial POR"):
    result = st.session_state.tdl.perform_POR()
    st.write("### Logical Assignment Output:")
    st.code(result)

# --- Manual POR override ---
st.write("---")
st.subheader("✍️ Manual POR Override (for disabled cells)")
manual_cell = st.number_input("Cell Number to Override", min_value=1, step=1)
override_string = st.text_input("Enter POR order (e.g., 21347658):")
if st.button("Submit Manual POR"):
    try:
        override_result = list(map(int, list(override_string.strip())))
        st.session_state.tdl.cells[manual_cell].disabled = False
        st.session_state.tdl.POR_result[st.session_state.tdl.cells[manual_cell].get_tapped_history()] = override_result
        st.success(f"Manual POR for Cell {manual_cell} saved.")
    except Exception as e:
        st.error(f"❌ Failed: {e}")

# --- Visualizations (planned) ---
# Use NetworkX or matplotlib in future for DAGs, POR comparisons, history tracking
