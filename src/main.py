import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parsing.parser import parse

# Main File detailing UI

st.title("Blazingly Fast Data Analytics")

file = st.file_uploader("Upload CSV Here")

if file is not None:
    input = st.text_input("Enter analytic command")
    df = pd.read_csv(file)

    if len(input)>0:
        st.text(parse(input))
    else:
        table = st.table(df.head())

