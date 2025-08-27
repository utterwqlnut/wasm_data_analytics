import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Main File detailing UI

st.title("Blazingly Fast Data Analytics")

file = st.file_uploader("Upload CSV Here")

if file is not None:
    input = st.text_input("Enter analytic command")
    df = pd.read_csv(file)

    if len(input)>0:
        fig,ax = plt.subplots()
        ax.plot([i for i in range(1,12)])
        st.pyplot(fig)
    else:
        table = st.table(df.head())

