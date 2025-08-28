import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parsing.parser import parse
from tools.command_ast_evaluator import AstEvaluator
from tools.command_builder import Builder
# Main File detailing UI

st.title("Blazingly Fast Data Analytics🔥🔥")

file = st.file_uploader("Upload CSV Here")

if file is not None:
    input = st.text_input("Enter analytic command")
    df = pd.read_csv(file)
    ase = AstEvaluator(Builder(df))
    if len(input)>0:
        try:
            ast = parse(input)
        except Exception:
            st.text('Error parsing values for AST, make sure your inputted command syntax is correct')
            
        try:
            result = ase.ast_evaluator_main(ast)
        except:
            st.text('Error completing command, double check input fields')

        if result != None:
            st.markdown('### Result: '+str(result))
    else:
        table = st.table(df.head())
