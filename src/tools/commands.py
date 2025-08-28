import pandas as pd
import numpy as np
import statsmodels.tsa.stattools
from statsmodels.tsa.arima.model import ARIMA
from . import types_
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
# Lets define all our commands

# First Line Commands
def line(data,*args):
    if isinstance(data,types_.Lines):
        data = data.get_line()
        return data
    if isinstance(data,types_.Points):
        data = data.get_points()
    if isinstance(data,types_.Distribution):
        data = data.get_distribution()

    return types_.Lines(data[list(args)].apply(pd.to_numeric, errors="coerce").dropna())

def smooth(line: types_.Lines,window,column):
    window = int(window)
    df = line.get_line().copy()
    df[column]=np.convolve(df[column],np.ones(window)/window,mode='same')
    return types_.Lines(df)

def trend(line: types_.Lines,window,column):
    window = int(window)
    # Literally the same thing as smoothing
    return smooth(line,window,column)

def seasonal(line:types_.Lines,window,column):
    window = int(window)
    df = line.get_line().copy()
    df[column] = df[column]-np.convolve(df[column],np.ones(window)/window,mode='same')

    return types_.Lines(df)

def acf(line: types_.Lines,nlags,column):
    nlags = int(nlags)
    df = line.get_line()
    auto_vals = statsmodels.tsa.stattools.acf(df[column],nlags=nlags)
    new_dt = {"Lags":[i for i in range(11)],"ACF":auto_vals}
    new_df = pd.DataFrame(new_dt)

    return types_.Lines(new_df)

def pacf(line: types_.Lines,nlags,column):
    nlags = int(nlags)
    df = line.get_line()
    auto_vals = statsmodels.tsa.stattools.pacf(df[column],nlags=nlags)
    new_dt = {"Lags":[i for i in range(11)],"PACF":auto_vals}
    new_df = pd.DataFrame(new_dt)

    return types_.Lines(new_df)

def forecast(line: types_.Lines,p,d,q,future_steps,column):
    p=int(p)
    d=int(d)
    q=int(q)
    future_steps=int(future_steps)
    df = line.get_line()
    model = ARIMA(df['column'],order=(p,d,q))
    model_fit = model.fit()

    return types_.Lines(pd.concat(df,pd.DataFrame({column: model_fit.forecast(future_steps)}),axis=0))

# Now for points
def points(data,*columns):
    if isinstance(data,types_.Lines):
        data = data.get_line()
    if isinstance(data,types_.Points):
        data = data.get_points()
    if isinstance(data,types_.Distribution):
        data = data.get_distribution()

    return types_.Points(data[list(columns)].apply(pd.to_numeric, errors="coerce").dropna())

def correlation(points: types_.Points, columnx, columny):
    df = points.get_points()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[columnx],df[columny])
    
    return r_value

def pca(points: types_.Points,n_components,*columns):
    n_components=int(n_components)
    # First scale the data
    df = points.get_points()[list(columns)]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    pca_result = pd.DataFrame(pca_result,columns=[f"PCA: {i}" for i in range(pca_result.shape[1])])

    return types_.Points(pca_result)

# And for Distributions
def dist(data,column):
    if isinstance(data,types_.Lines):
        data = data.get_line()
    if isinstance(data,types_.Points):
        data = data.get_points()
    if isinstance(data,types_.Distribution):
        data = data.get_distribution()
        return data
     
    return types_.Distribution(data[column].apply(pd.to_numeric, errors="coerce").dropna())

def kl_divergence(dist: types_.Distribution,dist2: types_.Distribution,bins):

    bins=int(bins)

    df1 = dist.get_distribution()
    df2 = dist2.get_distribution()


    density_A, bin_edges = np.histogram(df1, bins=int(bins), density=True)
    density_B, _ = np.histogram(df2, bins=bin_edges, density=True)

    dx = bin_edges[1]-bin_edges[0]

    prob_A = density_A*dx
    prob_B = density_B*dx

    # Compute KL divergence
    kl_div = stats.entropy(prob_A, prob_B)  # KL(A || B)
    
    return kl_div

def entropy(dist: types_.Distribution,bins):
    df = dist.get_distribution()

    density_A, bin_edges = np.histogram(df, bins=int(bins), density=True)
    dx = bin_edges[1]-bin_edges[0]

    prob_A = density_A*dx

    entropy = stats.entropy(prob_A)

    return entropy

def mean(dist: types_.Distribution,column):
    return dist.get_distribution()[column].mean()


def std(dist: types_.Distribution,column):
    return dist.get_distribution()[column].std()

def plot(data, *args):
    if isinstance(data, types_.Lines):
        df = data.get_line()
        columns = df.columns
        if len(columns) != 2:
            raise Exception("Precisely two dimensions required for plotting")
        
        fig, ax = plt.subplots()
        ax.plot(df[columns[0]], df[columns[1]])
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        st.pyplot(fig)

    if isinstance(data, types_.Points):
        df = data.get_points()
        columns = df.columns
        if len(columns) != 2:
            raise Exception("Precisely two dimensions required for plotting")

        fig, ax = plt.subplots()
        ax.scatter(df[columns[0]], df[columns[1]])
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        st.pyplot(fig)

    if isinstance(data, types_.Distribution):
        
        counts, bins = np.histogram(data.get_distribution(), bins=int(args[0]))

        fig, ax = plt.subplots()
        ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge", edgecolor="black")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Counts")
        st.pyplot(fig)

    return None
