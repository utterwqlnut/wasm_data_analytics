import pandas as pd
import numpy as np
import statsmodels.tsa.stattools
from scipy import stats
import types_
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Lets define all our commands

# First Line Commands

def smooth(line: types_.Lines,window,column):
    df = line.get_line().copy()
    df[column]=np.convolve(df[column],np.ones(window)/window,mode='same')
    return types_.Lines(df)

def trend(line: types_.Lines,window,column):
    # Literally the same thing as smoothing
    return smooth(line,window,column)

def seasonal(line:types_.Lines,window,column):
    df = line.get_line().copy()
    df[column] = df[column]-np.convolve(df[column],np.ones(window)/window,mode='same')

    return types_.Lines(df)

def acf(line: types_.Lines,nlags,column):
    df = line.get_line()
    auto_vals = statsmodels.tsa.stattools.acf(df[column],nlags=nlags)
    new_dt = {"Lags":[i for i in range(11)],"ACF":auto_vals}
    new_df = pd.DataFrame(new_dt)

    return types_.Lines(new_df)

def pacf(line: types_.Lines,nlags,column):
    df = line.get_line()
    auto_vals = statsmodels.tsa.stattools.pacf(df[column],nlags=nlags)
    new_dt = {"Lags":[i for i in range(11)],"PACF":auto_vals}
    new_df = pd.DataFrame(new_dt)

    return types_.Lines(new_df)

def forecast(line: types_.Lines,p,d,q,column,future_steps):
    df = line.get_line()
    model = statsmodels.tsa.arima.model.ARIMA(df['column'],order=(p,d,q))
    model_fit = model.fit()

    return types_.Lines(pd.concat(df,pd.DataFrame({column: model_fit.forecast(future_steps)}),axis=0))

# Now for points

def correlation(points: types_.Points, columnx, columny):
    df = points.get_points()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[columnx],df[columny])
    
    return r_value

def pca(points: types_.Points,n_components,*columns):
    # First scale the data
    df = points.get_points()[columns]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    return types_.Points(pca_result)

# Now for Distributions

def kl_divergence(dist: types_.Distribution,column1,column2,bins):
    df = dist.get_dist()


    density_A, bin_edges = np.histogram(df[column1], bins=bins, density=True)
    density_B, _ = np.histogram(df[column2], bins=bin_edges, density=True)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    hist_A += epsilon
    hist_B += epsilon

    # Compute KL divergence
    kl_div = stats.entropy(density_A, density_B)  # KL(A || B)
    
    return kl_div

def entropy(dist: types_.Distribution,column,bins):
    df = dist.get_dist()

    density_A, bin_edges = np.histogram(df[column], bins=bins, density=True)

    entropy = stats.entropy(density_A)

    return entropy

def mean(dist: types_.Distribution,column):
    return dist.get_dist()[column].mean()


def std(dist: types_.Distribution,column):
    return dist.get_dist()[column].std()


