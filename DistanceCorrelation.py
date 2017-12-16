import numpy as np
import pandas as pd

def dist(x, y):
  #1d only
  # Calculate the absolute value element-wise
  # Returns: An ndarray containing the absolute value of each element in `x`
  return np.abs(x[:, None] - y)

def d_n(x):
  d = dist(x, x)
  dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean() 
  return dn

def dcov_all(x, y):
  # Coerce type to numpy array if not already of that type.
  try: x.shape
  except AttributeError: x = np.array(x)
  try: y.shape
  except AttributeError: y = np.array(y)
  
  dnx = d_n(x)
  dny = d_n(y)
    
  denom = np.product(dnx.shape)
  dc = np.sqrt((dnx * dny).sum() / denom)
  dvx = np.sqrt((dnx**2).sum() / denom)
  dvy = np.sqrt((dny**2).sum() / denom)
  dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
  return dc, dr, dvx, dvy

def distance_correlation(x,y):
  return dcov_all(x,y)[1]