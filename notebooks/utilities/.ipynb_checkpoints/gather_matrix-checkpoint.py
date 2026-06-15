import pandas as pd
from sklearn import metrics
from scipy.io import loadmat
from os.path import exists
from tqdm import tqdm
from scipy.linalg import toeplitz
import time
import scipy
import os
import matplotlib.colors as mcolors 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
#import cooler
#import cooltools
import glob
import scanpy as sc
import math

source_path = os.path.abspath("../utilities/")
sys.path.append(source_path)
import utils as ut
import matrix
import matrix_v2 


resolution = 1000000
sc_path = f"/nfs/turbo/umms-indikar/shared/projects/poreC/pipeline_outputs/higher_order/anndata/singlecell_mESC_{resolution}_features.h5ad"

singlecell_path = sc_path

print(f"{singlecell_path=}")


# population
start_time = time.time()  # Record the start time
bdata_main = sc.read_h5ad(singlecell_path)
end_time = time.time()  # Record the end time
print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
sc.logging.print_memory_usage()

bdata = bdata_main.copy()
try:
    matrix_v2.expand_and_normalize_anndata(bdata, oe_kr=False)
    bdata.write('/scratch/indikar_root/indikar1/jduhamel/pore_c/singlecell_mESC_1000000_features_en.h5ad')
    print("Done!")
except Exception as e:
    print("Error: ", e)
    raise
