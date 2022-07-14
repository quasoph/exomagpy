# IMPORTS

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import lightkurve as lk
import io
import warnings

from tensorflow import keras

warnings.filterwarnings("ignore")

def download(search):

    lc = search.download()
    
    if lc is not None:
        
        fig,ax = plt.subplots()
        ax.scatter(lc.time.value.tolist(), lc.flux.value.tolist(),s=0.1, color='k')
        ax.autoscale()
        ax.set_xlabel('Time (BTJD)')
        ax.set_ylabel('Flux')
        plt.close(fig)
        io_buf = io.BytesIO()
        fig.savefig(io_buf,format="raw")
        io_buf.seek(0)
        img_arr = (np.frombuffer(io_buf.getvalue(),dtype=np.uint8)).reshape(288,-1)
        io_buf.close()

        return img_arr

def get_lightcurves(filename,length):

    tbl = pd.read_csv(os.path.abspath(filename),delimiter=",",comment="#")
    
    colnames = tbl.columns.values.tolist()
    if "tid" in colnames:
        TICs = tbl["tid"].astype(str)
    elif "tic_id" in colnames:
        TICs = tbl["tic_id"].astype(str).str[4:]
    else:
        print("No TIC ID column found.")

    #print(np.shape(TICs))

    pics = []

    for x in range(0,length): # change upper bound as needed
        name = TICs[x]
        
        search = lk.search_lightcurve(target=("TIC " + name),author="SPOC")
        pic = download(search)
        
        if pic is not None:
            pics.append(pic)
    
    shape = int(len(pics))
        
    return pics, shape