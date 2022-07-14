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

# DOWNLOAD (& ADD TO BIG ARRAY)

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