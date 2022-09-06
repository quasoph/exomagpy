# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import io
import warnings
import requests
import lightkurve as lk
from astropy.io import fits

warnings.filterwarnings("ignore")

# DOWNLOAD (& ADD TO BIG ARRAY)

def lc_to_array(search):

    if type(search) == LightCurve:

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

    else:

        if search is not None: # where "search" is the returned file from download_mast()
        
            fig,ax = plt.subplots()

            # time = time column in csv file
            # flux = flux column in csv file (or similar)

            ax.scatter(lc.time.value.tolist(), lc.flux.value.tolist(),s=0.1, color='k') # lc time, flux etc will be different
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

# DOWNLOAD MAST JSON FILE AND PLOT LIGHTCURVE

def download_mast(row,col):

    hdu = fits.open("MiniDataset_calints.fits.gz")
    flux = hdu["SCI"].data
    unit = hdu["SCI"].header["BUNIT"]
    median_img = np.median(flux,0) # only needed for pixel plot
    pixel_lightcurve = flux[:,row,col]
    time = hdu["INT_TIMES"].data["int_mid_BJD_TDB"]

    # lightcurve plot

    plt.plot(time, pixel_lightcurve, marker=".", color="black")
    plt.xlabel('Time (BJD - 2400000.5)')
    plt.ylabel(f'Pixel ({col}, {row}) Flux\n[in units of {unit}]')


    return

# CONVERT MAST JSON FILE TO CSV

def mast_json2csv(json):    
    csv_str =  ",".join([x['name'] for x in json['fields']])
    csv_str += "\n"
    csv_str += ",".join([x['type'] for x in json['fields']])
    csv_str += "\n"
 
    col_names = [x['name'] for x in json['fields']]  
    for row in json['data']:
        csv_str += ",".join([str(row.get(col,"nul")) for col in col_names]) + "\n"
        
    return csv_str