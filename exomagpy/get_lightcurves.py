# IMPORTS

import os
import pandas as pd
import lightkurve as lk
import warnings

warnings.filterwarnings("ignore")
from .download import download

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

def get_lightcurves_kep(filename,length):

    tbl = pd.read_csv(os.path.abspath(filename),delimiter=",",comment="#",chunksize=5)
    tbl.__next__()
    
    colnames = tbl.columns.values.tolist()
    if "kid" in colnames:
        KICs = tbl["kid"].astype("category")
    elif "kic_id" in colnames:
        KICs = (tbl["kic_id"].astype(str).str[4:]).astype("category")
    else:
        print("No KIC ID column found.")

    #print(np.shape(TICs))

    pics = []

    def namegen():
        for x in range(0,length): # change upper bound as needed
            yield str(KICs[x])
        
    name = namegen()
    for i in name:
        search = lk.search_lightcurve(target=("KIC " + i),author="Kepler")
        pic = download(search)
        
        if pic is not None:
            pics.append(pic)
    
    shape = int(len(pics))
        
    return pics, shape

def get_lightcurves_jwst(filename,length):

    # EDIT THIS TO RETRIEVE LIGHTCURVES FROM MAST PORTAL

    tbl = pd.read_csv(os.path.abspath(filename),delimiter=",",comment="#",chunksize=5)
    tbl.__next__()
    
    colnames = tbl.columns.values.tolist()
    if "kid" in colnames:
        KICs = tbl["kid"].astype("category")
    elif "kic_id" in colnames:
        KICs = (tbl["kic_id"].astype(str).str[4:]).astype("category")
    else:
        print("No KIC ID column found.")

    #print(np.shape(TICs))

    pics = []

    def namegen():
        for x in range(0,length): # change upper bound as needed
            yield str(KICs[x])
        
    name = namegen()
    for i in name:
        search = lk.search_lightcurve(target=("KIC " + i),author="Kepler")
        pic = download(search)
        
        if pic is not None:
            pics.append(pic)
    
    shape = int(len(pics))
        
    return pics, shape