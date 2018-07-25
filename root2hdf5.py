#######################################
# Imports
########################################

print "Imports: Starting..."

import sys
if len(sys.argv) != 4:
    print "Enter signal_file background_file n_constituents as arguments"
    sys.exit()
    
import os
import pickle
import pdb

print "Imported basics"

import ROOT
print "Imported ROOT"

import numpy as np
import pandas
import root_numpy
import h5py

import time

print "Imports: Done..."

##doesn't work if input files large - too much memory

########################################
# Configuration
########################################

"""
infname_sig = sys.argv[1]
infname_bkg = sys.argv[2]
n_cands = int(sys.argv[3])
"""
infname_sig = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'
infname_bkg = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'#'QCD_HT100To200_pfc_1j_skimed.root'
n_cands = 40
version = "v0_{0}nc".format(n_cands)


## load data
start = time.time()
df_sig = pandas.DataFrame(root_numpy.root2array(infname_sig, treename="tree44")) #brian
df_bkg = pandas.DataFrame(root_numpy.root2array(infname_bkg, treename="tree44")) #brian 

#-------------------------<
df_sig['is_signal'] = 1
df_bkg['is_signal'] = 0
#-------------------------<

df_all = pandas.concat([df_sig, df_bkg], ignore_index=True)
print time.time()-start, "seconds to load data"


## shuffle data
start = time.time()
df_all = df_all.iloc[np.random.permutation(len(df_all))].reset_index(drop=True)
print time.time()-start, "seconds to shuffle data"
    
open_time = time.time()

df = pandas.DataFrame()



print df_all

#df["truthE"]        = df_all["truth_e"  ]
#df["truthPX"]       = df_all["truth_px" ]
#df["truthPY"]       = df_all["truth_py" ]
#df["truthPZ"]       = df_all["truth_pz" ]
df["is_signal_new"] = df_all["is_signal"]
#jj = "Jets1."
for i in range(n_cands):     
    df["E_{0}".format(i) ] = df_all["Jet{0}s_pfc{1}_energy".format(1, i+1) ]
    df["PX_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_px".format(1, i+1) ]
    df["PY_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_py".format(1, i+1) ]
    df["PZ_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_pz".format(1, i+1) ]

"""       
    df["E_{0}".format(i) ] = [x[i] for x in df_all["e" ]]
    df["PX_{0}".format(i)] = [x[i] for x in df_all["px"]]
    df["PY_{0}".format(i)] = [x[i] for x in df_all["py"]]
    df["PZ_{0}".format(i)] = [x[i] for x in df_all["pz"]]
"""




# Train / Test / Validate
# ttv==0: 60% Train
# ttv==1: 20% Test
# ttv==2: 20% Final Validation
df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[0.6, 0.2, 0.2])

train = df[ df["ttv"]==0 ]
test  = df[ df["ttv"]==1 ]
val   = df[ df["ttv"]==2 ]
        
print len(df), len(train), len(test), len(val)

train.to_hdf('vbf+qcd-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf('vbf+qcd-test-{0}.h5'.format(version),'table',append=True)
val.to_hdf('vbf+qcd-val-{0}.h5'.format(version),'table',append=True)

close_time = time.time()

print "Time for the lot: ", (close_time - open_time)
