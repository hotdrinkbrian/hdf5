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
from ROOT import TDirectory, TFile, gFile, TBranch, TTree
print "Imported ROOT"

import numpy as np
import pandas as pd
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
"""
infname_sig = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'
infname_bkg = 'QCD_HT100To200_pfc_1j_skimed.root'#'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'#'QCD_HT100To200_pfc_1j_skimed.root'
"""
n_cands = 40
version = "v0_{0}cs".format(n_cands)


## load data
start = time.time()
"""
df_sig = pd.DataFrame(root_numpy.root2array(infname_sig, treename="tree44")) #brian
df_bkg = pd.DataFrame(root_numpy.root2array(infname_bkg, treename="tree44")) #brian 

#-------------------------<
df_sig['is_signal'] = 1
df_bkg['is_signal'] = 0
#-------------------------<
"""
xs = { '50To100': 246300000 , '100To200': 28060000 , '200To300': 1710000 , '300To500': 347500, 'sgn': 3.782 }
train_ratio = 0.6
test_ratio  = 0.2
val_ratio   = 0.2

bkg_multiple = 80
bkg_test_multiple = 1#100

random_seed = 4444
np.random.seed(random_seed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#path = '/nfs/dust/cms/user/hezhiyua/table/LLP/data/data_lola/'
path = '/beegfs/desy/user/hezhiyua/pfc_test/raw_data/Skim/data/data_lola/'
#path = '/beegfs/desy/user/hezhiyua/pfc_test/raw_data/Skim/data_test/'
bkg_name_dict = {}
bkg_name_dict['50To100' ] = 'QCD_HT50To100_pfc_1j_skimed.root'
bkg_name_dict['100To200'] = 'QCD_HT100To200_pfc_1j_skimed.root'
bkg_name_dict['200To300'] = 'QCD_HT200To300_pfc_1j_skimed.root'
bkg_name_dict['300To500'] = 'QCD_HT300To500_pfc_1j_skimed.root'
sgn_name                  = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_pfc_t1_1j_skimed.root'

for w in bkg_name_dict:
    bkg_name_dict[w] = path + bkg_name_dict[w]
sgn_name = path + sgn_name
#from collections import OrderedDict
#log = OrderedDict()

##load background#############################################################################################
df_bkg_dict = {}
N_bkg_dict = {}
N_available_bkg = 0
for w in bkg_name_dict:
    fb = TFile(bkg_name_dict[w],"r")
    print 'loading tree(for bkg) ....'
    tree_bkg = fb.Get('tree44')
    nevents = tree_bkg.GetEntries()
    N_bkg_dict[w] = nevents
    print 'tree loaded(for bkg) ....'
    #set up DataFrames
    print 'loading data(for bkg) ....'
    #df_bkg_dict[w] = pd.DataFrame(root_numpy.tree2array(tree_bkg, branches = lni))
    df_bkg_dict[w] = pd.DataFrame(root_numpy.root2array(bkg_name_dict[w], treename="tree44"))
    print 'data completely loaded(for bkg)'
    #rename columns
    #df_bkg_dict[w].columns      = lno
    #set up labels
    df_bkg_dict[w]['is_signal'] = 0
    #drop events with values = -1 
    #bkg_dropone = df_bkg_dict[w]['Jet1.pt'] != -1
    #df_bkg_dict[w] = df_bkg_dict[w][:][bkg_dropone]
    N_available_bkg = N_available_bkg + nevents#len( df_bkg_dict[w] )
#available background
print '#available background:'
print N_available_bkg
##load background#############################################################################################
##load signal############################################################
fs = TFile(sgn_name,"r")
print 'loading tree(for signal) ....'
tree_sig = fs.Get('tree44')
nevents = tree_sig.GetEntries()
N_available_sgn = nevents #len(df_sig)
print 'tree loaded(for signal) ....'
#set up DataFrames
print 'loading data(for signal) ....'
#df_sig = pd.DataFrame(root_numpy.tree2array(tree_sig, branches = lni))
df_sig = pd.DataFrame(root_numpy.root2array(sgn_name, treename="tree44"))
print 'data completely loaded(for signal)'
#rename columns
#df_sig.columns      = lno
#set up labels
df_sig['is_signal'] = 1
#drop events with values = -1
#sig_dropone = df_sig['Jet1.pt'] != -1
#df_sig = df_sig[:][sig_dropone]
#available signal
print '#available signal:'
print N_available_sgn
##load signal############################################################

#~~~~~~~~~~~~~~~~initialize data setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N_bkg_to_use_dict = {}
N_bkg_to_train_dict = {}
if N_available_bkg >= bkg_test_multiple * N_available_sgn:
    N_test_sgn_to_use = int( N_available_sgn * (1-train_ratio) )
    N_sgn_to_use      = N_available_sgn - N_test_sgn_to_use             #for training 
    N_test_bkg_to_use = int( N_test_sgn_to_use * bkg_test_multiple )
    N_bkg_to_use      = int( N_available_sgn * bkg_multiple )           #for training    

    df_sig['weight']    = 1 / float( N_sgn_to_use ) #add weight column
    for w in bkg_name_dict:
        N_bkg_to_use_dict[w]    = int(  N_bkg_dict[w] * ( (N_bkg_to_use + N_test_bkg_to_use)/float(N_available_bkg) )  )
        N_bkg_to_train_dict[w]  = int(  N_bkg_dict[w] * ( (N_bkg_to_use)/float(N_available_bkg) )  )
        df_bkg_dict[w]['weight']    = xs[w] / float( N_bkg_to_train_dict[w] * ( xs['50To100']+xs['100To200']+xs['200To300']+xs['300To500'] ) ) #add weight column
    
    df_bkg = pd.concat([ df_bkg_dict['50To100' ][:N_bkg_to_use_dict['50To100' ]] , \
                         df_bkg_dict['100To200'][:N_bkg_to_use_dict['100To200']] , \
                         df_bkg_dict['200To300'][:N_bkg_to_use_dict['200To300']] , \
                         df_bkg_dict['300To500'][:N_bkg_to_use_dict['300To500']]   ], ignore_index=True)
    df_bkg = df_bkg.iloc[np.random.permutation(len(df_bkg))]

    #set up data split
    #df_test_sig = df_sig[:N_test_sgn_to_use]
    #df_sig      = df_sig[len(df_test_sig):]
    #df_test_bkg = df_bkg[:N_test_bkg_to_use]
    #df_bkg      = df_bkg[len(df_test_bkg):(  len(df_test_bkg)+1  +  N_bkg_to_use  )]
else:
    print "data not enough~~"
"""
else:
    print 'background data not enough!'    #to be checked!!
    df_test_sig = df_sig[:N_test_sgn_to_use]
    df_sig      = df_sig[len(df_test_sig):]
    df_test_bkg = df_bkg[:int( len(df_test_sig) * 1 )]
    df_bkg      = df_bkg[len(df_test_bkg):]
"""
#~~~~~~~~~~~~~~~~initialize data setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#print df_sig
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



















df_all = pd.concat([df_sig, df_bkg], ignore_index=True)
print time.time()-start, "seconds to load data"


## shuffle data
start = time.time()
df_all = df_all.iloc[np.random.permutation(len(df_all))].reset_index(drop=True)
print time.time()-start, "seconds to shuffle data"
    
open_time = time.time()

df = pd.DataFrame()



print df_all

#df["truthE"]        = df_all["truth_e"  ]
#df["truthPX"]       = df_all["truth_px" ]
#df["truthPY"]       = df_all["truth_py" ]
#df["truthPZ"]       = df_all["truth_pz" ]
#df["is_signal_new"] = df_all["is_signal"]
#jj = "Jets1."
for i in range(n_cands):     
    df["E_{0}".format(i) ] = df_all["Jet{0}s_pfc{1}_energy".format(1, i+1) ]
    df["PX_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_px".format(1, i+1) ]
    df["PY_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_py".format(1, i+1) ]
    df["PZ_{0}".format(i)] = df_all["Jet{0}s_pfc{1}_pz".format(1, i+1) ]
df["is_signal_new"] = df_all["is_signal"]
df["weight"]        = df_all["weight"]
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
df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[train_ratio, test_ratio, val_ratio])

train = df[ df["ttv"]==0 ]
test  = df[ df["ttv"]==1 ]
val   = df[ df["ttv"]==2 ]
        
print len(df), len(train), len(test), len(val)

train.to_hdf('vbf_qcd-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf('vbf_qcd-test-{0}.h5'.format(version),'table',append=True)
val.to_hdf('vbf_qcd-val-{0}.h5'.format(version),'table',append=True)

close_time = time.time()

print "Time for the lot: ", (close_time - open_time)
