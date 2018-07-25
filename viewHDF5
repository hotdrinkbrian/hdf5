import numpy as np
import h5py
"""
fn="deeph-test-v1-resort.h5"

f = h5py.File(fn,'r')

for key in f.keys():
    print(key)

print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
data = list(f[a_group_key])


print data    
"""
   
import pandas

input_filename = "deeph-test-v1-resort.h5"
store = pandas.HDFStore(input_filename)

# Read the first 10 events
store.select("table",stop=10)

print store['table']  
