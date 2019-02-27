import os
import glob
"""
s

"""
""" 


"""

fpath = "bz_data/y2k/"
start = 7001


for filename in os.listdir(fpath):
    
   # if str(filenamse[-5]) == "1":
        os.rename(fpath + filename, fpath + str(start).zfill(6) + "_1.png")
        start += 1



    