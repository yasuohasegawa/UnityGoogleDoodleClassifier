import numpy as np
import sys

file = 'train'

# npy has extra 80bytes in the head. 
with open(file+".npy", "rb") as binary_file:
    data = binary_file.read()

width=28
height=28

total = 1000;
output = bytearray(total*(width*height))
outputindex = 0

for n in range(0,total):
    start = 80 + n * 784
    i = 0
    for x in range(0,width):
        for y in range(0,height):
            index = i+start
            val = data[index]
            
            output[outputindex] = val
            outputindex += 1
            i += 1

with open(file+str(total)+".bin", "wb") as fout:
    fout.write(output)