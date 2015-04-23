import numpy as np

var1 = np.arange(9)

print var1

for x in np.nditer(var1, op_flags=['readwrite']):
  x = x + 1

print var1
