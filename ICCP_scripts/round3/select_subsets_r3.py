import numpy as np 
from matplotlib import pyplot as plt 


invert = False

#np.random.seed(0)

subsets = [
    [16, 22, 15, 21], 
    [22, 28, 21, 27], 
    [16, 22, 28, 15, 21, 27, 14, 20, 26], 
    [17, 23, 29, 16, 22, 28, 15, 21, 27], 
    [22, 21, 20, 28, 27, 26, 34, 33, 32], 
    [10, 8, 6, 22, 20, 18, 34, 30, 30], 
    [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,24, 25, 26, 27, 28, 30, 31, 32, 33, 34],
    [13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 49, 40, 41], 
    [15, 17, 22, 27, 29], 
    [22, 20, 27, 32, 34], 
    [11, 7, 22, 35, 31], 
    [22, 19, 40, 37], 
    [22, 19, 33, 40, 37], 
    [22, 20, 32, 34], 
    [22, 40, 19, 27], 
    [22, 20, 18, 34, 32, 30, 46, 44, 42], 
    [16, 22, 28, 14, 16]
    #[22, 20, 27, 34, 32]
]


if invert:
    subsets = subsets[::-1]

for subset in subsets:
    assert 22 in subset

# not an efficient way to do it but its fine 
duplicated = []
for i, j in np.ndindex((len(subsets), len(subsets))): 
    if i == j:
        continue 

    s1 = subsets[i] 
    s2 = subsets[j] 
    if len(s1) != len(s2):
        continue 
    intersect = np.intersect1d(s1, s2) 
    if len(intersect) == len(s1):
        duplicated.append(s1) 