import numpy as np 
from matplotlib import pyplot as plt 


invert = True

np.random.seed(0)
subsets = [
    [14, 15, 20, 21], 
    [20, 22, 32, 24], 
    [20, 8, 10, 22], 
    [20, 22, 31, 34],
    [19, 20, 21, 31, 33], 
    [14, 20, 26, 16, 28], 

    # a few more 4x4 and 5x5 configs 
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29], 
    [7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28], 
    [20, 21, 22, 23, 26, 27, 28, 29, 32, 33, 34, 35, 38, 39, 40, 41], 
    

    [21, 20, 26, 27, 0, 5, 42, 47], 
    [20, 27, 0, 47], 
    [20, 27, 5, 42], 
    [20, 5, 42], 
    [7, 20, 40], 
    [14, 20, 34],  
    [6, 34, 20], 
    [10, 20, 30], 
    [20, 10, 22], 
    [20, 10, 22, 31], 
    [20, 0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23, 24, 26, 28, 31, 33, 35, 36, 38, 40, 43, 45, 47],
    [0, 3, 7, 10, 14, 20, 23, 24, 27, 31, 40, 44, 47], 
    [20, 0, 5, 2, 27],
    [20, 13, 25, 7], 
    [20, 14, 26, 17, 29],
    [6, 8, 10, 18, 20, 22, 30, 32, 34], 
    [6, 10, 30, 34, 20],
    [18, 20, 23], 
    [18, 24, 20, 23, 29], 
    [8, 20, 28, 9, 39], 
    [8, 20, 28], 
    [20, 2, 44], 
    [2, 20, 32, 44],  
    [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34],
    [21, 27, 20, 26], 
    [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,24, 25, 26, 27, 28, 30, 31, 32, 33, 34],
    [21, 20, 26, 27, 13, 16, 31, 34], 
    [21, 20, 26, 27, 7, 10, 37, 40], 
    [19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47],
    [20, 21, 26, 27, 6, 10, 30, 34],
    [20, 13, 15, 25, 27],
    [18, 29, 20, 21, 22, 23], 
    [8, 14, 20, 26, 32]
 
]


if invert:
    subsets = subsets[::-1]

for subset in subsets:
    assert 20 in subset


# loop to add in random iterations
subsets.append(np.arange(48).tolist())
num = [40, 30, 20, 10]
count = [2, 2, 3, 4]
for n, c in zip(num, count): 
    for i in range(c):
        np.random.seed(i)
        array = np.random.choice(range(48), size=n, replace=False)
        if 20 not in array:
            array[0] = 20
        subsets.append(array.tolist())



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

#("duplicates", duplicated)
#print("count", len(subsets))



# for subset in subsets:
#     fig, axes = plt.subplots(6, 8)
#     for i, j in np.ndindex(axes.shape):
#         ax = axes[i, j] 
#         ax.set_xticks([])
#         ax.set_yticks([])
#         number = (5 - i) + 6 * j
        
#         if number in subset:
#             ax.set_facecolor("red") 
#         else:
#             ax.set_facecolor("black") 
#     plt.show()




# # with these, put reference camera first 
# # so it can be swapped out as necessary 

# subsets_3 = [
#     [20, 10, 37],
#     #[20, 10, 37], 
#     [20, 10, 40], 
#     [20, 7, 40],
#     [20, 22, 37], 
#     [20, 16, 31], 
#     [20, 13, 34],
#     [20, 9, 39], 
#     [20, 21, ]

# ]


# subsets_4 = [
#     [20, 10, 37, 40],
#     [20, 7, 37, 40], 
#     [20, 7, 40, 27],
#     [20, 22, 10, 37], 
#     [20, 13, 16, 31], 
#     [20, 16, 31, 34],

# ]


# subsets_5 = [
#     [20, 10, 37, 40, 7],
#     [20, 7, 37, 40, 27], 
#     [20, 7, 40, 27, 10],
#     [20, 22, 10, 37, 40], 
#     [20, 13, 16, 31, 34], 
#     [20, 16, 31, 34, 27],

# ]

