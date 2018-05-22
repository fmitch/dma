import numpy as np

lookup = {0b11100:1, 0b11101:1, 0b11110:1, 0b11111:1,
          0b11000:1, 0b11001:1, 0b11010:1, 0b11011:1,
          0b10100:1, 0b10101:1, 0b10110:1, 0b10111:1,
          0b10000:0, 0b10001:0, 0b10010:0, 0b10011:0,
          0b01100:0, 0b01101:0, 0b01110:0, 0b01111:1,
          0b01000:0, 0b01001:0, 0b01010:0, 0b01011:1,
          0b00100:0, 0b00101:0, 0b00110:0, 0b00111:1,
          0b00000:0, 0b00001:0, 0b00010:0, 0b00011:1}

def generate_data(seed, dim):
    """
    dim should be even. default is 14x14
    """
    if (dim % 2) == 1:
        arr = np.concatenate((np.ones(dim*dim/2 ,dtype=np.int_), 
            np.zeros(dim*dim/2+1, dtype=np.int_)))
    elif (dim % 2) == 0:
        arr = np.concatenate((np.ones(dim*dim/2,dtype=np.int_), 
            np.zeros(dim*dim/2, dtype=np.int_)))
    np.random.seed(seed)
    np.random.shuffle(arr)
    return arr

def find_majority(a):
    dim = int(a.shape[0]**.5)
    arr = np.reshape(a,(dim,dim))
    count = 0
    while count < dim * dim:
        count = count + 1
        old_arr = np.copy(arr)
        for m in range(0,dim):
            for n in range(0,dim):
                code = ( (arr[m][n]<<4) + (arr[(m-1)%dim][n]<<3) + 
                (arr[m][(n+1)%dim]<<2)+(arr[(m+1)%dim][n]<<1)+(arr[m][(n-1)%dim] ))
                arr[m][n] = lookup[code]
        if np.array_equal(old_arr, arr):
            break
    if np.count_nonzero(arr) > .9*dim*dim:
        return 1
    if np.count_nonzero(arr) < .1*dim*dim:
        return 0
    else:
        return 2

def get_results(points,dim):
    results = []
    for seed in range(0,points):
        arr = generate_data(seed, dim)
        res = find_majority(arr)
        results.append(res);
    return results
