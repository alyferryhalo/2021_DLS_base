def generate_batches(X, y, batch_size):
    assert len(X) == len (y)
    np.random.seed(42)
    
    X = np.array(X)
    y = np.array(y)
    
    perm = np.random.permutation(len(X))
    num_of_batches = int(len(X) // batch_size)
    
    for i in range(0,len(X) // batch_size ):
          
        perm_start  = i * batch_size
        perm_end = perm_start + batch_size
        p2 = perm[perm_start :  perm_end]
       
        x_give = X[p2]
        y_give = y[p2]

        yield(x_give, y_give)
