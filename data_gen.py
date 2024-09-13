import numpy as np
from typing import Callable

# Given a function, use it to generate a data set
# To do: add "censoring" so that some of the input range is not covered
def regression_data_gen(d_in: int, d_out: int, 
             f: Callable[[np.ndarray], np.ndarray], 
             n: int, 
             stdev: float = 0.0,
             x_min: float = -1.0, x_max: float = 1.0):
    # Sorting X to make plotting easier later
    X = np.sort(np.random.uniform(x_min, x_max, (n, d_in)), axis = 0)
    Y = f(X) + np.random.normal(0, stdev, (n, d_out))
    return X, Y

