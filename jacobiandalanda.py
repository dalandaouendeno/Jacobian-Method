#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:52:54 2022

@author: dalandaouendeno
"""

import numpy as np
def jacobian(f, x, dx):
    """
    f: a vector of functions
    x: a n (>1) dimensional vector, where f(x) is evaluated.
    dx: a finite change in the given point
    method: use central-difference approximation by default;
            use forward-difference approximation if method=0
            use backward-difference approximation if method=1
    """
    n = len(x)
    jac = np.zeros([n,n])
    e = np.eye(n) * dx
    
    for i in range(n):
        for j in range(n):
            jac[i][j] = (f(x+e[j])[i] - f(x-e[j])[i]) / (2*dx)
    return jac