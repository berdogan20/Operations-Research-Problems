#!/usr/bin/env python
# coding: utf-8

# In[3]:


# load libraries
import numpy as np
import scipy.sparse as sp
import cplex as cp


# In[4]:


def mixed_integer_linear_programming(direction, A, senses, b, c, l, u, types):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), types = types.tolist())

    # define problem type
    if direction == "maximize":
        prob.objective.set_sense(prob.objective.sense.maximize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)

    # add constraints to the problem including their directions and right-hand side values
    prob.linear_constraints.add(senses = senses.tolist(), rhs = b.tolist())

    # add coefficients for each constraint
    row_indices, col_indices = A.nonzero()
    prob.linear_constraints.set_coefficients(zip(row_indices.tolist(), col_indices.tolist(), A.data.tolist()))

    # solve the problem
    print(prob.write_as_string())
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)


# In[7]:


def coin_distribution_problem(coins_file, M):
    coins = np.loadtxt(coins_file)

    N = coins.shape[0] # number of coins

    # number of decision variables = number of coins * number of children
    E = M * N
    # number of constraints = number of coins + number of children
    V = M + N
    # money per child = total money // number of chilren
    P = np.sum(coins) / M
    print(P)

    c = np.repeat(1, E)
    b = np.concatenate((np.repeat(P, M), np.repeat(1, N)))
    l = np.repeat(0, E)
    u = np.repeat(1, E)
    senses = np.repeat("E", V)
    types = np.repeat("B", E)


    aij = np.concatenate((np.tile(coins, M), np.repeat(1, E)))
    row = np.concatenate((np.repeat(range(M), N), M + np.repeat(range(N), M)))
    col = np.concatenate((np.array(range(E)).reshape(N, M).T.flatten(), range(E)))
    A = sp.csr_matrix((aij, (row, col)), shape = (V, E))

    X_star, obj_star = mixed_integer_linear_programming("maximize", A, senses, b, c, l, u, types)
    return(np.array(X_star).reshape(N, M))


# In[8]:


X_star = coin_distribution_problem("coins.txt", 2)
print(X_star)


# In[ ]:




