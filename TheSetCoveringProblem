#!/usr/bin/env python
# coding: utf-8

# # The Set Covering Problem
# 
# ## Beyza Erdoğan
# 
# ### November 28, 2022

# In[4]:


# load libraries
import numpy as np
import scipy.sparse as sp
import cplex as cp


# In[26]:


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


# In[33]:


def set_covering_problem(flights_file, costs_file, K):
    
    # number of decision variables = number of routes = N
    # number of constraints = number of legs + 1

    costs = np.loadtxt("costs.txt")
    flights = np.loadtxt("flights.txt")

    N = costs.shape[0]                   # number of routes
    M = np.max(flights[:,0]).astype(int) # number of legs
    
    # sort the data
    costs = costs[np.argsort(costs[:, 0])]
    flights = flights[np.argsort(flights[:, 1])]
    flights = flights[np.argsort(flights[:, 0])]

    c = costs[:, 1]
    b = np.concatenate((np.repeat(1, M), [K]))
    senses = np.concatenate((np.repeat("G", M), ["E"]))
    types = np.repeat("B", N)
    l = np.repeat(0, N)
    u = np.repeat(1, N)

    aij = np.concatenate((np.repeat(1, flights.shape[0]), np.repeat(1, N)))
    row = np.concatenate((flights[:, 0].astype(int) - 1, np.repeat(M, N)))
    col = np.concatenate((flights[:, 1].astype(int) - 1, range(N)))
    A = sp.csr_matrix((aij, (row, col)), shape = (M + 1, N))
    
    x_star, obj_star = mixed_integer_linear_programming("minimize", A, senses, b, c, l, u, types)
    return (x_star, obj_star)


# In[36]:


x_star, obj_star = set_covering_problem("flights.txt", "costs.txt", 3)
print(x_star)
print(obj_star)

