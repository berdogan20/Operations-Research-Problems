#!/usr/bin/env python
# coding: utf-8

# #  The Distance Between Squares
# ## Beyza Erdoğan
# ### December 12, 2022

# \begin{align*}
# \mbox{minimize} \;\;&  \dfrac{1}{2} \sum\limits_{i=1}^{N} \sum\limits_{j=1}^{N} q_{ij} x_{i} x_{j} \\
# \mbox{subject to:} \;\;& \sum\limits_{i=1}^{N} a_{mi} x_{i} \leq b_{m} \;\;\;\; m = 1, 2, \dots, M\\
# \;\;& l_{i} \leq x_{i} \leq u_{i} \;\;\;\; i = 1, 2, \dots, N
# \end{align*}

# In[37]:


# load libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.stats as sta
import cplex as cp


# In[38]:


def quadratic_programming(direction, A, senses, b, c, Q, l, u, names):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their linear coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), names = names.tolist())
    
    # add quadratic coefficients in objective
    row_indices, col_indices = Q.nonzero()
    prob.objective.set_quadratic_coefficients(zip(row_indices.tolist(), col_indices.tolist(), Q.data.tolist()))

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

    print(prob.write_as_string())
    # solve the problem
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    x_star = prob.solution.get_values()
    obj_star = prob.solution.get_objective_value()

    return(x_star, obj_star)


# In[39]:


def distance_between_squares(squares_file):
    
    # decision variables : x1 , y1 , x2 , y2
    # number of decision variables : 4
    # number of constraints        : 8
    # dimension of Q matrix        : 4 X 4


    names = np.array(["x1", "y1", "x2", "y2"])
    squares = np.loadtxt(squares_file) 
    
    a1 = squares[0, 0]
    b1 = squares[0, 1]
    r1 = squares[0, 2]
    a2 = squares[1, 0]
    b2 = squares[1, 1]
    r2 = squares[1, 2]

    c = np.repeat(0, 4)
    senses = np.tile(["G", "L"], 4)
    b = np.array([a1 - 1/2 * r1, 
                 a1 + 1/2 * r1,
                 b1 - 1/2 * r1,
                 b1 + 1/2 * r1,
                 a2 - 1/2 * r2,
                 a2 + 1/2 * r2,
                 b2 - 1/2 * r2,
                 b2 + 1/2 *r2])
    l = np.repeat(-cp.infinity, 4)
    u = np.repeat(cp.infinity, 4)

    aij = np.repeat(1, 8)
    row = np.array(range(8))
    col = np.repeat(range(4), 2)
    A = sp.csr_matrix((aij, (row, col)), shape = (8,4))

    Q = 2 * sp.csr_matrix(np.array([ [1, 0, -1, 0],
                                     [0, +1, 0, -1],
                                     [-1, 0, 1, 0],
                                     [0, -1, 0, 1]] ))
    
    x_star, obj_star = quadratic_programming("minimize", A, senses, b, c, Q, l, u, names)
    x1_star = x_star[0]
    y1_star = x_star[1]
    x2_star = x_star[2]
    y2_star = x_star[3]
    distance_star = np.sqrt(obj_star)
    
    return(x1_star, y1_star, x2_star, y2_star, distance_star)


# In[40]:


x1_star, y1_star, x2_star, y2_star, distance_star = distance_between_squares("squares.txt")
print(x1_star, y1_star, x2_star, y2_star, distance_star)


# In[ ]:





# In[ ]:




