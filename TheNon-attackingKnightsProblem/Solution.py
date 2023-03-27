#!/usr/bin/env python
# coding: utf-8

# # The Non-attacking Knights Problem

# In[79]:


# load libraries
import numpy as np
import scipy.sparse as sp

import cplex as cp


# In[92]:


def mixed_integer_linear_programming(direction, A, senses, b, c, l, u, types, names):
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their coefficients in objective and ranges
    prob.variables.add(obj = c.tolist(), lb = l.tolist(), ub = u.tolist(), types = types.tolist(), names = names.tolist())

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


# In[116]:


def nonattacking_knights_problem(M, N):

    #number of decision veriables = M * N
    # write the ingredients of the objective function
    c = np.repeat(1, M * N)
    l = np.repeat(0, M * N)
    u = np.repeat(1, M * N)
    types = np.repeat("B", M * N)
    names = np.array(["x_{}_{}".format(i + 1, j + 1) for i in range(M) for j in range(N)])
    
    
    # now, let's find constraint count
    constraint_count = 0;
    for i in range(M):
        for j in range(N):
            if (i + 1 in range(M) and j + 2 in range(N)):
                constraint_count+=1
            if (i + 2 in range(M) and j + 1 in range(N)):
                constraint_count+=1
            if (i + 1 in range(M) and j - 2 in range(N)):
                constraint_count+=1
            if (i + 2 in range(M) and j - 1 in range(N)):
                constraint_count+=1
                
                
    # Construct constraints, accordingly
    senses = np.repeat("L", constraint_count)
    b = np.repeat(1, constraint_count)
    A = np.zeros(constraint_count * M * N).reshape(constraint_count, M * N)
    
    # to fill the A matrix, use the chessboard itself.
    row = 0
    chessboard = np.zeros(M * N).reshape(M, N)
    
    # now, fill A matrix
    for i in range(M):
        for j in range(N):

            if (i + 1 in range(M) and j + 2 in range(N)):

                chessboard[i, j] = 1
                chessboard[i + 1, j + 2] = 1

                A[row] = chessboard.flatten()
                row += 1

                chessboard = np.zeros_like(chessboard)

            if (i + 2 in range(M) and j + 1 in range(N)):

                chessboard[i, j] = 1
                chessboard[i + 2, j + 1] = 1

                A[row] = chessboard.flatten()
                row += 1

                chessboard = np.zeros_like(chessboard)

            if (i + 1 in range(M) and j - 2 in range(N)):

                chessboard[i, j] = 1
                chessboard[i + 1, j - 2] = 1

                A[row] = chessboard.flatten()
                row += 1

                chessboard = np.zeros_like(chessboard)

            if (i + 2 in range(M) and j - 1 in range(N)):

                chessboard[i, j] = 1
                chessboard[i + 2, j - 1] = 1

                A[row] = chessboard.flatten()
                row += 1

                chessboard = np.zeros_like(chessboard)
            
    A = sp.csr_matrix(A)
    
    # solve the problem
    x_star, obj_star = mixed_integer_linear_programming("maximize", A, senses, b, c, l, u, types, names)
    X_star = np.array(x_star).reshape(M, N)
    
    return(X_star, obj_star)


# In[117]:


X_star, obj_star = nonattacking_knights_problem(2, 6)
print(X_star)


# In[118]:


X_star, obj_star = nonattacking_knights_problem(3, 3)
print(X_star)


# In[ ]:




