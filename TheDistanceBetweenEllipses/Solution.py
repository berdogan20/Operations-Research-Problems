#!/usr/bin/env python
# coding: utf-8

# # The Distance Between Ellipses

# In[1]:


# load libraries
import numpy as np
import scipy.sparse as sp
import cplex as cp


# In[39]:


def distance_between_ellipses(ellipses_file):
    
    # number of decision variables = 4 = fixed
    # number of constraints = 2 = fixed

    # load the file
    ellipses = np.loadtxt(ellipses_file)
    a1, b1, c1, d1, e1, f1 = ellipses[0, :]
    a2, b2, c2, d2, e2, f2 = ellipses[1, :]
    
    # objective matrix
    Q0 = 2 * sp.csr_matrix(np.array([[+1, 0, -1, 0],
                                 [0, +1, 0, -1],
                                 [-1, 0, +1, 0],
                                 [0, -1, 0, +1]]))
    # create an empty optimization problem
    prob = cp.Cplex()

    # add decision variables to the problem including their linear coefficients in objective and ranges
    prob.variables.add(obj = [0, 0, 0, 0],
                       lb = [-cp.infinity, -cp.infinity, -cp.infinity, -cp.infinity],
                       ub = [+cp.infinity, +cp.infinity, +cp.infinity, +cp.infinity],
                       names = ["x1", "y1", "x2", "y2"])

    # add quadratic coefficients in objective
    row_indices, col_indices = Q0.nonzero()
    prob.objective.set_quadratic_coefficients(zip(row_indices.tolist(), 
                                                  col_indices.tolist(), 
                                                  Q0.data.tolist()))

    # define problem type
    prob.objective.set_sense(prob.objective.sense.minimize)

    # add the first quadratic constraint to the problem 
    prob.quadratic_constraints.add(lin_expr = cp.SparsePair([0, 1], [d1, e1]),
                                   quad_expr = cp.SparseTriple(ind1 = [0, 1, 1],
                                                               ind2 = [0, 0, 1],
                                                               val = [a1, b1, c1]),
                                   sense = "L",
                                   rhs = -f1)

    # add the second quadratic constraint to the problem 
    prob.quadratic_constraints.add(lin_expr = cp.SparsePair([2, 3], [d2, e2]),
                                   quad_expr = cp.SparseTriple(ind1 = [2, 3, 3],
                                                               ind2 = [2, 2, 3],
                                                               val = [a2, b2, c2]),
                                   sense = "L",
                                   rhs = -f2)

    print(prob.write_as_string())

    # solve the problem
    prob.solve()

    # check the solution status
    print(prob.solution.get_status())
    print(prob.solution.status[prob.solution.get_status()])

    # get the solution
    solution = prob.solution.get_values()
    objective = prob.solution.get_objective_value()

    (x1_star, y1_star, x2_star, y2_star) = solution
    distance_star = np.sqrt(objective)
    
    return(x1_star, y1_star, x2_star, y2_star, distance_star)


# In[40]:


(x1_star, y1_star, x2_star, y2_star, distance_star) = distance_between_ellipses("ellipses.txt")
print(x1_star, y1_star, x2_star, y2_star, distance_star)


# In[ ]:




