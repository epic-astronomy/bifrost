# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Eqn 10 Mitch's paper
# dagger is Hermetian transpose
# Caret is the peel visibility matrix
# K indicates average over K frequency channels
# f0 is central frequency
# Nalpha number of stands so j, k stand index

import sys, random, scipy.optimize, copy, numpy as np



# Class to make matrix ops easier
class Matrix(object): 
  def __init__(self, a, b, c, d):
    self.matrix = [ [ a, b ], [ c, d ]  ]

  def multiply(self, m):
    a = self.matrix[0][0]*m.matrix[0][0]+self.matrix[0][1]*m.matrix[1][0]
    b = self.matrix[0][0]*m.matrix[0][1]+self.matrix[0][1]*m.matrix[1][1]
    c = self.matrix[1][0]*m.matrix[0][0]+self.matrix[1][1]*m.matrix[1][0]
    d = self.matrix[1][0]*m.matrix[0][1]+self.matrix[1][1]*m.matrix[1][1]
    return Matrix(a, b, c, d)

  def transpose(self):
    a = self.matrix[0][0]
    b = self.matrix[0][1]
    c = self.matrix[1][0]
    d = self.matrix[1][1]
    return Matrix(a, c, b, d)

  def subtract(self, m):
    a = self.matrix[0][0]-m.matrix[0][0]
    b = self.matrix[0][1]-m.matrix[0][1]
    c = self.matrix[1][0]-m.matrix[1][0]
    d = self.matrix[1][1]-m.matrix[1][1]
    return Matrix(a, b, c, d)

  def add(self, m):
    a = self.matrix[0][0]+m.matrix[0][0]
    b = self.matrix[0][1]+m.matrix[0][1]
    c = self.matrix[1][0]+m.matrix[1][0]
    d = self.matrix[1][1]+m.matrix[1][1]
    return Matrix(a, b, c, d)

  def inverse(self):
    a = self.matrix[0][0]
    b = self.matrix[0][1]
    c = self.matrix[1][0]
    d = self.matrix[1][1]
    det = a*d-b*c
    return Matrix(d/det, -b/det, -c/det, a/det)

  def det(self): 
    a = self.matrix[0][0]
    b = self.matrix[0][1]
    c = self.matrix[1][0]
    d = self.matrix[1][1]
    det = a*d-b*c
    return det

  def squared_norm(self):    # Squared frobenius norm
    a = self.matrix[0][0]
    b = self.matrix[0][1]
    c = self.matrix[1][0]
    d = self.matrix[1][1]
    return a*a+b*b+c*c+d*d   

  def multiply_scalar(self, val):
    a = self.matrix[0][0]*val
    b = self.matrix[0][1]*val
    c = self.matrix[1][0]*val
    d = self.matrix[1][1]*val
    return Matrix(a, b, c, d)    

  def perturb(self):
    self.matrix[0][0] *= 1.0+(random.random()-0.5)/2.5
    self.matrix[0][1] *= 1.0+(random.random()-0.5)/2.5
    self.matrix[1][0] *= 1.0+(random.random()-0.5)/2.5
    self.matrix[1][1] *= 1.0+(random.random()-0.5)/2.5

  def printm(self, newline=True):
    print "[",
    for i in range(2):
      for j in range(2):
        print self.matrix[i][j],
    print "]",
    if newline: print


def myrand():
  r = int((random.random()-0.5)*10)
  if r == 0: r = 1
  return r

def random_matrix():
  return Matrix(complex(myrand(), myrand()), complex(myrand(), myrand()), complex(myrand(), myrand()), complex(myrand(), myrand()))  


class V_P_J(object):

    def __init__(self, num_stands): 
	self.num_stands = num_stands
  
    def create_perfect_P(self, V):
        P = [ [ None for i in range(self.num_stands) ] for j in range(self.num_stands) ]
        J = [ None for i in range(self.num_stands) ]

        # Create random J
        for j in range(self.num_stands):
            J[j] = random_matrix()
	    while abs(J[j].det()) < 0.0001: J[j] = random_matrix()

        # Create P from J and V
        for j in range(self.num_stands):
            for k in range(j+1, self.num_stands):
                P[j][k] =  (J[j].inverse()).multiply(V[j][k]).multiply(J[k].transpose().inverse())

	print "Residual of perfect P, J, V (should be 0)", self.residual(V, J, P)
	return P, J

    def perturb_J(self, J):
 	J_perturb = copy.deepcopy(J)

        # Perturb J so we can perturb V. 
        for j in range(self.num_stands):
             J_perturb[j].perturb()
  
	return J_perturb

    def compare_vis(self, j_matrices_elements):
        J_in = [ None for j in range(self.num_stands) ]

        for j in range(self.num_stands):
            J_in[j] = Matrix(complex(j_matrices_elements[j*8+0], j_matrices_elements[j*8+1]), 
		complex(j_matrices_elements[j*8+2], j_matrices_elements[j*8+3]),
		complex(j_matrices_elements[j*8+4], j_matrices_elements[j*8+5]),
		complex(j_matrices_elements[j*8+6], j_matrices_elements[j*8+7]))
        return self.residual(self.V, J_in, self.P)


    def solve(self, V, J_perturb, P):		# Using an optimization routine
        self.V = V
	self.P = P

        flat = [ None for i in range(self.num_stands*8) ]
        for j in range(self.num_stands):
            flat[j*8+0] = J_perturb[j].matrix[0][0].real
            flat[j*8+1] = J_perturb[j].matrix[0][0].imag
            flat[j*8+2] = J_perturb[j].matrix[0][1].real
            flat[j*8+3] = J_perturb[j].matrix[0][1].imag
            flat[j*8+4] = J_perturb[j].matrix[1][0].real
            flat[j*8+5] = J_perturb[j].matrix[1][0].imag
            flat[j*8+6] = J_perturb[j].matrix[1][1].real
            flat[j*8+7] = J_perturb[j].matrix[1][1].imag

        res = scipy.optimize.minimize(self.compare_vis, flat)
        print "Success?", res.success

        result_J = res.x

        J_result = [ None for j in range(self.num_stands) ]

        for j in range(self.num_stands):
            J_result[j] = Matrix(complex(result_J[j*8+0], result_J[j*8+1]), 
		complex(result_J[j*8+2], result_J[j*8+3]),
		complex(result_J[j*8+4], result_J[j*8+5]),
		complex(result_J[j*8+6], result_J[j*8+7]))


	return J_result


    def residual(self, Vin, Jin, Pin):  # Between V and J.P.Jt
        res = 0
        for j in range(self.num_stands):
            for k in range(j+1,self.num_stands):
                Vresidual = Vin[j][k].subtract(Jin[j].multiply(Pin[j][k]).multiply(Jin[k].transpose()))
                res += abs(Vresidual.squared_norm())

        return res

    def ratio(self, x, y):
        #print x, y
        if x == 0 and y == 0: return 0
        elif x == 0: return abs(y)
        elif y == 0: return abs(x)
        else: return abs((x-y)/y)



if __name__ == "__main__":
    stands = 8
    V = [ [ None for i in range(stands) ] for j in range(stands) ]

    # Create random P
    for j in range(stands):
        for k in range(j+1, stands):
            V[j][k] = random_matrix()

    cal_matrices = V_P_J(stands)
    perfect_P, perfect_J = cal_matrices.create_perfect_P(V)
    V_perturb = cal_matrices.perturb_V(V, perfect_J, perfect_P)


    V_cal = cal_matrices.solve(V_perturb, perfect_J, perfect_P)

 


      
