import numpy as np
import sympy as sy
import utils

class MVD():

    def __init__(self, copula, margin, copula_para=[], margin_para=[]):
        '''
        '''
        self.copula  = copula
        self.copula_para = copula_para 

        self.margin = margin
        self.margin_para = margin_para 

    def fit(self, X):
        '''
        '''
        # fit margin and compute U
        U = np.zeros_like(X)
        margin_para = []
        for i in range(X.shape[1]):

            F, para_F = self.margin[i]
            values = utils.fit_margin(F, para_F, X[:,i])
            margin_para.append(values)

            for p,val in zip(para_F[1:],values):
                F = F.subs(p,val)

            F_ = sy.lambdify(para_F[0],F,'numpy')
            U[:,i] = F_(X[:,i])
        self.margin_para = np.array(margin_para)

        # fit copula
        C, para_C = self.copula
        values = utils.fit_copula(C, para_C, U)
        self.copula_para=values



        return self

    def generate_x(self, N=1000):
        '''
        '''
        y = sy.symbols('y')
        C, para_C = self.copula
        u0 = para_C[0]
        u1 = para_C[1]
        d = para_C[2:]

        C_inv = sy.solve(sy.Eq(sy.diff(C,u0),y),u1)[0]
        C_inv = sy.lambdify((u0,y,d),C_inv,'numpy') 

        M_inv = []
        for M,v in zip(self.margin,self.margin_para):

            u = sy.symbols('u')
            F, para_F = M
            x0 = para_F[0]
            p = para_F[1:]
            
            F_inv = sy.solve(sy.Eq(F,u),x0)[0]
            F_inv = sy.lambdify((u,p),F_inv,'numpy') 
            M_inv.append(F_inv)

        X = []
        for i in range(N):
            u0, y = np.random.uniform(size=2)

            u1 = C_inv(u0,y,self.copula_para)

            x0 = M_inv[0](u0,self.margin_para[0])
            x1 = M_inv[1](u1,self.margin_para[1])
            X.append([x0,x1])

        return np.array(X)
