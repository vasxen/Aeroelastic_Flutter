import numpy  as np
from matplotlib import pyplot as plt
import pandas as pd
import AerodynamicModule as aero
import Preprocessor as pre
from typing import List, Tuple, Union, Dict
from numpy.typing import NDArray
# from Preprocessor import PSHELL, PCOMP, OrthotropicMaterial, IsotropicMaterial, CORD1R, AeroelasticAnalysis

from scipy.io import savemat
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

class Q4ElementsCalculations():

    @staticmethod
    def __ShapeFunctions1GaussPoint() -> Tuple[List[float], NDArray[np.float64], NDArray[np.float64]]:
        # Node Coordinates in Natural system
        ksi_i: List[int] = [-1, 1, 1, -1]
        eta_i: List[int] = [-1, -1, 1, 1]

        #Shape functions and Derivatives at 1 gauss point
        Ni_G1: List[float] = 4*[0.25]
        Ni_partial_ksi_G1: NDArray = np.array([0.25*ksi_i[i] for i in range(4)])
        Ni_partial_eta_G1: NDArray = np.array([0.25*eta_i[i] for i in range(4)])
        return Ni_G1, Ni_partial_ksi_G1, Ni_partial_eta_G1
    
    @staticmethod
    def __ShapeFunctions4GaussPoints() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # Node Coordinates in Natural system
        ksi_i: List[int] = [-1, 1, 1, -1]
        eta_i: List[int] = [-1, -1, 1, 1]

        # Shape functions and derivatives at 4 gauss points
        # Coordinates of the gauss points
        ksi_G4: NDArray = (1/3**0.5) * np.array(ksi_i)
        eta_G4: NDArray = (1/3**0.5) * np.array(eta_i)

        Ni_G4: NDArray =  np.array([0.25 * (1 + np.array(ksi_i) * ksi_G4[i]) * (1 + np.array(eta_i)*eta_G4[i]) for i in range(4)])
        Ni_partial_ksi_G4: NDArray = np.array([0.25 * np.array(ksi_i) * (1 + np.array(eta_i) * eta_G4[i]) for i in range(4)])
        Ni_partial_eta_G4: NDArray = np.array([0.25 * np.array(eta_i) * (1 + np.array(ksi_i) * ksi_G4[i]) for i in range(4)])
        return Ni_G4, Ni_partial_ksi_G4, Ni_partial_eta_G4
    
#========== Global Class Variables Common for all Q4 Elements ================
    Z3 = np.zeros((3,3))
    Z6 = np.zeros((6,6))
    mask1: List[bool] = [False, False,True, False, True, False]
    mask2: List[bool] = [False, False,True, True, False, False]
    mask3: List[bool] = [True, True, False, False, False, False]
    mask4: List[bool] = [False, False, False, True, True, False]
    Ni_G1, Ni_partial_ksi_G1, Ni_partial_eta_G1 = __ShapeFunctions1GaussPoint()
    Ni_G4, Ni_partial_ksi_G4, Ni_partial_eta_G4 = __ShapeFunctions4GaussPoints()

    def __init__(self, Elematrix: NDArray[np.int32], Nodematrix: NDArray[np.float64], PropertiesDict: Dict[int, Union[pre.PSHELL, pre.PCOMP]]) -> None:
        self.NE: int = Elematrix.shape[0] # Number of elementss
        self.NN: int = Nodematrix.shape[0] # Number of Nodes
        self.NPE: int = Elematrix.shape[1] # Nodes per Element
        self.DofPN: int = 6 # Degrees of Freedom per node.
        self.Elematrix:NDArray[np.int32] = Elematrix
        self.Nodematrix: NDArray[np.float64] = Nodematrix
        self.Properties = PropertiesDict
        K, M, R, A = self.__GlobalMatrices()
        self.StifnessMatrix = K
        self.MassMatrix = M
        self.RotationMatrices = R
        self.Areas = A
        self.Qmatrix = self.__Qmatrix(self.NE, self.DofPN * self.NN, 11, self.Nodematrix, self.Elematrix) #TODO pass NxElement imfo to Q4Elements
        self.Dmatrix = self.__Dmatrix(self.NE, self.DofPN * self.NN, self.Nodematrix, self.Elematrix)
        self.Ematrix = self.__Ematrix(self.NE, self.DofPN * self.NN, self.Elematrix)

    @staticmethod
    def __RotationMatrix(Nodes: NDArray) -> Tuple[NDArray, np.float64]:
        S = np.cross(Nodes[2,:] - Nodes[0,:], Nodes[3,:] - Nodes[1,:] )/2
        Area = np.linalg.norm(S)
        d = Nodes[1,:] + Nodes[2,:] - Nodes[3,:] - Nodes[0,:]
        i = d / np.linalg.norm(d)
        k = S / Area
        j = np.cross(k,i)
        ijk = np.array([i,j,k])
        Z3 = Q4ElementsCalculations.Z3
        Z6 = Q4ElementsCalculations.Z6

        R_prime = np.block(arrays=[[ijk, Z3],[Z3 , ijk]])
        RotationMatrix = np.block( [[R_prime, Z6, Z6, Z6],
                                    [Z6, R_prime, Z6, Z6],
                                    [Z6, Z6, R_prime, Z6],
                                    [Z6, Z6, Z6, R_prime]])
        return RotationMatrix, Area

    @staticmethod 
    def __LocalMatricesSingleQ4Element(Nodes: NDArray[np.float64], RotationMatrix: NDArray[np.float64], Property: Union[pre.PSHELL,  pre.PCOMP]) \
        -> Tuple[NDArray, NDArray]:
        #Define Local Coordinate System
        i = RotationMatrix[0, 0:3]
        j = RotationMatrix[1, 0:3]
        #Calculate Jacobian

        J1 = np.zeros((2,2))
        for jj in range(4):
            J1 +=   np.block([[Q4ElementsCalculations.Ni_partial_ksi_G1[jj]], [Q4ElementsCalculations.Ni_partial_eta_G1[jj]]])    \
                    @ (np.atleast_2d(Nodes[jj,:])                \
                    @ np.block([[i], [j]]).T)

        # Partial Derivatives of Shape Function with respect to physical Local coordinates x' and y'
        Ni_partial_phys_G1 = np.linalg.solve(J1, np.block([[Q4ElementsCalculations.Ni_partial_ksi_G1], [Q4ElementsCalculations.Ni_partial_eta_G1]]))

        S1 = 4 * np.linalg.det(J1)
        if S1 < 0: raise ValueError(f"Element found on iteration with negative Jacobian. Element too Distorted cannot continue")

        # ======== Core Matrices =========
        if isinstance(Property, pre.PSHELL):
            Ct_prime, Cs_prime, Cm_prime, Cb_prime, Cmb_prime, rho_prime = Q4ElementsCalculations.__CoreMatricesIsotropicShell(Property)
        elif isinstance(Property, pre.PCOMP):
            Ct_prime, Cs_prime, Cm_prime, Cb_prime, Cmb_prime, rho_prime = Q4ElementsCalculations.__CoreMatricesOrthotropicShell(Property, LocalxAxis = i)

        # === Local Stifness Matrix Calculation  ===
        #Drilled Digree of Freedom theta_z
        Bt_prime = np.zeros((1,24))
        Bt_prime[0,5::6] = [Q4ElementsCalculations.Ni_G1[i] for i in range(4)]
        # Ct_prime =np.array(5*h*E / (12*(1+nu))).reshape(1,1)
        Kt_prime = Bt_prime.T * Ct_prime * Bt_prime


        #Shear
        Bs_prime = np.zeros((2,24))
        Bs_prime[0,4*Q4ElementsCalculations.mask1] = [item for i in range(4) for item in (Ni_partial_phys_G1[0, i], Q4ElementsCalculations.Ni_G1[i])]
        Bs_prime[1,4*Q4ElementsCalculations.mask2] = [item for i in range(4) for item in (Ni_partial_phys_G1[1, i], -Q4ElementsCalculations.Ni_G1[i])]
        # Cs_prime = np.array([[1, 0], [0, 1]]) * 5*h*E / 12 / (1 + nu)
        Ks_prime = Bs_prime.T @ Cs_prime @ Bs_prime


        # Membrane (inplane)
        Bm_prime = np.zeros((3,24))
        Bm_prime[0,0::6] = [Ni_partial_phys_G1[0, i] for i in range(4)]
        Bm_prime[1,1::6] = [Ni_partial_phys_G1[1, i] for i in range(4)]
        Bm_prime[2,4*Q4ElementsCalculations.mask3] = [item for i in range(4) for item in (Ni_partial_phys_G1[1, i], Ni_partial_phys_G1[0, i])]
        # Cm_prime = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1- nu)/2]]) * h * E /(1 - nu**2)

        Km_prime = Bm_prime.T @ Cm_prime @ Bm_prime

        # Summation of Matrices to Global Coordinate Elment Stifnesss Matrix
        StiffnessMatrix = S1 * RotationMatrix.T @ (Kt_prime + Ks_prime + Km_prime) @ RotationMatrix 

        # bending and Mass matrix caution quadrature 4
        # Jacobian at 4 Gauss Points

        MassMatrix = np.zeros((24,24))
        for k in range(4): # For each Gauss Point
            J4 = np.zeros((2,2))
            for jj in range(4):
                J4 += np.block([[Q4ElementsCalculations.Ni_partial_ksi_G4[k, jj]], [Q4ElementsCalculations.Ni_partial_eta_G4[k, jj]]])    \
                        @ (np.atleast_2d(Nodes[jj,:])                   \
                        @ np.block([[i], [j]]).T)

            Ni_partial_phys_G4 = np.linalg.solve(J4, np.block([[Q4ElementsCalculations.Ni_partial_ksi_G4[k,:]],[Q4ElementsCalculations.Ni_partial_eta_G4[k,:]]]))
            S4 = np.linalg.det(J4)
            if S1 < 0: raise ValueError(f"Element found on iteration with negative Jacobian. Element too Distorted cannot continue")
            # Bending
            Bb_prime = np.zeros((3,24))
            Bb_prime[0,4::6] = [Ni_partial_phys_G4[0, i] for i in range(4)]
            Bb_prime[1,3::6] = [Ni_partial_phys_G4[1, i] for i in range(4)]
            Bb_prime[2,4*Q4ElementsCalculations.mask4] = [item for i in range(4) for item in (-Ni_partial_phys_G4[0, i], Ni_partial_phys_G4[1, i])]
            # Cb_prime = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])* h**3 * E / 12 / (1-nu**2)
            Kb_prime = Bb_prime.T @ Cb_prime @ Bb_prime
            StiffnessMatrix += S4 * ( RotationMatrix.T @ Kb_prime @ RotationMatrix)

            # Mass
            N_prime = np.zeros((6,24))
            for ii in range(4):
                N_prime[:, 6*ii:6*ii+6] = Q4ElementsCalculations.Ni_G4[k,ii] * np.eye(6,6)
            
            # rho_prime = rho*h*np.diag([1,1,1,h**2/12,h**2/12,0])
            MassMatrix += S4 * (RotationMatrix.T @ N_prime.T @ rho_prime @ N_prime @ RotationMatrix)

        return StiffnessMatrix, MassMatrix

    @staticmethod
    def __CoreMatricesIsotropicShell(Property: pre.PSHELL) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

        h = Property.Thickness
        E = Property.MAT.E
        nu = Property.MAT.nu
        Density = Property.MAT.Density

        Ct =np.array(5*h*E / (12*(1+nu))).reshape(1,1)
        Cs = np.array([[1, 0], [0, 1]]) * 5*h*E / 12 / (1 + nu)
        Cm = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1- nu)/2]]) * h * E /(1 - nu**2)
        Cb = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])* h**3 * E / 12 / (1-nu**2)
        Cmb = np.zeros((2,2))
        rho = Density*h*np.diag([1,1,1,h**2/12,h**2/12,0])

        return Ct, Cs, Cm , Cb, Cmb, rho
       
    @staticmethod
    def __CoreMatricesOrthotropicShell(Property: pre.PCOMP, LocalxAxis: NDArray) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
       
        def DpDs(E1: float, E2: float , nu12: float, nu21: float, G1z: float, G2z: float, beta_i: float) \
            -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            
            def D1D2(E1, E2 , nu12, nu21, G1z, G2z) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
                a = 1 - nu12*nu21
                D1 = np.array([ [E1/a,       E2*nu12/a,  0           ],
                                [E2*nu12/a,  E2/a,       0           ],
                                [0,          0,          E1*nu21/a   ]])
                D2 = np.array([ [G1z, 0],
                                [0, G2z]])
                return D1, D2
            
            def T1T2(beta_i: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
                C = np.cos(beta_i)
                S = np.sin(beta_i)
                C2 = C*C
                S2 = S*S
                CS = C * S
                T1 = np.array([ [C2,     S2 ,    -CS  ],
                                [S2,     C2,      CS  ],
                                [2*CS, - 2*CS, C2 - S2]])
                T2 = np.array([ [C, -S],
                                [S, C]])
                return T1, T2
        
            T1, T2 = T1T2(beta_i)
            D1, D2 = D1D2(E1, E2 , nu12, nu21, G1z, G2z)
            Dp = T1.T @ D1 @ T1
            Ds = T2.T @ D2 @ T2
            return Dp, Ds
        
        AngularDifference = np.arccos(np.dot(LocalxAxis, Property.CoordinateSystem.UnitVectors[:,0]))
        beta : List[float]= [b + AngularDifference for b in Property.Theta]
        # Stress - Strain Marices 
        Cm = np.zeros((3,3), dtype= np.float64) # Membrane  
        Cmb = np.zeros((3,3),dtype= np.float64) # Membrane - Bending coupling
        Cb = np.zeros((3,3), dtype= np.float64) # Bending
        Cs = np.zeros((2,2), dtype= np.float64) # Shear 
        Ct = np.zeros((1,1), dtype= np.float64) # Fictitious Stifness
        ShearCorrection = np.zeros((2,2))

        Dps = [DpDs(Property.MAT[layer].E1,
                        Property.MAT[layer].E2,
                        Property.MAT[layer].nu12,
                        Property.MAT[layer].nu21,
                        Property.MAT[layer].G1z,
                        Property.MAT[layer].G2z,
                        beta[layer] ) for layer in range(Property.Nlayers)]
        Dp = [item[0] for item in Dps]
        Ds = [item[1] for item in Dps]

        g1 = 0
        g2 = 0
        G1zmean = 0 
        G2zmean = 0

        for layer in range(Property.Nlayers):
            tk = Property.Thickness[layer]
            Zi, Zf = Property.LayerHeight(layer)
            Zmean = 0.5 * (Zi + Zf)


            Cm += tk * Dp[layer]
            Cmb += - tk * Zmean * Dp[layer] ## CAUTION maybe minus is wrong maybe
            Cb += 1/3 * (Zf**3 - Zi**3) * Dp[layer]
            Cs += tk * Ds[layer]
            Ct += 5*tk*Property.MAT[layer].E1 / (12*(1+ Property.MAT[layer].nu12))

            g1 += 0.5 * (Zi**2 - Zf**2) * Dp[layer][0,0]
            g2 += 0.5 * (Zi**2 - Zf**2) * Dp[layer][1,1]
            ShearCorrection[0,0] += g1 / Property.MAT[layer].G1z * tk
            ShearCorrection[1,1] += g2 / Property.MAT[layer].G2z * tk
            G1zmean += Property.MAT[layer].G1z * tk
            G2zmean += Property.MAT[layer].G2z * tk

        ShearCorrection[0,0] *= G1zmean
        ShearCorrection[1,1] *= G2zmean
        ShearCorrection = np.reciprocal(ShearCorrection)
        ShearCorrection[0,0] *= Cb[0,0] **2
        ShearCorrection[1,1] *= Cb[1,1] **2
        ShearCorrection[0,1] = 1
        ShearCorrection[1,0] = 1

        Cs = Cs * ShearCorrection
        DensityList = [Property.MAT[i].Density for i in range(Property.Nlayers)]
        AverageDensity = sum(DensityList) / Property.Nlayers
        TotalThickness = sum(Property.Thickness)
        rho = AverageDensity * TotalThickness *np.diag([1,1,1,TotalThickness**2/12,TotalThickness**2/12,0])

        return Ct, Cs, Cm, Cb, Cmb, rho

    def __GlobalMatrices(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        NodeMatrix = self.Nodematrix
        Elematrix = self.Elematrix
        NE = self.NE
        NN = self.NN
        PropertiesDict = self.Properties

        # Store Element Data
        RotationMatrices = np.zeros((24,24,NE))
        Areas = np.zeros((NE,))

        GlobalStiffnessMatrix = np.zeros((6*NN, 6*NN))
        GlobalMassMatrix = np.zeros((6*NN, 6*NN))

        for ii in range(NE):
            ElementId = Elematrix[ii,0]
            NodeIds = Elematrix[ii,1:]
            Nodes = NodeMatrix[NodeIds,:]
            Property = PropertiesDict[ElementId]

            R, A = Q4ElementsCalculations.__RotationMatrix(Nodes)
            StiffnessMatrix, MassMatrix = Q4ElementsCalculations.__LocalMatricesSingleQ4Element(Nodes, R, Property)

            # Assemble to Global matrix 
            GlobalDofs = [value  for i in range(4) for value in range(6*NodeIds[i], 6*NodeIds[i]+6)]
            GlobalStiffnessMatrix[np.ix_(GlobalDofs, GlobalDofs)] += StiffnessMatrix
            GlobalMassMatrix[np.ix_(GlobalDofs, GlobalDofs)] += MassMatrix
            # Assign Element data
            RotationMatrices[:,:,ii] = R
            Areas[ii] = A
        
        return GlobalStiffnessMatrix, GlobalMassMatrix, Areas, RotationMatrices

    @staticmethod
    def __Qmatrix(Nel: int, Ndof: int, NxElements: int, NodeMatrix: NDArray[np.float64], ElementMatrix: NDArray[np.int32]) -> NDArray[np.float64]:
        '''Q matrix Relates pressure distribution with external forces'''
        Q = np.zeros((Ndof, Nel), dtype = np.float64)
        for i in range(Nel):
            y1 = NodeMatrix[ElementMatrix[i,0], 1]
            y2 = NodeMatrix[ElementMatrix[i,1], 1]
            y3 = NodeMatrix[ElementMatrix[i,2], 1]
            y4 = NodeMatrix[ElementMatrix[i,3], 1]
            Dy = (y3 + y4) / 2 - (y1 + y2) / 2
            Dofs = (ElementMatrix[i,:] + 1) * 6 - 4
            LE = i % (NxElements) == 0 # is leading edge panel?
            Q[Dofs, i] += Dy / 4
            if not LE:
                Q[Dofs, i - 1] += -Dy / 4

        return Q
    
    @staticmethod
    def __Dmatrix(Nel: int, Ndof: int, NodeMatrix: NDArray[np.float64], ElementMatrix: NDArray[np.int32]) -> NDArray[np.float64]:
        '''D matrix Relates AoA with vertical (z) displacement of the nodes'''
        D = np.zeros((Nel, Ndof), dtype = np.float64)
        for i in range(Nel):
            x1 = NodeMatrix[ElementMatrix[i,1], 0]
            x2 = NodeMatrix[ElementMatrix[i,2], 0]
            x3 = NodeMatrix[ElementMatrix[i,3], 0]
            x4 = NodeMatrix[ElementMatrix[i,4], 0]
            Dx = (x2 + x3) / 2 - (x1 + x4) /2
            D[i, 6*(ElementMatrix[i,1:] + 1) - 4] = 1 / (2 * Dx) * np.array([1, -1, -1, 1], dtype = np.float64)
        return D
    
    @staticmethod
    def __Ematrix(Nel: int, Ndof: int, ElementMatrix: NDArray[np.int32] ) -> NDArray[np.float64]:
        '''E matrix Relates structural velocity in the collocation points (zdirection ) to velocity vector { u_point }'''
        # Collocation point isoparametric coordinates
        E = np.zeros((Nel, Ndof), dtype = np.float64)
        xi = 0.5
        eta = 0
        a = np.array([-1, 1, 1, -1])
        b = np.array([-1, -1, 1, 1])
        for e in range(Nel):
            for i in range(4):
                E[e, (ElementMatrix[e,i] + 1) * 6 - 4] = 0.25 * (1 + xi * a[i]) * (1 + eta *b [i])

        return E


class VortexPanelElements():
    def __init__(self, PanelNodes: NDArray[np.float64], PanelConnectivity: NDArray[np.int32]):
        self.Nel: int = PanelConnectivity.shape[0]
        self.PanelNodes = PanelNodes
        self.PanelConnectivity = PanelConnectivity
        normals, Db = self.__NormalsAndWidth()
        self.nomals = normals
        self.Width = Db


    def __NormalsAndWidth(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        normals: NDArray[np.float64] = np.zeros((self.Nel, 3), dtype = np.float64)
        Db: NDArray[np.float64] = np.zeros((self.Nel), dtype = np.float64)
        for i in range(self.Nel):
            NodeIds = self.PanelConnectivity[i,1:5]
            Nodes = self.PanelNodes[NodeIds,:]
            S = np.cross(Nodes[2,:] - Nodes[0,:], Nodes[3,:] - Nodes[1,:] )/2
            Area = np.linalg.norm(S)
            normals[i,:] = S / Area
            Db[i] = (np.abs(Nodes[0, 1] - Nodes[3, 1]) + np.abs(Nodes[1, 1] - Nodes[2, 1])) / 2
        return normals, Db
    
    @property
    def CollocationPoints(self) -> NDArray[np.float64]:
        CollocationPoints = np.zeros((self.Nel, 3), dtype = np.float64)
        for i in range(self.Nel):
            NodeIds = self.PanelConnectivity[i, 1:5]
            Nodes = self.PanelNodes[NodeIds, :]
            CollocationPoints[i, :] = np.sum(Nodes, axis = 0) / 4
        
        return CollocationPoints

    def InfluenceCoefficients(self, Nshells: int, Panel2ShellConnectivity: NDArray[np.int32]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        A, B = aero.InfluenceCoefficients(Nshells, self.CollocationPoints, self.nomals, self.Nel, self.PanelNodes, self.PanelConnectivity, Panel2ShellConnectivity)
        return A, B


def Eigenproblem(K: NDArray[np.float64], M: NDArray[np.float64], N: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    Vals, Vecs = eigs(A = K, M = M, k = N, which= 'SM') #type: ignore
    idx = np.argsort(Vals)
    Vals = Vals[idx]
    Vecs = Vecs[:,idx]
    return Vals, Vecs

def AerodynamicFlutter(Analysis: pre.AeroelasticAnalysis):
    Structural = Analysis.Structural
    Aerodynamics = Analysis.AerodynamicMesh
    Shells = Q4ElementsCalculations(Structural.Mesh.ConnectivityMatrix, Structural.Mesh.NodeMatrix, Structural.Element2Prop)
    Panels = VortexPanelElements(Aerodynamics.NodeMatrix, Aerodynamics.ConnectivityMatrix)
    Vinf = np.atleast_1d(Analysis.Controls.Velocities)
    Neig = Analysis.Controls.NumModes
    rho = Analysis.Controls.Density
    BoundedDofs = Structural.Boundary
    AllDofs = np.arange(Shells.NN * Shells.DofPN)
    FreeDofs = np.setdiff1d(AllDofs, BoundedDofs)
    NumFreeDofs = FreeDofs.shape[0]


    A, _ = Panels.InfluenceCoefficients(Shells.NE, Analysis.Panel2ShellConnection)
    invA = np.linalg.inv(A)


    lenV = Vinf.shape[0]
    pR = np.zeros((Neig, lenV), dtype = np.float64 )
    pI = np.zeros_like(pR)
    Lamda = np.zeros_like(pR, dtype = np.complex128)

    for i in range(lenV):
        Ca = rho * Vinf[i] *  Shells.Qmatrix @ invA @ Shells.Ematrix
        Ka = Shells.StifnessMatrix + rho * Vinf[i]**2 * (Shells.Qmatrix @ invA @ Shells.Dmatrix)
        mask = np.ix_(FreeDofs, FreeDofs)
        Z =  np.zeros((NumFreeDofs, NumFreeDofs))
        I = np.eye(NumFreeDofs, NumFreeDofs)
        Mbar = np.block([[Ca[mask], Shells.MassMatrix[mask]],
                         [-I, Z]])
        Kbar = np.block([[Ka[mask], Z],
                         [Z, I]])
        
        EigenValues, EigenVectors = Eigenproblem(Kbar, Mbar, Neig)
        pR[:, i] = EigenValues.real
        pI[:, i] = EigenValues.imag

def DDmain() -> None:
    X,Y = np.meshgrid(np.arange(-5, 5, 1),np.arange(-5, 5, 1) )
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros(100)
    NodeMatrix  = np.column_stack([X,Y,Z])
    Elematrix = pre.ConnectStructuredGrid(10,10)
    M1 = pre.IsotropicMaterial(1, 7.89E-9, 210000, 0.3)
    P1 = pre.PSHELL(1, M1, 0.001)
    PropDict = dict((int(ElId), P1) for ElId in list(Elematrix[:,0]))

    ShellElements = Q4ElementsCalculations(Elematrix, NodeMatrix, PropDict) #type: ignore
    K = ShellElements.StifnessMatrix
    M = ShellElements.MassMatrix
    K = (K.T + K ) / 2 
    M = (M.T + M ) / 2
    savemat('Elematrix.MAT', {'Elements': Elematrix, 'label': 'Elements'})
    savemat('NodeMatrix.mat', {'Nodes': NodeMatrix, 'label': 'Nodes'})

    # np.savetxt('Stifness.txt', K)
    # np.savetxt('Mass.txt', M)

    # K, M = Q4Elements.StiffnessMass(NodeMatrix, Elematrix[:,1:], Elematrix.shape[0], NodeMatrix.shape[0])
    # W, V = eigs(K, M = M, k = 20) # type: ignore

    W, V = eig(K, M) # type: ignore
    Wr = W.real
    sorted_indices = np.argsort(Wr)
    sorted_eigenvalues = Wr[sorted_indices]
    # print(sorted_eigenvalues)
    V = V[:, sorted_indices]

    Displacements = np.zeros_like(NodeMatrix)
    for i in range(NodeMatrix.shape[0]):
        Displacements[i,:] = NodeMatrix[i,:] +  V[[6*i, 6*i+1, 6*i +2], 13]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Displacements[:,0],Displacements[:,1],Displacements[:,2])

    for element in Elematrix:
        # Extract x, y, z coordinates for each point in the element
        e = np.append(element[1:], element[1])
        x_coords = Displacements[e, 0]
        y_coords = Displacements[e, 1]
        z_coords = Displacements[e, 2]
        
        # Plot the line connecting the four points
        ax.plot(x_coords, y_coords, z_coords, color = 'black')
        


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    Wing = pre.WingPlanform(Wingspan = 12, Croot = 6, Ctip = 6, Sweepback= np.deg2rad(5))
    StructuralMesh, AeroMesh, Panel2ShellConnectivity = pre.MainMeshFunction(Wing, Nspan = 15, Nchord = 10)
    FixedDoFs = pre.ClampLeftEdge(StructuralMesh)
    Mat1 = pre.IsotropicMaterial(1, 1600, 9E10, 0.33)
    Prop1 = pre.PSHELL(1, Mat1, 0.001)
    
    Mat2 = pre.OrthotropicMaterial(2, 1600, 9E10, 9E9, 0.4, 40000, 40000)
    C1 = pre.CORD1R(1, np.array([0.,0., 0.]), np.eye(3,3, dtype = np.float64) )
    Prop2 = pre.PCOMP(2,[Mat2, Mat2], [0.001, 0.002], C1, [0, 3.14/2] )

    PropertyAssignment = pre.AssignProperty(StructuralMesh.ConnectivityMatrix[:,0].tolist(), Prop1)
    Structural = pre.StructuralProperties(StructuralMesh, FixedDoFs, [Prop1, Prop2], PropertyAssignment)
    controls = pre.AeroelasticAnalysisControls(np.array(1),1,1)
    Analysis = pre.AeroelasticAnalysis(Structural, AeroMesh, controls, Panel2ShellConnectivity)
    AerodynamicFlutter(Analysis)


if __name__ == '__main__': 
    main()