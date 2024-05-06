import numpy as np
import scipy.io as sio
from typing import List, Tuple
from numpy.typing import NDArray
from numba import jit
from time import time

@jit(nopython = True) # type: ignore
def VortexLineInducedVelocity(P1: NDArray[np.float64], P2: NDArray[np.float64], P: NDArray[np.float64], Gamma: float, tol: float = 1E-4) -> NDArray[np.float64]:
    r0 = P2 - P1
    r1 = P - P1
    r2 = P - P2
    L1 = np.linalg.norm(r1)
    L2 = np.linalg.norm(r2)

    r1crossr2 = np.cross(r1, r2)
    r1crossr2_2 = np.dot(r1crossr2, r1crossr2)
    r0r1 = np.dot(r0, r1)
    r0r2 = np.dot(r0, r2)

    Velocity = np.empty((3,), dtype = np.float64)

    if L1 < tol or L2 < tol or np.sqrt(r1crossr2_2) < tol :
        Velocity = np.array([0, 0, 0], dtype = np.float64)
    else:    
        K = Gamma / (4 * np.pi * r1crossr2_2) * (r0r1 / L1 - r0r2/L2)
        Velocity = K * r1crossr2
    
    return Velocity 

def InfluenceCoefficients(Nel: int, CollocationPoints: NDArray[np.float64], Normals: NDArray[np.float64],
                           Npanels: int, PanelNodes: NDArray[np.float64], PanelConnectivity: NDArray[np.int32],
                             Panel2ShellConnectivity: NDArray[np.int32]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Influence Coefficient matrix init
    A = np.zeros((Nel, Nel), dtype = np.float64)
    B = np.zeros((Nel, Nel), dtype = np.float64)

    gamma = 1

    for i in range(Nel):
        norm = Normals[i,:]
        CollocationPoint = CollocationPoints[i,:]
        for j in range(Npanels):
            P1 = PanelNodes[PanelConnectivity[j, 1], :]
            P2 = PanelNodes[PanelConnectivity[j, 4], :]
            P3 = PanelNodes[PanelConnectivity[j, 3], :]
            P4 = PanelNodes[PanelConnectivity[j, 2], :]

            if PanelConnectivity[j, 5] == -1: # trailing edgle horshoe vortex panel
                u12 = VortexLineInducedVelocity(P1, P2, CollocationPoint, gamma)
                u23 = VortexLineInducedVelocity(P2, P3, CollocationPoint, gamma)
                u41 = VortexLineInducedVelocity(P4, P1, CollocationPoint, gamma)

                Vinduced = u12 + u23 + u41
                VinducedFreeStream = u23 + u41

            else:
                u12 = VortexLineInducedVelocity(P1, P2, CollocationPoint, gamma)
                u23 = VortexLineInducedVelocity(P2, P3, CollocationPoint, gamma)
                u34 = VortexLineInducedVelocity(P3, P4, CollocationPoint, gamma)
                u41 = VortexLineInducedVelocity(P4, P1, CollocationPoint, gamma)

                Vinduced = u12 + u23 + u34 + u41
                VinducedFreeStream = u23 + u41

            A[i, Panel2ShellConnectivity[j,1]] += np.dot(Vinduced, norm)
            B[i, Panel2ShellConnectivity[j,1]] += np.dot(VinducedFreeStream, norm)

    return A, B

def Qmatrix(Nel: int, Ndof: int, NxElements: int, NodeMatrix: NDArray[np.float64], ElementMatrix: NDArray[np.int32]):
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

def Dmatrix(Nel: int, Ndof: int, NodeMatrix: NDArray[np.float64], ElementMatrix: NDArray[np.int32]):
    '''D matrix Relates AoA with vertical (z) displacement of the nodes'''
    D = np.zeros((Nel, Ndof), dtype = np.float64)
    for i in range(Nel):
        x1 = NodeMatrix[ElementMatrix[i,0], 0]
        x2 = NodeMatrix[ElementMatrix[i,1], 0]
        x3 = NodeMatrix[ElementMatrix[i,2], 0]
        x4 = NodeMatrix[ElementMatrix[i,3], 0]
        Dx = (x2 + x3) / 2 - (x1 + x4) /2
        D[i, 6*(ElementMatrix[i,:] + 1) - 4] = 1 / (2 * Dx) * np.array([1, -1, -1, 1], dtype = np.float64)
    return D

def Ematrix(Nel: int, Ndof, ElementMatrix: NDArray[np.int32] ):
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

def RHSVector(Nel: int, Vinf: NDArray[np.float64], normals: NDArray[np.float64]):
    RHS = np.zeros((Nel, 1))
    for i in range(Nel):
        RHS[i,0] = np.dot(-Vinf, normals[i,:])
    
    return RHS

def Circulations(InfluenceCoefficientsA: NDArray[np.float64], RHSVector: NDArray[np.float64]):
    Gamma = np.linalg.solve(InfluenceCoefficientsA, RHSVector)
    return Gamma

def InducedVelocity(InfluenceCoefficientsB: NDArray[np.float64], Gamma: NDArray[np.float64]):
    Wind = InfluenceCoefficientsB @ Gamma
    return Wind

def CalculateLift(Nel: int, rho: float, Vinf: NDArray[np.float64], Gamma: NDArray[np.float64], PanelConnectivity: NDArray[np.int32], Db: NDArray[np.float64]):
    # assert Nel == NxElements * NyElements, f'Check Number of elements and Grid Size: Nel must equal Nx * Ny. Nel = {Nel}, Nx = {NxElements}, Ny = {NyElements}'
    Lift = np.zeros((Nel, 3), dtype = np.float64)
    for i in range(Nel):
        if PanelConnectivity[i, 5] == 1: # if LE panel
            Lift[i, :] = Vinf * Gamma[i,0] * Db[i]
        else:
            Lift[i, :] = Vinf * (Gamma[i,0] - Gamma[i-1, 0]) * Db[i]

    Lift *= rho
    return np.sum(Lift, axis = 0)

def CalculateDrag():
    ...


######## Code Test ##########
if __name__ == '__main__':
    Dm = sio.loadmat('Dmatrix')
    Dq = sio.loadmat('matrixQ')
    De = sio.loadmat('Ematrix')
    dim = Dm['dim']
    X = Dm['X']
    Tn = (Dm['Tn'] - 1).astype(np.int32)
    # Dmatlab = Dm['D']
    # D = Dmatrix(160, 1122, X, Tn)
    Q = Qmatrix(160, 1122, 10, X, Tn)
    Qmatlab = Dq['Q']
    DiffQ = Q - Qmatlab
    E = Ematrix(160, 1122, Tn)
    Ematlab = De['E']
    DiffE = E -Ematlab


    # P1 = np.array([0, 0, 0], dtype = np.float64)
    # P2 = np.array([1, 1, 1], dtype = np.float64)
    # P = np.array([2, 2, 2], dtype = np.float64)
    Gamma = 2
    # V = VortexLineInducedVelocity(P1, P2, P , Gamma)

    # InfCoeffInput = sio.loadmat('InfCoeffInput')
    # InfCoeffOut = sio.loadmat('InfCoeffOut')

    # Nel = int((InfCoeffInput['xel'] *  InfCoeffInput['yel'])[0,0])
    # Npan = int(InfCoeffInput['npan'][0,0])

    # CollocationPoints = InfCoeffInput['c75elem']
    # Norm = InfCoeffInput['normals']
    # PanelNodes = InfCoeffInput['c4corner']

    # PanelConnectivity = np.zeros((Npan, 6), dtype = np.int32)
    # PanelConnectivity[:,0] = np.arange(Npan)
    # PanelConnectivity[:,1:5] = InfCoeffInput['Tnp'][:,[0,3,2,1]] - 1 
    # PanelConnectivity[:,5] = InfCoeffInput['Tmp'][:,0]

    # Panel2ShellCon = np.zeros((Npan,2) , dtype = np.int32)
    # Panel2ShellCon[:,0] = np.arange(Npan)
    # Panel2ShellCon[:,1] = InfCoeffInput['Tdp'][:,0] - 1

    # s = time()
    # A, B = InfluenceCoefficients(Nel, CollocationPoints, Norm, Npan, PanelNodes, PanelConnectivity, Panel2ShellCon)
    # f = time()

    # print('time : ', f - s)
    # DifferenceA: NDArray = A - InfCoeffOut['A']
    # DifferenceB: NDArray = B - InfCoeffOut['B']
    # print(np.max(DifferenceA.flat))
    # print(np.max(DifferenceB.flat))




    # s = time()

    # f = time()

    # print(f-s)
    # print(V)

    # print('-------------------')

    # s = time()
    # V = VortexLineInducedVelocity(P1, P2, P , Gamma)
    # f = time()

    # print(f-s)
    # print(V)