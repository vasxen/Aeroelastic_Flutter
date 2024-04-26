from ast import Dict
from dataclasses import dataclass
from typing import List, Union, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

# WING high level Geometric Data
@dataclass(kw_only = True)
class WingPlanform():
    Wingspan: float
    Croot: float
    Ctip: float
    Sweepback: float

#------------ Materials ------------
@dataclass
class IsotropicMaterial():
    Id: int
    Density: float
    E: float
    nu: float

    @property
    def type(self) -> str:
        return 'isotropic'

    @property
    def G(self) -> float:
        return  0.5* self.E / (1+ self.nu)

@dataclass
class OrthotropicMaterial():
    Id: int
    Density: float
    E1: float
    E2: float
    nu12: float
    G12: float
    G2z : float

    @property
    def G1z(self)-> float:
        return self.G12
    
    @property
    def nu21(self) ->float:
        return self.E2 / self.E1 * self.nu12

# ----------- Properties -----------
@dataclass
class PSHELL():
    Id: int
    MAT: IsotropicMaterial
    Thickness: float

    @property
    def type(self) -> str:
        return 'PSHELL'

@dataclass
class PCOMP():
    Id: int
    MAT: List[OrthotropicMaterial]
    Thickness: List[float]
    CoordinateSystem: NDArray[np.float64] #i,j,k as 3 column vectors
    Theta: List[float]

    @property
    def type(self) -> str:
        return 'PCOMP'
    
    @property 
    def Nlayers(self) -> int:
        return len(self.Thickness)
    
    def LayerHeight(self, i: int) -> Tuple[float, float]:
        Htot = sum(self.Thickness)
        H0 = sum(self.Thickness[0:i])
        H1 = sum(self.Thickness[0:i+1])
        return (-Htot/2 + H0, -Htot/2 + H1)

@dataclass
class PBEAM():
    Id: int
    MAT: IsotropicMaterial
    ...

# -------- Aeroelastic Analysis --------
@dataclass
class AerolasticAnalysis():
    NodeMatrix: NDArray[np.float64]
    ShellConnectiviry: NDArray[np.int32]
    Boundary: List[int]
    Properties: List[PSHELL | PCOMP]
    Element2Prop: Dict[int, PSHELL | PCOMP]
    PanelConnectivity: NDArray[np.int32]
    Panel2ShellConnectivity: Dict[int, int]



# ----------- Preprocessor Functions ----------
def CreateWingNodes(Wing: WingPlanform, Nspan: int, Nchord: int, WakeFactor: float = 2) -> NDArray:
    Ycoords = np.linspace(0, Wing.Wingspan, Nspan)
    ChordLengths = Wing.Croot - Ycoords /Wing. Wingspan * (Wing.Croot - Wing.Ctip)
    wakelength  = WakeFactor * Wing.Wingspan
    # Get Equation for x coordinates on C/4
    if Wing.Sweepback == 0:
        XcoordsC4 = Wing.Croot/4 * np.ones_like(Ycoords)
    else:
        A1 = 1 / np.tan(Wing.Sweepback)
        A0 = -A1 * Wing.Croot / 4
        XcoordsC4 = (Ycoords - A0) / A1
    
    LeadingEdgeXcoord = XcoordsC4 - ChordLengths/4

    NodeMatrixShell = np.zeros((Nspan * Nchord, 3))
    NodeMatrixAeroPanel = np.zeros((Nspan * (Nchord + 1), 3))

    counter = 0
    for i in range(Nspan):
        X, Xstep = np.linspace(LeadingEdgeXcoord[i], LeadingEdgeXcoord[i] + ChordLengths[i], Nchord, retstep= True)
        
        Xaeropanel = X + Xstep / 4
        for j in range(Nchord + 1):
            if j == Nchord:
                NodeMatrixAeroPanel[counter, :] = [Xaeropanel[j -1] + wakelength, Ycoords[i], 0]
                counter += 1
                continue

            NodeMatrixShell[i*Nchord + j, :] = [X[j], Ycoords[i], 0]
            NodeMatrixAeroPanel[counter, :] = [Xaeropanel[j], Ycoords[i], 0]
            counter += 1


    NodeMatrix = np.concatenate((NodeMatrixShell, NodeMatrixAeroPanel), axis = 0)
    return NodeMatrix

def ConnectStructuredGrid(Nspan: int, Nchord: int, NodeIndeces: List[int] = [], FirstIndex: int = 0) -> NDArray[np.int32]:
    if not NodeIndeces:
        NodeIndeces = list(range(0, Nspan*Nchord))
    NxElements = Nchord - 1 
    NyElements = Nspan - 1
    NElements = NyElements *  NxElements
    Elemtatrix = np.zeros((NElements,5), dtype = np.int32)
    i: int = 0
    j: int = 0
    while j < NElements:  # in range(0, Nspan*Nchord - 11):
        if (i+1)%Nchord  == 0 and i != 0:
            i += 1
            continue
        Elemtatrix[j,:] = [j+ FirstIndex, NodeIndeces[i], NodeIndeces[i+1], NodeIndeces[i+Nchord + 1], NodeIndeces[i+ Nchord]]
        i += 1
        j += 1
    return Elemtatrix

def CreateSPC(NodeIds: List[int], Dofs: str)-> List[int]:
    validDofs = ['1' ,'2', '3' ,'4', '5', '6']
    GlobalDofsConstrained = []
    for Id in NodeIds:
        GlobalDofsConstrained += [6*Id + int(char) - 1 for char in Dofs if char in validDofs]

    return GlobalDofsConstrained

def ClampLeftEdge(Nspan: int, Nchord: int) -> List[int]:
    NodeIds = [i*Nspan for i in range(Nchord)]
    Dofs = '123456'
    FixedDofs = CreateSPC(NodeIds, Dofs)
    return FixedDofs

def AssignProperty(ElementIds: List[int], Property: Union[PSHELL, PCOMP]) -> dict[int, Union[PSHELL, PCOMP]]:
    return dict((ElId, Property) for ElId in ElementIds)

def CreateCaeroPanels(Nspan: int, Nchord: int, NodeIndeces: List[int] = [], FirstIndex: int = 0 ) -> NDArray[np.int32]:
    Elematrix =  ConnectStructuredGrid(Nspan, Nchord, NodeIndeces, FirstIndex)
    NxElements = Nchord - 1
    Aeromatrix = np.zeros((Elematrix.shape[0], 6), dtype = np.int32)
    Aeromatrix[:,0:5] = Elematrix
    for i in range(Aeromatrix.shape[0]):
        if i % (NxElements) == 0:
            Aeromatrix[i, 5] = 1
        elif i % (NxElements)  == NxElements - 1:
            Aeromatrix[i, 5] = -1
    return Aeromatrix

def main() -> None:
    Wing = WingPlanform(Wingspan = 12, Croot = 6, Ctip = 6, Sweepback= np.deg2rad(5))
    Nspan = 15 +1 
    Nchord = 10 + 1
    NodeMatrix = CreateWingNodes(Wing, Nspan,Nchord)
    Elematrix = ConnectStructuredGrid(Nspan, Nchord)
    Aeropanels = CreateCaeroPanels(Nspan , Nchord + 1, list(range(Nspan*Nchord, 2*Nspan*Nchord + Nspan +1 )) )
    FixedDoFs = ClampLeftEdge(Nspan, Nchord)
    Mat1 = IsotropicMaterial(1, 1600, 9E10, 0.33)
    Prop1 = PSHELL(1, Mat1, 0.001)

    PropertyAssignment = AssignProperty(Elematrix[:,0].tolist(), Prop1)
    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.scatter(NodeMatrix[:,0], NodeMatrix[:,1])
    for i in range(NodeMatrix.shape[0]):
        ax.text(NodeMatrix[i,0], NodeMatrix[i,1], str(i))
        ax.scatter(NodeMatrix[i,0], NodeMatrix[i,1], color = 'blue')
    for element in Elematrix:
        # Extract x, y, z coordinates for each point in the element
        element = np.append(element[1:], element[1])
        x_coords = NodeMatrix[element, 0]
        y_coords = NodeMatrix[element, 1]
        
        # Plot the line connecting the four points
        ax.plot(x_coords, y_coords, color = 'red')

    for element in Aeropanels:
        # Extract x, y, z coordinates for each point in the element
        element = np.append(element[1:-1], element[1])
        x_coords = NodeMatrix[element, 0]
        y_coords = NodeMatrix[element, 1]
        z_coords = NodeMatrix[element, 2]
        
        # Plot the line connecting the four points
        ax.plot(x_coords, y_coords, color = 'green')
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()