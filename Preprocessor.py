import numpy as np
import h5py 
import os

from dataclasses import dataclass
from typing import List, Union, Tuple, Dict
from numpy.typing import NDArray
from matplotlib import pyplot as plt

# WING high level Geometric Data
@dataclass(kw_only = True)
class WingPlanform():
    Wingspan: float
    Croot: float
    Ctip: float
    Sweepback: float

# Coordinate System
@dataclass
class CORD1R():
    Id : int
    Origin: NDArray[np.float64]
    UnitVectors: NDArray[np.float64] #i,j,k as 3 column vectors

    @property
    def type(self):
        return 'CORD1R'
   
#------------ Materials ------------
@dataclass
class IsotropicMaterial():
    Id: int
    Density: float
    E: float
    nu: float

    @property
    def type(self) -> str:
        return 'MAT1'

    @property
    def G(self) -> float:
        return  0.5* self.E / (1+ self.nu)
    
    def to_string(self) -> str:
        data = [str(e) for e in [self.type, self.Id, self.Density, self.E, self.nu]]
        return ','.join(data)
    
    @staticmethod
    def from_string(S: str) -> 'IsotropicMaterial':
        parsed = S.split(',')
        if len(parsed) != 5 or parsed[0] != 'MAT1':
            raise ValueError(f'Invalid String format: {S}')
        Id, Dens, E, nu = parsed[1:]
        Id = int(Id)
        Dens = float(Dens)
        E = float(E)
        nu = float(nu)
        return IsotropicMaterial(Id, Dens, E, nu)
        
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
    
    @property
    def type(self) -> str:
        return 'MAT8'
    
    def to_string(self) -> str:
        data = [str(e) for e in [self.type, self.Id, self.Density, self.E1, self.E2, self.nu12, self.G12, self.G2z]]
        return ','.join(data)
    
    @staticmethod
    def from_string(S: str) -> 'OrthotropicMaterial':
        parsed = S.split(',')
        if len(parsed) != 8 or parsed[0] != 'MAT8':
            raise ValueError(f'Invalid String format: {S}')

        Id, Dens, E1, E2, nu12, G12, G2z = parsed[1:]
        Id = int(Id)
        Dens = float(Dens)
        E1 = float(E1)
        E2 = float(E2)
        nu12 = float(nu12)
        G12 = float(G12)
        G2z = float(G2z)
        return OrthotropicMaterial(Id, Dens, E1, E2, nu12, G12, G2z)

# ----------- Properties -----------
@dataclass
class PSHELL():
    Id: int
    MAT: IsotropicMaterial
    Thickness: float

    @property
    def type(self) -> str:
        return 'PSHELL'
    
    def to_string(self) -> str:
        data = (self.type, self.Id, self.MAT.Id, self.Thickness)
        return ','.join([str(e) for e in data])
    
    @staticmethod
    def from_string(S: str, MAT: IsotropicMaterial) -> 'PSHELL':
        parsed = S.split(',')
        if len(parsed) != 4 or parsed[0] != 'PSHELL':
            raise ValueError(f'Invalid String format: {S}')

        Id, Mid, Thick = parsed[1:]
        Id = int(Id)
        Mid = int(Mid)
        Thick = float(Thick)
        if Mid != MAT.Id:
            raise ValueError(f'''Material Id does not match Property referenced material ID.
                             Property mat reference: {Mid}
                             Provided mat refernece: {MAT.Id}''')
        
        
        return PSHELL(Id, MAT, Thick)

@dataclass
class PCOMP():
    Id: int
    MAT: List[OrthotropicMaterial]
    Thickness: List[float]
    CoordinateSystem: CORD1R 
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
    
    def to_string(self) -> str:
        data = ','.join([str(e) for e in [self.type, self.Id, self.Nlayers, self.CoordinateSystem.Id]])
        for i in range(self.Nlayers):
            data += ','.join([ str(e) for e in [self.MAT[i].Id, self.Thickness[i], self.Theta[i]]] )
        
        return data
    
    @staticmethod
    def from_string(S: str, CoordinateSystem: CORD1R, Materials: List[OrthotropicMaterial]) -> 'PCOMP':
        parsed = S.split(',')
        if len(parsed) < 7 or parsed[0] != 'PCOMP':
            raise ValueError(f'Invalid String format: {S}')
        
        Id, Nlayers, CORDId = [int(e) for e in parsed[1:4]]
        assert len(parsed[4:]) == 3 * Nlayers, \
            f'Input does not have approprite number of arguments {len(parsed)}  Required: {3 * Nlayers + 4}'
        
        if CORDId != CoordinateSystem.Id:
            raise ValueError(f'''CORD1R Id does not match Property referenced CORD1R Id.
                             Property CORD1R reference: {CORDId}
                             Provided CORD1R refernece: {CoordinateSystem.Id}''')
        Mid: List[int] = []
        Thickness: List[float] = []
        Theta: List[float] = []
        
        for i in range(Nlayers):
            Mid_i, Thick_i, Theta_i = parsed[4+i : 4+i+3]
            if Mid_i != Materials[i].Id:
                raise ValueError(f'''MAT8 Id with index {i+1} does not match Property referenced MAT8 Id.
                                Property MAT8 reference: {Mid_i}
                                Provided MAT8 refernece: {Materials[i].Id}''')
            
            Mid.append(int(Mid_i))
            Thickness.append(float(Thick_i))
            Theta.append(float(Theta_i))

        return PCOMP(Id, Materials, Thickness, CoordinateSystem, Theta)

@dataclass
class PBEAM():
    Id: int
    MAT: IsotropicMaterial
    ...

@dataclass
class CAERO():
    PanelNodes: NDArray[np.float64]
    PanelConnectivity: NDArray[np.int32]

# -------- Aeroelastic Analysis --------
@dataclass
class AerolasticAnalysis():
    NodeMatrix: NDArray[np.float64]
    ShellConnectivity: NDArray[np.int32]
    Boundary: List[int]
    Properties: List[PSHELL | PCOMP]
    Element2Prop: Dict[int, PSHELL | PCOMP]
    Aerodynamics: CAERO

    def ExportData(self, filepath: str):
        directory = os.path.dirname(filepath)
        filename = filepath.replace(directory, '')
        if not(filename.endswith(('.h5', '.hdf5'))):
            filename += '.h5'
        
        name, _ = filename.split('.')

        if not(name.isalnum()):
            raise ValueError(f'filename {filename} contains special characters')
        filepath = directory + filename

        #Prepare Data
        PropData = np.array([p.to_string() for p in self.Properties], dtype = 'S')
        MatData:List[str] = []

        for p in self.Properties:
            if isinstance(p, PSHELL):
                MatData.append(p.MAT.to_string())
            elif isinstance(p, PCOMP):
                for i in range(p.Nlayers):
                    s = p.MAT[i].to_string()
                    if s not in MatData:
                        MatData.append(p.MAT[i].to_string())
                


            else: raise ValueError(f'property is not PSHELL or PCOMP it is {type(p)}')

        MatDataArray = np.array(MatData, dtype = 'S')


        keys = list(self.Element2Prop.keys())
        values = [e.Id for e in self.Element2Prop.values()]
        length = len(keys)
        Element2PropData = np.empty((length, 2), dtype = np.int32)
        Element2PropData[:,0] = keys 
        Element2PropData[:,1] = values

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('NODES', data = self.NodeMatrix)
            f.create_dataset('CQUAD4', data = self.ShellConnectivity)
            f.create_dataset('SPC', data = np.array(self.Boundary))
            f.create_dataset('PROPERTIES', data = PropData)
            f.create_dataset('ASSIGN_PROPERTIES', data = Element2PropData)
            f.create_dataset('MATERIALS', data = MatDataArray)
            CAEROGroup = f.create_group('CAERO')
            CAEROGroup.create_dataset('NODES', data = self.Aerodynamics.PanelNodes)
            CAEROGroup.create_dataset('PANELS', data = self.Aerodynamics.PanelConnectivity)
            CORDGroup = f.create_group('CORD')
            for p in self.Properties:
                if isinstance(p, PCOMP):
                    name = p.CoordinateSystem.type + '_' + str(p.CoordinateSystem.Id)
                    temp = CORDGroup.create_group(name)
                    temp.create_dataset('ORIGIN', data = p.CoordinateSystem.Origin)
                    temp.create_dataset('UNIT_VECTORS', data = p.CoordinateSystem.UnitVectors)

# ----------- Preprocessor Functions ----------
def CreateWingNodes(Wing: WingPlanform, Nspan: int, Nchord: int, WakeFactor: float = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

    NodeMatrixShell = np.zeros((Nspan * Nchord, 3), dtype = np.float64)
    NodeMatrixAeroPanel = np.zeros((Nspan * (Nchord + 1), 3), dtype = np.float64)

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


    return NodeMatrixShell, NodeMatrixAeroPanel

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
    NodeMatrix, PanelNodes = CreateWingNodes(Wing, Nspan,Nchord)
    Elematrix = ConnectStructuredGrid(Nspan, Nchord)
    Aeropanels = CreateCaeroPanels(Nspan , Nchord + 1 )
    FixedDoFs = ClampLeftEdge(Nspan, Nchord)
    Mat1 = IsotropicMaterial(1, 1600, 9E10, 0.33)
    Prop1 = PSHELL(1, Mat1, 0.001)
    Caero = CAERO(PanelNodes, Aeropanels)
    
    Mat2 = OrthotropicMaterial(2, 1600, 9E10, 9E9, 0.4, 40000, 40000)
    C1 = CORD1R(1, np.array([0.,0., 0.]), np.eye(3,3, dtype = np.float64) )
    Prop2 = PCOMP(2,[Mat2, Mat2], [0.001, 0.002], C1, [0, 3.14/2] )

    PropertyAssignment = AssignProperty(Elematrix[:,0].tolist(), Prop1)
    Analysis = AerolasticAnalysis(NodeMatrix, Elematrix, FixedDoFs, [Prop1, Prop2], PropertyAssignment, Caero)
    Analysis.ExportData('test')
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
        x_coords = PanelNodes[element, 0]
        y_coords = PanelNodes[element, 1]
        z_coords = PanelNodes[element, 2]
        
        # Plot the line connecting the four points
        ax.plot(x_coords, y_coords, color = 'green')
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()