import os
import subprocess
import pandas as pd
import numpy as np
from enum import Enum
from scipy.optimize import minimize, Bounds
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any, Dict
import plotly.graph_objects as go
import pickle
# from time import time


# ---------- Classes -------------
@dataclass
class MAT8():
    '''A MAT8 class which contains all the information of a MAT8 orthotropic material bulk data entry in optistruct'''
    LineIndex: int
    MID: int
    E1: float
    E2: float
    NU12: float
    G12: float
    G1Z: str
    G2Z: str
    RHO: float

    def to_string(self) -> str:
        ''' This methods returns the information contained in the class in a string format identical to the one found in a .fem input file'''
        s = f'MAT8,{self.MID},{self.E1},{self.E2},{self.NU12},{self.G12},{self.G1Z},{self.G2Z},{self.RHO},'
        return s
    
@dataclass
class Ply():
    '''A Ply class which contains all the information of a Ply of an PCOMP composite shell property bulk data entry in optistruct'''
    index: int
    MID: int
    Thickness: float
    Theta: float

    def to_string(self, SOUT: str = 'NO') -> str:
        ''' This methods returns the information contained in the class in a string format identical to the one found in a .fem input file'''
        s = f'{self.MID},{self.Thickness},{self.Theta},{SOUT}'
        return s

class PCOMP():
    '''A PCOMP class which contains all the information of a Ply of an PCOMP composite shell property bulk data entry in optistruct'''

    def __init__(self,Id: int, Plies: List[Ply], original_txt_lines: List[str], Indeces: Tuple[int,int]):
        self._Id = Id
        self._Plies = Plies
        self._original_txt_lines = original_txt_lines
        self._Indeces = Indeces


    #=============== PROPERTIES =================

    # --------- Id -----------
    @property
    def Id(self):
        return self._Id
    
    @Id.setter
    def Id(self, val):
        raise AttributeError('Id is immutable and cannot be changed')
    # --------- // -----------
    
    # -------- Plies ---------
    @property
    def Plies(self):
        return self._Plies
    
    @Plies.setter
    def Plies(self, val: List[Ply]):
        self._Plies = val
    # --------- // -----------

    # --------- Original_txt -----------
    @property
    def original_txt_lines(self):
        return self._original_txt_lines
    
    @original_txt_lines.setter
    def original_txt_lines(self, val):
        raise AttributeError('original_txt_lines is immutable and cannot be changed')
    # --------- // -----------
    
    # --------- Indeces -----------
    @property
    def Indeces(self):
        return self._Indeces
    
    @Indeces.setter
    def Indeces(self, val: Tuple[int, int]):
        raise AttributeError('Indeces is immutable and cannot be changed')

    # --------- // -----------
    
    
    # --------- NumPlies -----------
    @property
    def NumPlies(self) -> int:
        return len(self.Plies)
    # --------- // -----------
    

    #=============== METHODS =================

    def to_string(self) -> List[str]:
        ''' This methods returns the information contained in the class in a string format identical to the one found in a .fem input file'''

        Lines: List[str] = []
        Lines.append(self.original_txt_lines[0][:-1])

        for i in range(0,self.NumPlies - 1,2):
            ply1 = self.Plies[i]
            ply2 = self.Plies[i + 1] 
            line = f'+,{ply1.to_string()},{ply2.to_string()},'
            Lines.append(line)

        if i + 1 < self.NumPlies - 1:
            line = f'+,{self.Plies[-1].to_string()},'
            Lines.append(line)

        return Lines

@dataclass
class FlutterAnalysisPoint():
    ModeNumber : int
    MachNumber : float
    DensityRatio: float
    Method: str
    Data: pd.DataFrame

    def DetectFlutter(self) -> Tuple[List[float], List[Tuple[int, int]]]:
        Stable = self.Data['DAMPING'] < 0
        FlutterIndeces: List[Tuple[int, int]] = []
        for i in range(1, Stable.shape[0]):
            if (Stable[i-1]) and (not Stable[i]):
                FlutterIndeces.append((i-1, i))
        
        FlutterSpeed = []
        for ind in FlutterIndeces:
            D1 = self.Data['DAMPING'][ind[0]]
            D2 = self.Data['DAMPING'][ind[1]]
            V1 = self.Data['VELOCITY'][ind[0]]
            V2 = self.Data['VELOCITY'][ind[1]]
            m = (D2 - D1) / (V2 - V1)
            Vflutter = V1 - D1/m
            FlutterSpeed.append(Vflutter)

        return FlutterSpeed, FlutterIndeces
    
    def Plot(self,fig: go.Figure, ix: int, iy: int, Plotname: str = '') -> go.Figure:
        scatter = go.Scatter(x = self.Data.iloc[:,ix], y =  self.Data.iloc[:,iy], mode = 'lines+markers', name = Plotname)
        fig.add_trace(scatter)
        return fig


    def PlotDamping(self) -> go.Figure:
        fig = go.Figure()
        self.Plot(fig, 2, 3, f'MODE {self.ModeNumber}')
        fig.update_layout(xaxis_title = 'VELOCITY', yaxis_title = 'DAMPING', showlegend = True)
        return fig


    def PlotFrequency(self) -> go.Figure:
        fig = go.Figure()
        self.Plot(fig, 2, 4, f'MODE {self.ModeNumber}')
        fig.update_layout(xaxis_title = 'VELOCITY', yaxis_title = 'DAMPING', showlegend = True)
        return fig


    
@dataclass
class FlutterSubcase():
    SubcaseId: int
    XY_Symmetry: bool
    XZ_Symmetry: bool
    Method: str
    NumPoints: int
    Points: List[FlutterAnalysisPoint]

    def FlutterInfo(self):
        FlutterInfo: Dict[int, float] = {}
        for point in self.Points:
            Vel, _ = point.DetectFlutter()
            if Vel: FlutterInfo[point.ModeNumber] = min(Vel)
        
        return FlutterInfo

class FlutterSummary():

    @staticmethod
    def __split_subcases(lines: List[str]) -> List[List[str]]:
        subcase_id = []
        for item in lines:
            if item.startswith('Subcase'):
                split = item.split()
                subcase_id.append(int(split[2]))
        
        subcase_id = np.array(subcase_id)
        Subcaseschange: List = (np.where(subcase_id[:-1] != subcase_id[1:])[0] + 1).tolist()
        Subcaseschange.insert(0, 0)
        Subcaseschange.append(len(lines))
        subcase_indeces = Subcaseschange

        Subcases = []
        for i in range(len(subcase_indeces) - 1):
            Subcases.append(lines[subcase_indeces[i]:subcase_indeces[i+1]])
        return Subcases
    
    @staticmethod
    def __split_points(lines: List[str]) -> List[List[str]]:
        point_indeces: List[int] = []
        for ind, item in enumerate(lines):
            if item.startswith('Subcase'):
                point_indeces.append(ind)

        point_indeces.append(len(lines))

        points:List[List[str]] = []
        for i in range(len(point_indeces) -1 ):
            points.append(lines[point_indeces[i]: point_indeces[i+1]])
        return points
    
    @staticmethod
    def __read_analysis_point(point: List[str]) -> FlutterAnalysisPoint :
        def remove_letters(input_string: str) -> str:
            tempresult = ''       
            # Iterate through each character in the input string
            for char in input_string:
                # Check if the character is not a letter
                if not char.isalpha():
                    tempresult += char
            
            return tempresult.replace('+', 'E+')
        
        def read_header(header: str)-> Tuple[int, float, float, str]:
            split = header.strip().split('=')
            # assert len(split) == 12, 'length of header of analysis point incorrect'
            headerdata =[remove_letters(item) for item in split[1:-1]]
            headerdata.append(split[-1].strip())
            out = (int(headerdata[0]), float(headerdata[1]), float(headerdata[2]), str(headerdata[3]))
            return out
        
        def read_data(data: List[str]) -> pd.DataFrame:
                # Initialize an empty list to store rows
            rows = []
            
            # Iterate over each string in the input list
            for string in data:
                # Split the string into numbers
                numbers = string.split()
                row = [float(num) for num in numbers]
                rows.append(row)
            
            # Create a DataFrame from the list of rows
            df = pd.DataFrame(rows, columns = ['KFREQ', '1/KFREQ', 'VELOCITY', 'DAMPING', 'FREQUENCY', 'COMPLEX', 'EIGENVALUE'])
            return df

        header = point[3]
        data = point[5:]

        header = read_header(header)
        data = read_data(data)
        return FlutterAnalysisPoint(*header, data)
    
    @staticmethod
    def __read_subcase_header(subcase_header: List[str]) -> Tuple[int, bool, bool]:
        Id = int(subcase_header[0].split('=')[1])
        row3 = subcase_header[2]
        row3eqsplit = row3.split('=')
        row3split = []
        for s in row3eqsplit:
            row3split.extend(s.split())
        
        XY_Symmetry: bool = False
        XZ_Symmetry: bool = False
        if row3split[2] == 'SYMMETRIC':
            XY_Symmetry = True
        
        if row3split[5] == 'SYMMETRIC':
            XZ_Symmetry = True
        return (Id, XY_Symmetry, XZ_Symmetry)

    def __init__(self, filepath: str):
        self.title: str = ''
        self.Filepath: str = ''
        self.NumSubcases: int = 0
        self.Subcases: List[FlutterSubcase] = []

        assert os.path.isfile(filepath), f'File at {filepath} does not exist'

        #Title
        self.title = os.path.basename(filepath).split('.')[0]
        
        #Filepath
        self.Filepath = filepath

        #Subcases
        with open(filepath, 'r') as file:
            lines = file.readlines()
        stripedlines = []
        for line in lines:
            if line.isspace():
                lines.remove(line)
            else:
                stripedlines.append(line.strip())
        
        # Split on Subcase level
        subcases = self.__split_subcases(stripedlines)
        Subcases: List[FlutterSubcase] = [] 
        for subcase in subcases:
            subcase_header = subcase[0:3] 
            SubcaseHeader = self.__read_subcase_header(subcase_header)
            points = self.__split_points(subcase)
            Points: List[FlutterAnalysisPoint] = []
            for point in points:
                Points.append(self.__read_analysis_point(point))
            
            SubcaseMethod = Points[0].Method
            Subcases.append(FlutterSubcase(*SubcaseHeader, SubcaseMethod, len(Points), Points))
        
        self.Subcases = Subcases

        #Number of Subcases
        self.NumSubcases = len(Subcases)

    def FlutterInfo(self) -> str:
        s = f'\n\n********* FLUTTER INFORMATION *********\n\n'
        for subcase in self.Subcases:
            s += f'|------ SUBCASE {subcase.SubcaseId} ------|\n'
            s += '|                       |\n'
            for point, vel in subcase.FlutterInfo().items():
                s += f'| POINT {point} -> {round(vel, 2)} m/s |\n'
        return s

class PlySymmetry(Enum):
    AntiSymmetric = -1
    NoSymmetry = 0
    Symmetric = 1
    
class ToleranceWrapper():
    def __init__(self, func: Callable,
                  ThicknessTolerance: float,
                      AngleTolerance: float,
                        FlutterVelocityConstraint: float,
                          inputfile: str,
                            solverpath: str,
                              Penalty: float = 1E10,
                                PlySymmetry: PlySymmetry = PlySymmetry.AntiSymmetric):
        self.func = func
        self.ThicknessTolerance = ThicknessTolerance
        self.AngleTolerance = AngleTolerance
        self.FlutterVelocityConstraint = FlutterVelocityConstraint
        self.inputfile = inputfile
        self.solverpath = solverpath
        self.Penalty = Penalty
        self.PlySymmetry = PlySymmetry
        self.cache: Dict[Tuple[float], float] = {}
        self.history: Dict[Tuple[float], float] = {}
    
    def __call__(self, x: NDArray[np.float64]) -> float:
        t = x[0]
        t = np.round(t / self.ThicknessTolerance) * self.ThicknessTolerance

        a = x[1:]
        a = (np.round(a / self.AngleTolerance) * self.AngleTolerance).tolist()
        NumLayers = len(a)
        t = NumLayers * [t]
        rounded_input = (*t, *a)
        if rounded_input in self.cache:
            result = self.cache[rounded_input]
            self.history[rounded_input] = result
            return result
        else:
            result = self.func(t, a, self.inputfile, self.solverpath, self.FlutterVelocityConstraint, self.PlySymmetry, self.Penalty)
            self.history[rounded_input] = result
            self.cache[rounded_input] = result
            return result
    
    def __str__(self) -> str:
        s = f'Thickness Tolerance: {self.ThicknessTolerance}\n'
        s += f'Angle Tolerance: {self.AngleTolerance}\n'
        s += '\nCached Data:\n'
        for k, v in self.cache.items():
            s += f'x = {k} -> f(x) = {v}\n'
        return s
    
    def savecahce(self, file: str) -> None:
        assert file.endswith('.xlsx'), f'The file must be an .xlsx file not a .{file.split('.')[1]} file'
        Dataframe = pd.DataFrame(list(self.cache.items()), columns = ['Input Vector', 'Function Value'])
        Dataframe.to_excel(file)
        return
    
    def savehistory(self, file: str) -> None:
        assert file.endswith('.xlsx'), f'The file must be an .xlsx file not a .{file.split('.')[1]} file'
        Dataframe = pd.DataFrame(list(self.history.items()), columns = ['Input Vector', 'Function Value'])
        Dataframe.to_excel(file)
        return

# -------- Functions --------------
def ReadFem(inputfile: str) -> Tuple[List[PCOMP], List[MAT8]]:
    '''This funciton reads a .fem optistruct iinput file and detects any PCOMP and any MAT8 entries. This function assumes that PCOMP refers exlusively to MAT8 material types.
     ### Parameters:
     - inputfile: a file path to a .fem file given as a string
     ### Returns:
     - A List of PCOMP objects
     - A list of MAT8 objects '''
    def readply(PlyIndex: int, input: List[str]) -> Ply:
        Mid = int(input[0])
        Thickness = float(input[1])
        Theta = float(input[2])
        return Ply(PlyIndex, Mid, Thickness, Theta)
    
    def readPCOMP(StartIndex: List[int]) -> List[PCOMP]:
        numPCOMP = len(StartIndex)
        EndIndex: List[int] = []
        for index in StartIndex:
            i = 1
            while lines[index + i].startswith('+'):
                i += 1
            
            EndIndex.append(index + i)

        PCOMPs: List[PCOMP] = []

        for i in range(numPCOMP):
            Section = lines[StartIndex[i] : EndIndex[i]]
            Id = int(Section[0].split(',')[1])
            Plies_str: List[List[str]] = []
            for line in Section[1:]:
                SplitLine = line.split(',')
                lenSplit = len(SplitLine)
                if lenSplit == 10:
                    if SplitLine[8] == 'YES' or SplitLine[8] == 'NO':
                        Plies_str.append(SplitLine[1:5])
                        Plies_str.append(SplitLine[5:9])
                    else:
                        pass
                elif lenSplit == 6:
                    if SplitLine[4] == 'YES' or SplitLine[4] == 'NO':
                        Plies_str.append(SplitLine[1:5])
            
            Plies: List[Ply] = []
            for Plyindex, ply in enumerate(Plies_str):
                Plies.append(readply(Plyindex, ply))
                
            PCOMPs.append(PCOMP(Id, Plies, Section, (StartIndex[i], EndIndex[i]) ))
        return PCOMPs

    def readMAT8(Indeces: List[int]) -> List[MAT8]:
        MAT8s: List[MAT8] = []
        for index in Indeces:
            line = lines[index]
            split = line.split(',')[1:]
            MID = int(split[0])
            E1 = float(split[1])
            E2 = float(split[2])
            NU12 = float(split[3])
            G12 = float(split[4])
            G1Z = split[5]
            G2Z = split[6]
            RHO = float(split[7])
            M = MAT8(index, MID,E1,E2,NU12,G12,G1Z,G2Z,RHO)
            MAT8s.append(M)

        return MAT8s

    if not inputfile.endswith('.fem'):
        raise ValueError('Incorrect file type. File must be of type .fem')
    
    with open(inputfile, 'r') as f:
        lines = f.readlines()

    StartIndex: List[int] = []
    Mat8Indeces: List[int] = []

    for index, line in enumerate(lines):
        if line.startswith('PCOMP'):
            StartIndex.append(index)
        elif line.startswith('MAT8'):
            Mat8Indeces.append(index)
    
    PCOMPs = readPCOMP(StartIndex)
    MAT8s = readMAT8(Mat8Indeces)

    return PCOMPs, MAT8s

def WriteFem(Properties: List[PCOMP], Materials: List[MAT8], inputfile: str) -> None:
    '''Writes a back to a .fem file the modified PCOMP and MAT8 objects given in the input. It is designed to be used on the same file as ReadFem function
    ### Parameters:
    - Properties: A list of PCOMP objects
    - Materials: A list of MAT 8 objects
    - inputfile: A path to a file as a string to which the objects will be returned
    '''
    def replace_lines_in_file(file_path: str, line_numbers : List[int], new_lines: List[str]):
        if len(line_numbers) != len(new_lines):
            raise ValueError("The number of line numbers must match the number of new lines.")

        try:
            # Read all lines from the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Replace the specified lines
            for line_number, new_line in zip(line_numbers, new_lines):
                if 0 <= line_number < len(lines):
                    lines[line_number] = new_line + '\n'
                else:
                    Warning(f"Warning: Line number {line_number} is out of range. Skipping this replacement.")

            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)

            print("Lines replaced successfully.")

        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    if Properties:
        for Pcomp in Properties:
            LineIndeces = list(range(Pcomp.Indeces[0], Pcomp.Indeces[1]))
            replace_lines_in_file(inputfile, LineIndeces, Pcomp.to_string())
    
    if Materials:
        for Mat in Materials:
            replace_lines_in_file(inputfile, [Mat.LineIndex], [Mat.to_string()])

def readmass(outputfile: str) -> float:
    ''' Reads a .out file and checks if it contans Mass infirmation if it does it returns the mass as a float otherwise it throws a Value Error
    ### Parameters:
    - outputfile: A path as a string to a .out file'''
    if not outputfile.endswith('.out'):
        raise ValueError('Incorrect file type. File must be of type .out')
    
    with open(outputfile, 'r') as f:
        lines = f.readlines()
    
    Mass: float = -1.0
    for line in lines:
        if 'Mass' in line:
            MassString = line.split('Mass')[1]
            MassString = MassString.replace('=', '')
            MassString = MassString.strip()
            Mass = float(MassString)
            break

    if Mass == -1.0: raise ValueError('Provided file does not contain any Mass information')

    return Mass

def CallSolver(inputfile: str, solverpath: str, options: str) -> subprocess.CompletedProcess:
    '''This function calls the optistruct solver on the input file.
    The function works by writing a temporary batch file the excecuting it and finally deleting it.
    It returns the std out to the consoleand the completedProcess object provided by the subprocess module
    ### Parameters:
    - inputfile: a path as a string to a .fem solver input file 
    - solverpath: the path to the altair yperwors solver scripts as a string
    - options: a string of options excactly as they would be set in the command window or in the altair compute consosle'''
    inputfile = f'"{inputfile.replace('/', '\\')}"'
    solverpath = f'"{solverpath.replace('/', '\\')}"'


    lines = ['@echo off\n',
             'optistruct ' + inputfile + ' ' + options]
    
    with open('temp.bat', 'w') as file:
        file.writelines(lines)


    s = subprocess.run([f'temp.bat'])
    # os.remove('temp.bat')
    return s

def ObjectiveFunction(thicknesses: List[float], angles: List[float], inputfile: str, solverpath: str, FlutterVelocityConstraint: float, sym: PlySymmetry, penalty: float) -> float:
    assert len(thicknesses) == len(angles), f'Thicknesses and angles must haver the same length'
    assert all([e > 0 for e in thicknesses]), ' All thicknesses must be strictly positive'

    Properties, _ = ReadFem(inputfile)
    assert len(Properties) == 1, f'Function expected only one PCOMP to optimize, {len(Properties)} found'
    Property = Properties[0]
    # Property.to_string()

    match sym.value:
        case -1: #Antisymmetric
            assert Property.NumPlies / 2 == len(thicknesses), f'for antisymmetric laminates the length of the inputs should be half the number of plies'
            thicknesses.extend(thicknesses)
            angles.extend([-e for e in angles])

            for i in range(Property.NumPlies):
                Property.Plies[i].Thickness = thicknesses[i]
                Property.Plies[i].Theta = angles[i]

        case 0: # No symmetry
            assert Property.NumPlies == len(thicknesses), f'length of the number of inputs should be half the number of plies'
            for i in range(Property.NumPlies):
                Property.Plies[i].Thickness = thicknesses[i]
                Property.Plies[i].Theta = angles[i]

        case 1: # Symmetric
            assert Property.NumPlies / 2 == len(thicknesses), f'for symmetric laminates the length of the inputs should be half the number of plies'
            thicknesses.extend(thicknesses)
            angles.extend(angles)

            for i in range(Property.NumPlies): 
                Property.Plies[i].Thickness = thicknesses[i]
                Property.Plies[i].Theta = angles[i]

    
    WriteFem([Property], [], inputfile)
    CallSolver(inputfile, solverpath, '-nt 6')
    outfile = inputfile.replace('.fem', '.out')
    fltfile = inputfile.replace('.fem', '.flt')

    Mass = readmass(outfile)
    Flutter = FlutterSummary(fltfile)
    assert len(Flutter.Subcases) == 1, f'Analysis should include only one subcase not {len(Flutter.Subcases)}'
    Subcase = Flutter.Subcases[0]

    Velocities: List[float] = []
    for point in Subcase.Points:
        Vel, _ = point.DetectFlutter()
        if Vel: Velocities.append(Vel[0])
    
    P: float = 0
    if Velocities:
        FlutterVelocity = min(Velocities)
        if FlutterVelocity < FlutterVelocityConstraint:
            P = penalty * (FlutterVelocityConstraint - FlutterVelocity)

    Objective = Mass + P

    return Objective

def DeleteUnessesaryFiles(directory: str, FileExtensions: Tuple[str, ...]) -> None:
    '''This function deletes al;l file with certain extension within a directory USE CAREFULLY
    ### Parameters:
    - directory: path a string to a directory (folder)
    - FileExtensions: A tuple of string containing the extension that will be deleted including the dot eg: ('.txt', '.exe') '''
    files = os.listdir(directory)

    for file in files:
        if file.endswith(FileExtensions):
            os.remove(os.path.join(directory, file))

def main():
    # Define optimization problem parameters
    inputFile = "C:/Users/vasxen/OneDrive/Thesis/code/ASW28 Wing.fem"
    solverpath = "C:/My_Programms/Altair/hwsolvers/scripts"
    x0 = np.array([0.0005, 45, -45, 45], dtype = np.float64)# initial solution vector
    lower_bounds = [0.0002] + 3 * [-90]                     # lower constraints of thickness and ply angles
    upper_bounds = [0.0008] + 3 * [+90]                     # Upper constraints of thickness and ply angles
    bounds = Bounds(lower_bounds, upper_bounds)             # type: ignore
    options = {'disp' : True,
               'maxfev' : 1000,
               'return_all' : True}
    WrappedObj = ToleranceWrapper(ObjectiveFunction, 0.0001, 1, 90, inputFile, solverpath)
    Min = minimize(WrappedObj, x0 = x0,method = 'powell', bounds = bounds, options = options)
    DeleteUnessesaryFiles(os.path.dirname(inputFile), FileExtensions = ('.out', '.stat', '.mvw'))
    DeleteUnessesaryFiles(os.getcwd(), FileExtensions = ('.bat', ))


    WrappedObj.savecahce('FunctionEvaluations.xlsx')
    WrappedObj.savehistory('OptimizationHistory.xlsx')
    with open('minimization.pkl', 'wb') as f:
        pickle.dump(Min, f)
    
    with open('Optimization Summary.txt', 'w') as f:
        print('\n\n=========== OPTIMIZATION SUMMARY ===========', file = f)
        print(Min, file = f)
        print(FlutterSummary(inputFile.replace('.fem', '.flt')).FlutterInfo(), file = f)


if __name__ == '__main__':
    main()