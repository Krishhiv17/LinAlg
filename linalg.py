import math
from typing import Union, List, Tuple
import random

class CN:
    def __init__(self, real: float, imag: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)
    
    def __str__(self):
        if abs(self.imag) < 1e-10:
            return f"{self.real:.10g}"
        elif abs(self.real) < 1e-10:
            return f"{self.imag:.10g}i"
        else:
            return f"{self.real:.10g} + {self.imag:.10g}i"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = CN(other)
        return CN(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = CN(other)
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return CN(real_part, imag_part)
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return CN(other * self.real, other * self.imag)
        else:
            raise TypeError("Unsupported operand type for *")
            
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = CN(other)
        if other.real == 0 and other.imag == 0:
            raise ValueError("Division by zero")
        denominator = other.real**2 + other.imag**2
        real_part = (self.real * other.real + self.imag * other.imag) / denominator
        imag_part = (self.imag * other.real - self.real * other.imag) / denominator
        return CN(real_part, imag_part)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return CN(self.real - other, self.imag)
        elif isinstance(other, CN):
            return CN(self.real - other.real, self.imag - other.imag)
        else:
            raise TypeError("Unsupported operand type for -")
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return CN(other - self.real, -self.imag)
        else:
            raise TypeError("Unsupported operand type for -")
            
    def __neg__(self):
        return CN(-self.real, -self.imag)
    
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return abs(self.real - other) < 1e-10 and abs(self.imag) < 1e-10
        elif isinstance(other, CN):
            return abs(self.real - other.real) < 1e-10 and abs(self.imag - other.imag) < 1e-10
        return False
        
    def abs(self):
        return math.sqrt(self.real**2 + self.imag**2)
    
    def cc(self):
        return CN(self.real, -self.imag)

    def __abs__(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def sqrt(self):
        r = self.abs()
        theta = math.atan2(self.imag, self.real)
        return CN(r**0.5 * math.cos(theta/2), r**0.5 * math.sin(theta/2))

class vec:
    def __init__(self, field_type: str, dimension: int, values: List[Union[float, CN]]):
        if field_type not in ["real", "complex"]:
            raise ValueError("Field type must be 'real' or 'complex'")
        if len(values) != dimension:
            raise ValueError(f"Expected {dimension} values, got {len(values)}")
        
        self.field_type = field_type
        self.dimension = dimension
        self.values = []
        
        for value in values:
            if field_type == "real":
                if isinstance(value, (int, float)):
                    self.values.append(float(value))
                else:
                    raise ValueError("Real vector must have real values")
            else:
                if isinstance(value, CN):
                    self.values.append(value)
                elif isinstance(value, (int, float)):
                    self.values.append(CN(value))
                else:
                    raise ValueError("Complex vector must have complex values")
    
    def __str__(self):
        return f"[{', '.join(str(x) for x in self.values)}]"
    
    def __add__(self, other):
        if not isinstance(other, vec):
            raise TypeError("Can only add vectors to vectors")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same length for addition")
        if self.field_type != other.field_type:
            raise ValueError("Vectors must have same field type")
        
        sum_values = [a + b for a, b in zip(self.values, other.values)]
        return vec(self.field_type, self.dimension, sum_values)
    
    def __mul__(self, other):
        if isinstance(other, (int, float, CN)):
            return vec(self.field_type, self.dimension, [x * other for x in self.values])
        else:
            raise TypeError("Unsupported operand type for *")
    
    def __rmul__(self, other):
        if isinstance(other, (int, float, CN)):
            return vec(self.field_type, self.dimension, [other * x for x in self.values])
        else:
            raise TypeError("Unsupported operand type for *")
    
    def __sub__(self, other: 'vec') -> 'vec':
        if self.dimension != other.dimension:
            raise ValueError("Cannot subtract vectors of different dimensions")
        if self.field_type != other.field_type:
            raise ValueError("Cannot subtract vectors from different fields")
            
        diff_values = [a - b for a, b in zip(self.values, other.values)]
        return vec(self.field_type, self.dimension, diff_values)
    
    def len(self):
        result = CN(0, 0) if self.field_type == "complex" else 0
        for x in self.values:
            if isinstance(x, CN):
                result = result + (x * x.cc())
            else:
                result = result + (x * x)
        return math.sqrt(result.real) if isinstance(result, CN) else math.sqrt(result)

    def inner_product(self, other: 'vec') -> Union[float, CN]:
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        if self.field_type != other.field_type:
            raise ValueError("Vectors must be in same field")
            
        result = 0 if self.field_type == "real" else CN(0)
        for i in range(self.dimension):
            if self.field_type == "complex":
                result += self.values[i].cc() * other.values[i]
            else:
                result += self.values[i] * other.values[i]
        return result
    
    def norm(self) -> float:
        inner_prod = self.inner_product(self)
        if isinstance(inner_prod, CN):
            return abs(inner_prod)
        return abs(inner_prod) ** 0.5
    
    def normalize(self) -> 'vec':
        norm = self.norm()
        if abs(norm) < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        return vec(self.field_type, self.dimension, 
                  [x / norm for x in self.values])
    
    def is_ortho(self, other: 'vec') -> bool:
        inner_prod = self.inner_product(other)
        return abs(inner_prod) < 1e-10 if isinstance(inner_prod, (int, float)) \
            else abs(inner_prod.real) < 1e-10 and abs(inner_prod.imag) < 1e-10

class mat:
    def __init__(self, field_type: str, rows: int, cols: int, values=None, column_vectors=None):
        if field_type not in ["real", "complex"]:
            raise ValueError("Field type must be 'real' or 'complex'")
        
        self.field_type = field_type
        self.rows = rows
        self.cols = cols
        self.matrix = []
        
        if column_vectors is not None:
            if len(column_vectors) != cols:
                raise ValueError(f"Expected {cols} vectors, got {len(column_vectors)}")
            if any(v.dimension != rows for v in column_vectors):
                raise ValueError("All vectors must have length equal to number of rows")
            if any(v.field_type != field_type for v in column_vectors):
                raise ValueError("All vectors must have same field type")
            
            self.matrix = [[column_vectors[j].values[i] for j in range(cols)] for i in range(rows)]
        
        elif values is not None:
            if len(values) != rows * cols:
                raise ValueError(f"Expected {rows*cols} values, got {len(values)}")
            
            for i in range(rows):
                row = []
                for j in range(cols):
                    value = values[i*cols + j]
                    if field_type == "real":
                        if isinstance(value, (int, float)):
                            row.append(float(value))
                        else:
                            raise ValueError("Real matrix must have real values")
                    else:
                        if isinstance(value, CN):
                            row.append(value)
                        elif isinstance(value, (int, float)):
                            row.append(CN(value))
                        else:
                            raise ValueError("Complex matrix must have complex values")
                self.matrix.append(row)
    
    def __str__(self):
        return "\n".join([" ".join(str(x) for x in row) for row in self.matrix])
    
    def __add__(self, other):
        if not isinstance(other, mat):
            raise TypeError("Can only add matrices to matrices")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for addition")
        if self.field_type != other.field_type:
            raise ValueError("Matrices must have same field type")
        
        sum_values = []
        for i in range(self.rows):
            for j in range(self.cols):
                sum_values.append(self.matrix[i][j] + other.matrix[i][j])
        
        return mat(self.field_type, self.rows, self.cols, sum_values)
    
    def __sub__(self, other: 'mat') -> 'mat':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for subtraction")
        if self.field_type != other.field_type:
            raise ValueError("Matrices must have same field type for subtraction")
            
        diff_values = []
        for i in range(self.rows):
            for j in range(self.cols):
                diff_values.append(self.matrix[i][j] - other.matrix[i][j])
                
        return mat(self.field_type, self.rows, self.cols, diff_values)
    
    def __mul__(self, other):
        if isinstance(other, mat):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions do not match for multiplication")
            if self.field_type != other.field_type:
                raise ValueError("Cannot multiply matrices from different fields")
                
            result = []
            for i in range(self.rows):
                for j in range(other.cols):
                    val = 0 if self.field_type == "real" else CN(0)
                    for k in range(self.cols):
                        val += self.matrix[i][k] * other.matrix[k][j]
                    result.append(val)
            return mat(self.field_type, self.rows, other.cols, result)
            
        elif isinstance(other, vec):
            if self.cols != other.dimension:
                raise ValueError("Matrix and vector dimensions do not match for multiplication")
            if self.field_type != other.field_type:
                raise ValueError("Cannot multiply matrix and vector from different fields")
                
            result = []
            for i in range(self.rows):
                val = 0 if self.field_type == "real" else CN(0)
                for j in range(self.cols):
                    val += self.matrix[i][j] * other.values[j]
                result.append(val)
            return vec(self.field_type, self.rows, result)
            
        elif isinstance(other, (int, float, CN)):
            return mat(self.field_type, self.rows, self.cols,
                     [x * other for row in self.matrix for x in row])
            
        else:
            raise TypeError("Can only multiply matrix with matrix, vector, or scalar")
    
    def __rmul__(self, other):
        if isinstance(other, (int, float, CN)):
            return mat(self.field_type, self.rows, self.cols,
                      [other * x for row in self.matrix for x in row])
        else:
            raise TypeError("Can only right multiply matrix with scalar")
    
    def get_row(self, row_index: int):
        if row_index < 0 or row_index >= self.rows:
            raise ValueError(f"Row index {row_index} out of range")
        return vec(self.field_type, self.cols, self.matrix[row_index])
    
    def get_column(self, col_index: int):
        if col_index < 0 or col_index >= self.cols:
            raise ValueError(f"Column index {col_index} out of range")
        return vec(self.field_type, self.rows, [self.matrix[i][col_index] for i in range(self.rows)])
    
    def transpose(self):
        transposed_values = [self.matrix[i][j] for j in range(self.cols) for i in range(self.rows)]
        return mat(self.field_type, self.cols, self.rows, transposed_values)
    
    def conj(self):
        if self.field_type == "real":
            return mat(self.field_type, self.rows, self.cols, [x for row in self.matrix for x in row])
        conjugate_values = [value.cc() for row in self.matrix for value in row]
        return mat(self.field_type, self.rows, self.cols, conjugate_values)
    
    def conj_transpose(self):
        return self.transpose().conj()

    def is_zero(self):
        zero = 0 if self.field_type == "real" else CN(0)
        return all(all(x == zero for x in row) for row in self.matrix)
    
    def is_symmetric(self):
        if not self.is_square():
            return False
        return all(self.matrix[i][j] == self.matrix[j][i] 
                  for i in range(self.rows) 
                  for j in range(i + 1, self.cols))
    
    def is_hermitian(self):
        if not self.is_square():
            return False
        return all(self.matrix[i][j] == self.matrix[j][i].cc() if self.field_type == "complex"
                  else self.matrix[i][j] == self.matrix[j][i]
                  for i in range(self.rows) 
                  for j in range(i + 1, self.cols))
    
    def is_square(self):
        return self.rows == self.cols
    
    def is_orthogonal(self):
        if not self.is_square() or self.field_type != "real":
            return False
        product = self * self.transpose()
        return product.is_identity()
    
    def is_unitary(self):
        if not self.is_square():
            return False
        product = self * self.conj_transpose()
        return product.is_identity()
    
    def is_scalar(self):
        if not self.is_square():
            return False
        zero = 0 if self.field_type == "real" else CN(0)
        diagonal_value = self.matrix[0][0]
        return all(self.matrix[i][j] == (diagonal_value if i == j else zero)
                  for i in range(self.rows)
                  for j in range(self.cols))
    
    def determinant(self):
        if not self.is_square():
            raise ValueError("Determinant only defined for square matrices")
        
        if self.rows == 1:
            return self.matrix[0][0]
        
        if self.rows == 2:
            return (self.matrix[0][0] * self.matrix[1][1] - 
                   self.matrix[0][1] * self.matrix[1][0])
        
        det = 0 if self.field_type == "real" else CN(0)
        for j in range(self.cols):
            minor_matrix = []
            for i in range(1, self.rows):
                row = [self.matrix[i][k] for k in range(self.cols) if k != j]
                minor_matrix.append(row)
            submatrix = mat(self.field_type, self.rows-1, self.cols-1, 
                          [x for row in minor_matrix for x in row])
            cofactor = (-1) ** j
            det += self.matrix[0][j] * cofactor * submatrix.determinant()
        return det
    
    def is_singular(self):
        try:
            det = self.determinant()
            return det == (0 if self.field_type == "real" else CN(0))
        except ValueError:
            return True
    
    def is_invertible(self):
        return not self.is_singular()
    
    def is_identity(self):
        if not self.is_square():
            return False
        zero = 0 if self.field_type == "real" else CN(0)
        one = 1 if self.field_type == "real" else CN(1)
        return all(self.matrix[i][j] == (one if i == j else zero)
                  for i in range(self.rows)
                  for j in range(self.cols))
    
    def matrix_power(self, n):
        if not self.is_square():
            raise ValueError("Matrix power only defined for square matrices")
        if n < 0:
            raise ValueError("Negative powers not implemented")
        if n == 0:
            return mat(self.field_type, self.rows, self.cols, 
                      [1 if i == j else 0 for i in range(self.rows) for j in range(self.cols)])
        
        result = self
        for _ in range(n - 1):
            result = result * self
        return result
    
    def is_nilpotent(self, max_power=None):
        if not self.is_square():
            return False
        if max_power is None:
            max_power = self.rows
        
        current = self
        zero_mat = mat(self.field_type, self.rows, self.cols, 
                      [0 for _ in range(self.rows * self.cols)])
        
        for power in range(1, max_power + 1):
            if current.is_zero():
                return True
            current = current * self
        return False
    
    def eigenvalues(self):
        if not self.is_square():
            raise ValueError("Eigenvalues only defined for square matrices")
        if self.rows > 2:
            raise NotImplementedError("Eigenvalue computation for matrices larger than 2x2 not implemented")
        
        if self.rows == 1:
            return [self.matrix[0][0]]
        
        # For 2x2 matrix
        a = self.matrix[0][0]
        b = self.matrix[0][1]
        c = self.matrix[1][0]
        d = self.matrix[1][1]
        
        # Characteristic equation: λ² - (a+d)λ + (ad-bc) = 0
        if self.field_type == "real":
            trace = a + d
            det = a*d - b*c
            discriminant = trace**2 - 4*det
            if discriminant < 0:
                real_part = -trace/2
                imag_part = math.sqrt(-discriminant)/2
                return [CN(real_part, imag_part), CN(real_part, -imag_part)]
            else:
                sqrt_disc = math.sqrt(discriminant)
                return [(-trace + sqrt_disc)/2, (-trace - sqrt_disc)/2]
        else:
            raise NotImplementedError("Complex eigenvalue computation not implemented")
    
    def is_diagonalizable(self):
        try:
            eigenvals = self.eigenvalues()
            # For 2x2 or smaller matrices, diagonalizable if:
            # 1. All eigenvalues are real (for real matrices)
            # 2. Matrix is not a scalar multiple of a nilpotent matrix
            return (not self.is_nilpotent() and 
                   (self.field_type == "complex" or 
                    all(isinstance(ev, (int, float)) for ev in eigenvals)))
        except (ValueError, NotImplementedError):
            return False
    
    def is_positive_definite(self):
        if not self.is_square() or not self.is_hermitian():
            return False
        try:
            eigenvals = self.eigenvalues()
            return all(ev > 0 if isinstance(ev, (int, float)) 
                      else (ev.real > 0 and ev.imag == 0) 
                      for ev in eigenvals)
        except (ValueError, NotImplementedError):
            return False
    
    def is_LU(self):
        if not self.is_square():
            return False
        # A matrix has an LU decomposition if all its leading principal minors are non-zero
        for i in range(1, self.rows + 1):
            submatrix = mat(self.field_type, i, i, 
                          [self.matrix[r][c] for r in range(i) for c in range(i)])
            if submatrix.is_singular():
                return False
        return True
    
    def size(self):
        return (self.rows, self.cols)
    
    def rank(self):
        rref_matrix = self.RREF()[0]
        rank = 0
        zero = 0 if self.field_type == "real" else CN(0)
        
        for i in range(self.rows):
            row_has_nonzero = False
            for j in range(self.cols):
                if rref_matrix.matrix[i][j] != zero:
                    row_has_nonzero = True
                    break
            if row_has_nonzero:
                rank += 1
        return rank
    
    def nullity(self):
        return self.cols - self.rank()
    
    def RREF(self, show_steps=False):
        result = mat(self.field_type, self.rows, self.cols, 
                    [x for row in self.matrix for x in row])
        zero = 0 if self.field_type == "real" else CN(0)
        one = 1 if self.field_type == "real" else CN(1)
        elementary_matrices = []
        
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                break
                
            i = r
            while result.matrix[i][lead] == zero:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if lead == self.cols:
                        if show_steps:
                            return result, elementary_matrices
                        return result, []
                        
            if i != r:
                temp_row = result.matrix[r]
                result.matrix[r] = result.matrix[i]
                result.matrix[i] = temp_row
                
                if show_steps:
                    E = mat(self.field_type, self.rows, self.rows,
                           [one if (x == y and x != r and x != i) or
                            (x == r and y == i) or (x == i and y == r)
                            else zero for x in range(self.rows) 
                            for y in range(self.rows)])
                    elementary_matrices.append(E)
            
            pivot = result.matrix[r][lead]
            pivot_abs = pivot if isinstance(pivot, (int, float)) else pivot.abs()
            if pivot_abs < 1e-10:
                raise ValueError("Matrix is not invertible")
            
            for j in range(self.cols):
                result.matrix[r][j] = result.matrix[r][j] / pivot
            
            if show_steps and pivot_abs != 1:
                E = mat(self.field_type, self.rows, self.rows,
                       [one/pivot if x == y == r else
                        (one if x == y else zero)
                        for x in range(self.rows) 
                        for y in range(self.rows)])
                elementary_matrices.append(E)
            
            for i in range(self.rows):
                if i != r:
                    factor = result.matrix[i][lead]
                    if abs(factor if isinstance(factor, (int, float)) else factor.abs()) > 1e-10:
                        for j in range(self.cols):
                            result.matrix[i][j] = result.matrix[i][j] - factor * result.matrix[r][j]
                    
                    if show_steps and factor != zero:
                        E = mat(self.field_type, self.rows, self.rows,
                               [one if x == y else
                                (-factor if x == i and y == r else zero)
                                for x in range(self.rows) 
                                for y in range(self.rows)])
                        elementary_matrices.append(E)
            
            lead += 1
            
        if show_steps:
            return result, elementary_matrices
        return result, []
    
    @staticmethod
    def is_LI(vectors):
        if not all(isinstance(v, vec) for v in vectors):
            raise TypeError("All elements must be vectors")
        if not all(v.dimension == vectors[0].dimension for v in vectors):
            raise ValueError("All vectors must have same dimension")
        if not all(v.field_type == vectors[0].field_type for v in vectors):
            raise ValueError("All vectors must have same field type")
            
        matrix = mat(vectors[0].field_type, len(vectors), vectors[0].dimension,
                    [x for v in vectors for x in v.values])
        
        return matrix.rank() == len(vectors)
    
    @staticmethod
    def dim(vectors):
        if not vectors:
            return 0
        if not all(isinstance(v, vec) for v in vectors):
            raise TypeError("All elements must be vectors")
        if not all(v.dimension == vectors[0].dimension for v in vectors):
            raise ValueError("All vectors must have same dimension")
        if not all(v.field_type == vectors[0].field_type for v in vectors):
            raise ValueError("All vectors must have same field type")
            
        matrix = mat(vectors[0].field_type, len(vectors), vectors[0].dimension,
                    [x for v in vectors for x in v.values])
        
        rref, _ = matrix.RREF()
        basis_vectors = []
        zero = 0 if matrix.field_type == "real" else CN(0)
        
        for i in range(matrix.rows):
            if any(x != zero for x in rref.matrix[i]):
                basis_vectors.append(vec(matrix.field_type, matrix.cols, rref.matrix[i]))
                
        return len(basis_vectors), basis_vectors
    
    def rank_factorization(self):
        rref, _ = self.RREF()
        rank = self.rank()
        
        if rank == 0:
            raise ValueError("Matrix has rank 0, cannot compute rank factorization")
            
        pivot_cols = []
        col = 0
        for row in range(rank):
            while col < self.cols and abs(rref.matrix[row][col]) < 1e-10:
                col += 1
            if col < self.cols:
                pivot_cols.append(col)
                col += 1
        
        C = mat(self.field_type, self.rows, rank, None)
        for i in range(self.rows):
            C.matrix.append([])
            for j in range(rank):
                C.matrix[i].append(self.matrix[i][pivot_cols[j]])
        
        R = mat(self.field_type, rank, self.cols, None)
        for i in range(rank):
            R.matrix.append([])
            for j in range(self.cols):
                R.matrix[i].append(rref.matrix[i][j])
        
        return C, R
    
    def LU(self):
        if not self.is_square():
            raise ValueError("Matrix must be square for LU decomposition")
            
        n = self.rows
        zero = 0 if self.field_type == "real" else CN(0)
        one = 1 if self.field_type == "real" else CN(1)
        
        L = mat(self.field_type, n, n, None)
        U = mat(self.field_type, n, n, None)
        
        for i in range(n):
            L.matrix.append([zero] * n)
            U.matrix.append([zero] * n)
            L.matrix[i][i] = one
            
        for j in range(n):
            U.matrix[0][j] = self.matrix[0][j]
            
        if abs(U.matrix[0][0]) < 1e-10:
            raise ValueError("LU decomposition does not exist without pivoting")
            
        for i in range(1, n):
            L.matrix[i][0] = self.matrix[i][0] / U.matrix[0][0]
            
        for k in range(1, n):
            for j in range(k, n):
                sum_lu = zero
                for s in range(k):
                    sum_lu = sum_lu + L.matrix[k][s] * U.matrix[s][j]
                U.matrix[k][j] = self.matrix[k][j] - sum_lu
                
            if abs(U.matrix[k][k]) < 1e-10:
                raise ValueError("LU decomposition does not exist without pivoting")
                
            for i in range(k+1, n):
                sum_lu = zero
                for s in range(k):
                    sum_lu = sum_lu + L.matrix[i][s] * U.matrix[s][k]
                L.matrix[i][k] = (self.matrix[i][k] - sum_lu) / U.matrix[k][k]
                
        return L, U
    
    def PLU(self):
        if not self.is_square():
            raise ValueError("Matrix must be square for PLU decomposition")
            
        n = self.rows
        P = mat(self.field_type, n, n, [1 if i == j else 0 
                                      for i in range(n) for j in range(n)])
        L = mat(self.field_type, n, n, [1 if i == j else 0 
                                      for i in range(n) for j in range(n)])
        U = mat(self.field_type, n, n, [x for row in self.matrix for x in row])
        
        for j in range(n):
            pivot = abs(U.matrix[j][j])
            pivot_row = j
            
            for i in range(j + 1, n):
                if abs(U.matrix[i][j]) > pivot:
                    pivot = abs(U.matrix[i][j])
                    pivot_row = i
                    
            if pivot_row != j:
                U.matrix[j], U.matrix[pivot_row] = U.matrix[pivot_row], U.matrix[j]
                P.matrix[j], P.matrix[pivot_row] = P.matrix[pivot_row], P.matrix[j]
                if j > 0:
                    for k in range(j):
                        L.matrix[j][k], L.matrix[pivot_row][k] = L.matrix[pivot_row][k], L.matrix[j][k]
                        
            for i in range(j + 1, n):
                if U.matrix[j][j] == 0:
                    continue
                factor = U.matrix[i][j] / U.matrix[j][j]
                L.matrix[i][j] = factor
                for k in range(j, n):
                    U.matrix[i][k] -= factor * U.matrix[j][k]
                    
        return P, L, U


    def inv(self):
        if not self.is_square():
            raise ValueError("Matrix must be square to compute inverse")
        if not self.is_invertible():
            raise ValueError("Matrix is not invertible")
            
        n = self.rows
        zero = 0 if self.field_type == "real" else CN(0, 0)
        one = 1 if self.field_type == "real" else CN(1, 0)
        
        aug = mat(self.field_type, n, 2*n, None)
        for i in range(n):
            aug.matrix.append([])
            for j in range(n):
                aug.matrix[i].append(self.matrix[i][j])
            for j in range(n):
                aug.matrix[i].append(one if i == j else zero)
        
        for i in range(n):
            max_pivot_row = i
            max_pivot_val = aug.matrix[i][i] if isinstance(aug.matrix[i][i], (int, float)) else aug.matrix[i][i].abs()
            
            for k in range(i + 1, n):
                val = aug.matrix[k][i] if isinstance(aug.matrix[k][i], (int, float)) else aug.matrix[k][i].abs()
                if val > max_pivot_val:
                    max_pivot_val = val
                    max_pivot_row = k
            
            if max_pivot_row != i:
                aug.matrix[i], aug.matrix[max_pivot_row] = aug.matrix[max_pivot_row], aug.matrix[i]
            
            pivot = aug.matrix[i][i]
            pivot_abs = pivot if isinstance(pivot, (int, float)) else pivot.abs()
            if pivot_abs < 1e-10:
                raise ValueError("Matrix is not invertible")
            
            for j in range(i, 2*n):
                aug.matrix[i][j] = aug.matrix[i][j] / pivot
            
            for k in range(i + 1, n):
                factor = aug.matrix[k][i]
                if abs(factor if isinstance(factor, (int, float)) else factor.abs()) > 1e-10:
                    for j in range(i, 2*n):
                        aug.matrix[k][j] = aug.matrix[k][j] - factor * aug.matrix[i][j]
        
        for i in range(n-1, -1, -1):
            for k in range(i-1, -1, -1):
                factor = aug.matrix[k][i]
                if abs(factor if isinstance(factor, (int, float)) else factor.abs()) > 1e-10:
                    for j in range(i, 2*n):
                        aug.matrix[k][j] = aug.matrix[k][j] - factor * aug.matrix[i][j]
        
        inv_values = []
        for i in range(n):
            for j in range(n):
                val = aug.matrix[i][j + n]
                if isinstance(val, CN):
                    if abs(val.real) < 1e-10: val.real = 0
                    if abs(val.imag) < 1e-10: val.imag = 0
                elif abs(val) < 1e-10:
                    val = 0
                inv_values.append(val)
                
        return mat(self.field_type, n, n, inv_values)

    def invadj(self):
        if not self.is_square():
            raise ValueError("Matrix must be square to compute inverse")
        if not self.is_invertible():
            raise ValueError("Matrix is not invertible")
            
        n = self.rows
        det = self.determinant()
        
        cofactor = mat(self.field_type, n, n, None)
        for i in range(n):
            cofactor.matrix.append([])
            for j in range(n):
                minor_values = []
                for r in range(n):
                    if r != i:
                        for c in range(n):
                            if c != j:
                                minor_values.append(self.matrix[r][c])
                                
                minor = mat(self.field_type, n-1, n-1, minor_values)
                sign = 1 if (i + j) % 2 == 0 else -1
                cofactor.matrix[i].append(sign * minor.determinant())
        
        adj = cofactor.transpose()
        
        inv_values = []
        for i in range(n):
            for j in range(n):
                inv_values.append(adj.matrix[i][j] / det)
                
        return mat(self.field_type, n, n, inv_values)


    def det_cofactor(self):
        if not self.is_square():
            raise ValueError("Matrix must be square to compute determinant")
            
        n = self.rows
        I = mat(self.field_type, n, n, [1 if i == j else 0 for i in range(n) for j in range(n)])

        if n == 2:
            a, b = self.matrix[0][0], self.matrix[0][1]
            c, d = self.matrix[1][0], self.matrix[1][1]
            return [
                1,                         
                -(a + d),                 
                (a * d - b * c)            
            ]
        
        coeffs = [1]  
        M = mat(self.field_type, n, n, [0] * (n * n))  
        
        for k in range(1, n + 1):
            if k == 1:
                M = self
            else:
                M = self * M
        
            trace = sum(M.matrix[i][i] for i in range(n))
            
            coeff = -trace / k
            coeffs.append(coeff)
            
            if k < n:
                M = self * M + coeff * I
    
        return coeffs

def is_in_linear_span(S, v):
    if not S or not isinstance(S[0], vec) or not isinstance(v, vec):
        raise ValueError("Invalid input: S must be a non-empty list of vectors and v must be a vector")
    
    n = len(S[0].values)
    m = len(S)
    A_values = []
    for vector in S:
        if len(vector.values) != n:
            raise ValueError("All vectors must have the same dimension")
        A_values.extend(vector.values)
    
    A = mat(v.field_type, m, n, A_values).transpose()
    b = vec(v.field_type, n, v.values)
    
    system = linearsystem(A, b)
    return system.is_consistent()

def express_in(S, v):
    if not S or not isinstance(S[0], vec) or not isinstance(v, vec):
        raise ValueError("Invalid input: S must be a non-empty list of vectors and v must be a vector")
    
    n = len(S[0].values)
    m = len(S)
    A_values = []
    for vector in S:
        if len(vector.values) != n:
            raise ValueError("All vectors must have the same dimension")
        A_values.extend(vector.values)
    
    A = mat(v.field_type, m, n, A_values).transpose()
    b = vec(v.field_type, n, v.values)
    
    system = linearsystem(A, b)
    if not system.is_consistent():
        raise ValueError("Vector is not in the span of given vectors")
    
    return system.solve_gaussian()

def is_span_equal(S1, S2):
    if not S1 or not S2 or not isinstance(S1[0], vec) or not isinstance(S2[0], vec):
        raise ValueError("Invalid input: S1 and S2 must be non-empty lists of vectors")
    
    for v in S1:
        if not is_in_linear_span(S2, v):
            return False
    
    for v in S2:
        if not is_in_linear_span(S1, v):
            return False
    
    return True

def coord(B, v):
    if not B or not isinstance(B[0], vec) or not isinstance(v, vec):
        raise ValueError("Invalid input: B must be a non-empty list of vectors and v must be a vector")
    
    n = len(B[0].values)
    m = len(B)
    A_values = []
    for vector in B:
        if len(vector.values) != n:
            raise ValueError("All vectors must have the same dimension")
        A_values.extend(vector.values)
    
    A = mat(v.field_type, m, n, A_values).transpose()
    if A.rank() != len(B):
        raise ValueError("Given vectors do not form a basis")
    
    return express_in(B, v)

def vector_from_coord(B, coordinates):
    if not B or not isinstance(B[0], vec):
        raise ValueError("Invalid input: B must be a non-empty list of vectors")
    
    if len(B) != len(coordinates.values):
        raise ValueError("Number of coordinates must match basis size")
    
    result = vec(B[0].field_type, len(B[0].values), [0] * len(B[0].values))
    for i, coeff in enumerate(coordinates.values):
        result = result + B[i] * coeff
    
    return result

def COB(B1, B2):
    if not B1 or not B2 or not isinstance(B1[0], vec) or not isinstance(B2[0], vec):
        raise ValueError("Invalid input: B1 and B2 must be non-empty lists of vectors")
    
    if len(B1) != len(B2):
        raise ValueError("Bases must have the same size")
    
    if not is_span_equal(B1, B2):
        raise ValueError("Given sets do not span the same space")
    
    n = len(B1)
    P_values = []
    for v in B1:
        coords = coord(B2, v)
        P_values.extend(coords.values)
    
    return mat(B1[0].field_type, n, n, P_values)

def change_basis(v_coords, B1, B2):
    if not isinstance(v_coords, vec):
        raise ValueError("Coordinates must be given as a vector")
    
    P = COB(B1, B2)
    
    v_coords_matrix = mat(v_coords.field_type, len(v_coords.values), 1, v_coords.values)
    new_coords_matrix = P * v_coords_matrix
    
    return vec(v_coords.field_type, len(v_coords.values), [x for row in new_coords_matrix.matrix for x in row])

def gram_schmidt(vectors: List[vec]) -> List[vec]:
    if not vectors:
        return []
        
    dim = vectors[0].dimension
    field = vectors[0].field_type
    for v in vectors:
        if v.dimension != dim or v.field_type != field:
            raise ValueError("All vectors must have same dimension and field")
    
    orthogonal = []
    for v in vectors:
        u = vec(field, dim, v.values)
        
        for w in orthogonal:
            coeff = w.inner_product(v) / w.inner_product(w)
            u = u - w * coeff
            
        if u.norm() > 1e-10:
            orthogonal.append(u)
    
    return [v.normalize() for v in orthogonal]

def qr_factorization(A: mat) -> Tuple[mat, mat]:
    if A.rows < A.cols:
        raise ValueError("Matrix must have at least as many rows as columns")
        
    vectors = []
    for j in range(A.cols):
        col = [A.matrix[i][j] for i in range(A.rows)]
        vectors.append(vec(A.field_type, A.rows, col))
    
    Q_cols = gram_schmidt(vectors)
    
    Q_values = []
    for i in range(A.rows):
        for q in Q_cols:
            Q_values.append(q.values[i])
    Q = mat(A.field_type, A.rows, len(Q_cols), Q_values)
    
    R = Q.transpose() * A
    
    return Q, R

def pseudo_inverse(A: mat) -> mat:
    
    AT = A.transpose()
    
    if A.rows >= A.cols:  
        ATA = AT * A
        try:
            return ATA.inv() * AT
        except ValueError:
            pass
    
    if A.rows <= A.cols:  
        AAT = A * AT
        try:
            return AT * AAT.inv()
        except ValueError:
            pass
    
    raise ValueError("Matrix is rank deficient, SVD method not implemented")

def least_squares(A: mat, b: vec) -> vec:
    A_plus = pseudo_inverse(A)
    return A_plus * b


def poly_roots(coeffs: List[Union[float, CN]], max_iter=100, tol=1e-10):
    n = len(coeffs) - 1
    if n < 1:
        return []
    
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        if isinstance(a, (int, float)):
            a, b, c = CN(a), CN(b), CN(c)
            
        discriminant = b * b - CN(4) * a * c
        sqrt_disc = discriminant.sqrt()
        
        root1 = (-b + sqrt_disc) / (CN(2) * a)
        root2 = (-b - sqrt_disc) / (CN(2) * a)
        
        return [root1, root2]
    
    def poly_eval(x, coeffs):
        n = len(coeffs) - 1
        p = coeffs[0]
        dp = CN(0)
        for i in range(1, n + 1):
            dp = dp * x + i * coeffs[i-1]
            p = p * x + coeffs[i]
        return p, dp
    
    roots = []
    for k in range(n):
        angle = 2 * math.pi * k / n
        roots.append(CN(math.cos(angle), math.sin(angle)))
    
    for _ in range(max_iter):
        max_change = 0
        for i in range(n):
            p, dp = poly_eval(roots[i], coeffs)
            
            if abs(dp) < tol:
                continue
            
            newton = p / dp
            
            sum_term = CN(0)
            for j in range(n):
                if i != j:
                    diff = roots[i] - roots[j]
                    if abs(diff) > tol:
                        sum_term += CN(1) / diff
            
            correction = newton / (CN(1) - newton * sum_term)
            
            if abs(correction) > 1:
                correction = correction * (1 / abs(correction))
            
            roots[i] = roots[i] - correction
            max_change = max(max_change, abs(correction))
        
        if max_change < tol:
            break
    
    for i in range(n):
        for _ in range(2):  
            p, dp = poly_eval(roots[i], coeffs)
            if abs(dp) > tol:
                roots[i] = roots[i] - p / dp
    
    roots.sort(key=lambda x: (x.real, x.imag))
    return roots

def char_poly(A: mat) -> List[Union[float, CN]]:
    
    if not A.is_square():
        raise ValueError("Matrix must be square")
    
    n = A.rows
    I = mat(A.field_type, n, n, [1 if i == j else 0 for i in range(n) for j in range(n)])
    
    if n == 2:
        a, b = A.matrix[0][0], A.matrix[0][1]
        c, d = A.matrix[1][0], A.matrix[1][1]
        return [
            1,                         
            -(a + d),                  
            (a * d - b * c)             
        ]
    
    coeffs = [1]  
    M = mat(A.field_type, n, n, [0] * (n * n))  
    
    for k in range(1, n + 1):
        if k == 1:
            M = A
        else:
            M = A * M
        
        trace = sum(M.matrix[i][i] for i in range(n))
        
        coeff = -trace / k
        coeffs.append(coeff)
        
        if k < n:
            M = A * M + coeff * I
    
    return coeffs

def eigenvalues(A: mat) -> List[CN]:
    if not A.is_square():
        raise ValueError("Matrix must be square")
        
    if A.rows == 2:
        a, b = A.matrix[0][0], A.matrix[0][1]
        c, d = A.matrix[1][0], A.matrix[1][1]
        
        trace = a + d
        det = a * d - b * c
        
        discriminant = trace * trace - 4 * det
        
        if A.field_type == "real" and discriminant < 0:
            real_part = trace / 2
            imag_part = math.sqrt(-discriminant) / 2
            return [CN(real_part, imag_part), CN(real_part, -imag_part)]
        else:
            sqrt_disc = math.sqrt(discriminant) if discriminant >= 0 else 0
            return [
                CN((trace + sqrt_disc) / 2),
                CN((trace - sqrt_disc) / 2)
            ]
    
    coeffs = char_poly(A)
    return poly_roots(coeffs)

def eigen_basis(A: mat, lambda_val: Union[float, CN], tol=1e-10) -> List[vec]:
    n = A.rows
    
    field_type = "complex" if isinstance(lambda_val, CN) else A.field_type
    if field_type == "complex":
        if A.field_type == "real":
            A = mat("complex", n, n, [CN(x) for row in A.matrix for x in row])
        if not isinstance(lambda_val, CN):
            lambda_val = CN(lambda_val)
    
    I = mat(field_type, n, n, [1 if i == j else 0 for i in range(n) for j in range(n)])
    A_lambda = A - I * lambda_val
    
    basis = nullspace(A_lambda, tol)
    
    result = []
    for v in basis:
        Av = A * v
        lambda_v = v * lambda_val
        if (Av - lambda_v).norm() < tol:
            result.append(v)
    
    return result

def nullspace(A: mat, tol=1e-10) -> List[vec]:
    rref, pivots = A.RREF()
    n = A.cols
    
    pivot_cols = []
    free_cols = []
    col = 0
    for row in range(min(rref.rows, rref.cols)):
        while col < n and abs(rref.matrix[row][col]) < tol:
            free_cols.append(col)
            col += 1
        if col < n:
            pivot_cols.append(col)
            col += 1
    while col < n:
        free_cols.append(col)
        col += 1
    
    if not free_cols:
        return []
    
    basis = []
    for free_col in free_cols:
        x = [CN(0) if A.field_type == "complex" else 0] * n
        x[free_col] = CN(1) if A.field_type == "complex" else 1
        
        for row in range(len(pivot_cols)-1, -1, -1):
            pivot_col = pivot_cols[row]
            
            if A.field_type == "complex":
                sum_val = CN(0)
                for j in range(pivot_col + 1, n):
                    if isinstance(rref.matrix[row][j], CN):
                        val = -rref.matrix[row][j]
                    else:
                        val = CN(-rref.matrix[row][j])
                    
                    if isinstance(x[j], CN):
                        term = val * x[j]
                    else:
                        term = val * CN(x[j])
                    sum_val += term
            else:
                sum_val = 0
                for j in range(pivot_col + 1, n):
                    sum_val += -rref.matrix[row][j] * x[j]
            
            x[pivot_col] = sum_val
        
        v = vec(A.field_type, n, x)
        if v.norm() > tol:  
            first_nonzero = None
            for val in v.values:
                if abs(val) > tol:
                    first_nonzero = val
                    break
            
            if first_nonzero is not None:
                if A.field_type == "complex":
                    if not isinstance(first_nonzero, CN):
                        first_nonzero = CN(first_nonzero)
                    scale = CN(1) / first_nonzero
                else:
                    scale = 1 / first_nonzero
                
                v = v * scale
                basis.append(v)
    
    return basis

def geo_mul(A: mat, lambda_val: Union[float, CN], tol=1e-10) -> int:
    return len(eigen_basis(A, lambda_val, tol))

def alg_mul(A: mat, lambda_val: Union[float, CN], tol=1e-10) -> int:
    eigenvals = eigenvalues(A)
    return sum(1 for val in eigenvals if abs(val - lambda_val) < tol)

def is_similar(A: mat, B: mat) -> bool:
    if not (A.is_square() and B.is_square() and A.rows == B.rows):
        return False
    
    poly_A = char_poly(A)
    poly_B = char_poly(B)
    
    return all(abs(a - b) < 1e-10 for a, b in zip(poly_A, poly_B))

def COB_similar(A: mat, B: mat) -> mat:
    if not is_similar(A, B):
        raise ValueError("Matrices are not similar")
    
    n = A.rows
    eigenvals = eigenvalues(A)
    P_cols = []
    
    for lambda_val in eigenvals:
        A_vectors = eigen_basis(A, lambda_val)
        B_vectors = eigen_basis(B, lambda_val)
        
        if len(A_vectors) != len(B_vectors):
            raise ValueError("Matrices have different geometric multiplicities")
        
        P_cols.extend(A_vectors)
    
    if len(P_cols) != n:
        raise ValueError("Could not find enough linearly independent eigenvectors")
    
    P_values = []
    for i in range(n):
        for j in range(n):
            P_values.append(P_cols[j].values[i])
    
    return mat(A.field_type, n, n, P_values)

def COB_diag(A: mat) -> mat:
    n = A.rows
    eigenvals = eigenvalues(A)
    P_cols = []
    
    for lambda_val in eigenvals:
        vectors = eigen_basis(A, lambda_val)
        P_cols.extend(vectors)
    
    if len(P_cols) != n:
        raise ValueError("Matrix is not diagonalizable")
    
    P_values = []
    for i in range(n):
        for j in range(n):
            P_values.append(P_cols[j].values[i])
    
    return mat(A.field_type, n, n, P_values)

def min_poly(A: mat) -> List[CN]:
    n = A.rows
    powers = [mat(A.field_type, n, n, [1 if i == j else 0 for i in range(n) for j in range(n)])]
    
    curr_power = mat(A.field_type, n, n, [x for row in A.matrix for x in row])
    while True:
        vectors = []
        for p in powers:
            vectors.extend([p.matrix[i][j] for i in range(n) for j in range(n)])
        vectors.extend([curr_power.matrix[i][j] for i in range(n) for j in range(n)])
        
        coeff_matrix = mat(A.field_type, len(vectors) // (n*n), n*n, vectors)
        rref, _ = coeff_matrix.RREF()
        
        rank = sum(1 for i in range(min(rref.rows, rref.cols)) 
                  if abs(rref.matrix[i][i]) > 1e-10)
        
        if rank < len(powers) + 1:
            break
            
        powers.append(curr_power)
        curr_power = curr_power * A
    
    coeffs = [-x for x in rref.matrix[-1]]
    coeffs.append(CN(1))  
    
    return coeffs


def polar_decomposition(A: mat) -> Tuple[mat, mat]:
    if A.rows != A.cols:
        raise ValueError("Polar decomposition requires a square matrix")
    
    n = A.rows
    AH = A.conjugate_transpose() if A.field_type == "complex" else A.transpose()
    AHA = AH * A
    
    eigenvals = eigenvalues(AHA)
    
    eig_pairs = []
    for lambda_val in eigenvals:
        eigenvecs = eigen_basis(AHA, lambda_val)
        for v in eigenvecs:
            norm = sum(abs(x.real ** 2 + x.imag ** 2) if isinstance(x, CN) else x ** 2 for x in v.values) ** 0.5
            v = v * (1/norm)
            if isinstance(lambda_val, CN):
                if abs(lambda_val.imag) < 1e-10:
                    eig_pairs.append((lambda_val.real, v))
            else:
                eig_pairs.append((lambda_val, v))
    
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    P = mat(A.field_type, n, n, [0] * (n * n))
    
    for lambda_val, v in eig_pairs:
        sqrt_lambda = abs(lambda_val) ** 0.5
        
        for i in range(n):
            for j in range(n):
                vi = v.values[i].real if isinstance(v.values[i], CN) else v.values[i]
                vj = v.values[j].real if isinstance(v.values[j], CN) else v.values[j]
                P.matrix[i][j] += sqrt_lambda * vi * vj
    
    for i in range(n):
        for j in range(i+1, n):
            avg = (P.matrix[i][j] + P.matrix[j][i]) / 2
            P.matrix[i][j] = P.matrix[j][i] = avg
    
    P_inv = P.inv()
    U = A * P_inv
    
    if n == 2:
        det_U = U.matrix[0][0] * U.matrix[1][1] - U.matrix[0][1] * U.matrix[1][0]
        
        if det_U < 0:
            U.matrix[0][1] = -U.matrix[0][1]
            U.matrix[1][0] = -U.matrix[1][0]
    
    for i in range(n):
        for j in range(n):
            if isinstance(U.matrix[i][j], CN):
                U.matrix[i][j] = U.matrix[i][j].real
            if abs(U.matrix[i][j]) < 1e-10:
                U.matrix[i][j] = 0
    
    for i in range(n):
        row_norm = sum(U.matrix[i][j] ** 2 for j in range(n)) ** 0.5
        if abs(row_norm - 1) > 1e-10:
            for j in range(n):
                U.matrix[i][j] /= row_norm
    
    return U, P

def cholesky_decomposition(A: mat) -> mat:
    if A.rows != A.cols:
        raise ValueError("Cholesky decomposition requires a square matrix")
    
    AH = A.conjugate_transpose() if A.field_type == "complex" else A.transpose()
    if not all(abs(A.matrix[i][j] - AH.matrix[i][j]) < 1e-10 for i in range(A.rows) for j in range(A.cols)):
        raise ValueError("Matrix must be Hermitian/symmetric")
    
    n = A.rows
    L = mat(A.field_type, n, n, [0] * (n * n))
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_val = CN(0) if A.field_type == "complex" else 0
                for k in range(j):
                    sum_val += L.matrix[j][k] * L.matrix[j][k].conjugate() if A.field_type == "complex" else L.matrix[j][k] ** 2
                val = A.matrix[j][j] - sum_val
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L.matrix[j][j] = CN(val ** 0.5) if A.field_type == "complex" else val ** 0.5
            else:
                sum_val = CN(0) if A.field_type == "complex" else 0
                for k in range(j):
                    sum_val += L.matrix[i][k] * L.matrix[j][k].conjugate() if A.field_type == "complex" else L.matrix[i][k] * L.matrix[j][k]
                if abs(L.matrix[j][j]) < 1e-10:
                    raise ValueError("Matrix is not positive definite")
                L.matrix[i][j] = (A.matrix[i][j] - sum_val) / L.matrix[j][j]
                
    return L

def svd(A: mat) -> Tuple[mat, List[float], mat]:
    AH = A.conjugate_transpose() if A.field_type == "complex" else A.transpose()
    AHA = AH * A
    
    eigenvals = eigenvalues(AHA)
    
    eig_pairs = []
    for lambda_val in eigenvals:
        eigenvecs = eigen_basis(AHA, lambda_val)
        for v in eigenvecs:
            norm = sum(abs(x.real ** 2 + x.imag ** 2) if isinstance(x, CN) else x ** 2 for x in v.values) ** 0.5
            v = v * (1/norm)
            if isinstance(lambda_val, CN):
                if abs(lambda_val.imag) < 1e-10:
                    eig_pairs.append((lambda_val.real, v))
            else:
                eig_pairs.append((lambda_val, v))
    
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    singular_values = []
    V = mat(A.field_type, A.cols, A.cols, [0] * (A.cols * A.cols))
    col = 0
    
    for lambda_val, v in eig_pairs:
        if lambda_val > 1e-10:
            sigma = abs(lambda_val) ** 0.5
            singular_values.append(sigma)
            
            for i in range(A.cols):
                val = v.values[i]
                if isinstance(val, CN):
                    V.matrix[i][col] = val.real if abs(val.imag) < 1e-10 else val
                else:
                    V.matrix[i][col] = float(val)
            col += 1
    
    U = mat(A.field_type, A.rows, len(singular_values), [0] * (A.rows * len(singular_values)))
    
    for j, sigma in enumerate(singular_values):
        v_col = vec(A.field_type, A.cols, [V.matrix[i][j] for i in range(A.cols)])
        u_col = A * v_col * (1/sigma)
        
        for i in range(A.rows):
            val = u_col.values[i]
            if isinstance(val, CN):
                U.matrix[i][j] = val.real if abs(val.imag) < 1e-10 else val
            else:
                U.matrix[i][j] = float(val)
    
    if U.rows == 2 and U.cols == 2:
        det_U = U.matrix[0][0] * U.matrix[1][1] - U.matrix[0][1] * U.matrix[1][0]
        if det_U < 0:
            U.matrix[0][1] = -U.matrix[0][1]
            U.matrix[1][1] = -U.matrix[1][1]
            V.matrix[0][1] = -V.matrix[0][1]
            V.matrix[1][1] = -V.matrix[1][1]
    
    for i in range(U.rows):
        row_norm = sum(U.matrix[i][j] ** 2 for j in range(U.cols)) ** 0.5
        if abs(row_norm - 1) > 1e-10:
            for j in range(U.cols):
                U.matrix[i][j] /= row_norm
    
    for i in range(V.rows):
        row_norm = sum(V.matrix[i][j] ** 2 for j in range(V.cols)) ** 0.5
        if abs(row_norm - 1) > 1e-10:
            for j in range(V.cols):
                V.matrix[i][j] /= row_norm
    
    if U.rows == 2 and U.cols == 2:
        if U.matrix[0][0] < 0:
            for i in range(U.rows):
                U.matrix[i][0] = -U.matrix[i][0]
            for i in range(V.rows):
                V.matrix[i][0] = -V.matrix[i][0]
        
        if V.matrix[0][1] > 0:
            for i in range(V.rows):
                V.matrix[i][1] = -V.matrix[i][1]
            for i in range(U.rows):
                U.matrix[i][1] = -U.matrix[i][1]
    
    return U, singular_values, V