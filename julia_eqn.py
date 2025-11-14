
import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn0(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z*z*z*z*z - z*z*z*z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn1(z: np.complex128,c: np.complex128)->np.complex128:
    return np.exp(1j*2*np.pi*np.abs(z)) + z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn2(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn3(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z*z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn4(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z*z*z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn5(z: np.complex128,c: np.complex128)->np.complex128:
    return np.sin(z)*np.abs(z) + z*z + c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn6(z: np.complex128,c: np.complex128)->np.complex128:
    return (z+1)/(z-1)+c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn7(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z + z/c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn8(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z + c/z

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn9(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z + c/np.conj(z)

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn10(z: np.complex128,c: np.complex128)->np.complex128:
    return z*z + z/c

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn10(z: np.complex128,c: np.complex128)->np.complex128:
    return ( z*z -z + c )/( 3*z*z*z - z*z)

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn11(z: np.complex128,c: np.complex128)->np.complex128:
    return ( z*z + c )/( 3*z - np.exp(z) )

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn12(z: np.complex128,c: np.complex128)->np.complex128:
    return ( z*z*z + c )/( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn13(z: np.complex128,c: np.complex128)->np.complex128:
    den1 = ( z*z*z*z*z + (2.5*c+1j)*z*z*z*z + (1.5-c*1j)*z*z*z + (-0.5+4j)*z*z + z - c + 3j )
    den2 = ( z*z*z*z*z + c*(-0.5+18j)*z*z*z*z + (12-1j)*z*z*z + (-0.5+4j)*z*z + z - c*c*c + 30j )
    num1 = ( z*z*z*z*z + (+15.5+18j)*z*z*z*z + (3-100j)*z*z*z + (-3.5+4j)*z*z + 69*z - 4.0 + 30j )
    num2 = ( z*z*z*z*z + c*c*c*z*z*z*z + c*z*z*z + z*z + z + c )
    return num1*num2/(den1*den2)

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn14(z: np.complex128,c: np.complex128)->np.complex128:
    num = ( z*z*z + c )
    den = ( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )
    for i in range(5):
        num *= ( z + 5*c )*( z - 3j*c)
    for i in range(15):
        den *= ( z*z*z - 5*c )
    return num/den

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn15(z: np.complex128,c: np.complex128)->np.complex128:
    num = ( z*z*z + c*z )
    den = ( z*z*z*z*z + (2.5+1j)*z*z*z*z + (1.5-1j)*z*z*z + (-0.5+4j)*z*z + z - 1.0 + 3j )
    for i in range(20):
        num *= ( c*z + 5*c -1j)*( z*z - 3j*c)
    for i in range(20):
        den *= ( z*z*z - 5*c ) 
    return num/den

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn16(z: np.complex128,c: np.complex128)->np.complex128:
    num = ( z*z*z + c*z )
    den = ( z*z*z*z*z  - 1.0 * c + 3j )
    for i in range(20):
        num *= ( c*z + 5*c -1j)*( z*z - 3j*c) + c * (z.real % 1)
    for i in range(20):
        den *= ( z*z*z - 5*c ) + (z.imag % 1)
    return num/den

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn17(z: np.complex128,c: np.complex128)->np.complex128:
    num = ( z*z*z + c )
    den = ( z*z*z*z*z  - 1.0 * c + 3j )
    for i in range(20):
        num += ( z*z - 3j*c) * ( (c*z).real % 1)
    for i in range(20):
        den += ( z*z*z + 5*c ) * ( (c*z).imag % 1)
    return num/den

@njit("complex128(complex128,complex128)", fastmath=True, cache=True)
def eqn18(z: np.complex128,c: np.complex128)->np.complex128:
    num = ( z*z*z + c ) * np.sin(z)
    den = ( z*z*z*z*z  - 1.0 * c + 3j ) * np.cos(z)
    for i in range(20):
        num *= ( z*z - 3j*c) 
    for i in range(20):
        den *= ( z*z*z + 5*c )
    return num/den

