import sympy as sym
import numpy as np
import matplotlib as plt

"""

Flawed program do not use :)

"""





V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u,dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = u(t).subs(t,dt) - I - dt*V + 0.5*(w**2)*I*(dt**2) - 0.5*(dt**2)*f.subs(t,0)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt) -2*u(t) + u(t-dt))/(dt**2)

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """

    print('=== Testing exact solution: %s ===' % u)
    print( "Initial conditions u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f # source term in the ODE
    f = ode_source_term(u)

    # Residual in discrete equations (should be 0)
    print('residual step1:', residual_discrete_eq_step1(u))
    print( 'residual:', residual_discrete_eq(u))


    syms = f.free_symbols

    I = 1.0
    V = 2.0
    w = 0.5
    a = 0.0
    b = 1.0

    global args
    args = np.array([I,V,w, a, b])

def linear():
    main(lambda t: V*t + I)

def quadratic():
    b = sym.symbols('b')
    main(lambda t: b*t**2 + V*t + I)

def qubic():
    a, b = sym.symbols('a b')
    main(lambda t: a*t**3 + b*t**2 + V*t + I)

def solver(n,dt, w,f, args):
    t = np.linspace(0,n,dt)
    u = np.zeros(n)

    h = sym.lambdify(t, args, f)

    u[0] = float(args[0])
    u[1] = u[0] + float(args[2])*dt - 0.5*(float(args[3])**2)*u[0]*(dt**2) + 0.5*(dt**2)*h

    for i in range(1,n):
        u[i+1] = u[i]*(2-w**2*dt**2) - u[i-1] + f*dt**2

    return u, t

if __name__ == '__main__':
    linear()
    quadratic()
    qubic()
    print(f)
    solver(100, 0.1, 1.0, f, args)
