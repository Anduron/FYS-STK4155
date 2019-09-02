import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

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

def linear():
    main(lambda t: V*t + I)

def quadratic():
    b = sym.symbols('b')
    main(lambda t: b*t**2 + V*t + I)

def qubic():
    a, b = sym.symbols('a b')
    main(lambda t: a*t**3 + b*t**2 + V*t + I)

def solver(n,dt,I,V,w,f):
    """
    Solves the differential equation,
    """
    dt = float(dt)
    n = int(n)
    T = n*dt

    t = np.linspace(0,T,n+1)    #mesh of points in time
    u = np.zeros(n+1)

    u[0] = I
    u[1] = u[0] + V*dt - 0.5*(w**2)*u[0]*(dt**2) + 0.5*(dt**2)*f(t[0])

    for i in range(1,n):
        u[i+1] = u[i]*(2-w**2*dt**2) - u[i-1] + f(t[i])*dt**2

    return u, t

def nosetest_quadratic():
    """
    Noesetest that calculates the quadratic solution and compares it
    with exact solution.
    """

    global V, I, w, b
    I = 2.0; V = 3.0; w = 0.5; b = 100 #b is a large number to see a curve in the plot
    u_e = lambda t: b*(t**2) + V*t + I

    global f, t
    f = ode_source_term(u_e)
    f = sym.lambdify(t,f)

    dt = 0.01; n = 10 #n cannot be too large, this will lead to accumulation of error

    u, t = solver(n=n, dt=dt, I=I, V=V, w=w, f=f)

    e = u - u_e(t)

    eps = 1e-14

    err = np.max(abs(e))

    plt.plot(t,u_e(t),t,u)
    plt.legend(["Exact", "Numerical"], prop={'size':15})
    plt.show()

    print("Max error: %s" %err)
    assert err < eps

if __name__ == '__main__':
    linear()
    quadratic()
    qubic()
    nosetest_quadratic()
