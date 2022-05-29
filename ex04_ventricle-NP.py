
'''

Coupled large strain poroelasticity and chemotaxis

Holzapfel material law. Fibers are rule-based generated

Replicating initial 2D test now in the ventricle

P1b-P1-P1 discretisation for displ-porosity-pressure
P1-P1 elements for the chemotaxis

Backward Euler scheme for the time discretisation

'''

from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# parameter-dependent functions

varAngle = lambda s : ((thetaEpi - thetaEndo) * s + thetaEndo)/ 180.0 * pi
normalizeAndProj_vec = lambda u : project(u/sqrt(dot(u,u)+0.001),FiberSpace)
normalize_vec = lambda u : u/sqrt(dot(u,u))
subplus = lambda u : conditional(ge(u, 0.0), u, 0.0)
kappa  = lambda J,phi: kappa0*(1 + (1-phi0)**2/phi0**3*J*phi**3*(J-phi)**2)  # isotropic Kozeny-Carman
react_p = lambda phi,cp,cl: phi*cp*(gammap - lmbdrp*cl)
react_l = lambda phi,cp,cl: phi*cp*cl * lmbdpr
ell = lambda p, cp: Sv*Lp0*(1.0+Lbp*cp) * (pc - p - sigma0*(pic-pii)/(1.+Lbp*cp)) \
      - l0*(1.+vmax*(p-p0)**nHill/(km**nHill+(p-p0)**nHill))

# time constants
t = 0.0
dt = 0.05
Tfinal = 40.
freqsave = 50
cont = 0

# ********* model constants ******* #

E = Constant(60.)
nu = Constant(0.35)
lmbdas = Constant(E*nu/((1. + nu)*(1. - 2.*nu)))

rhos = Constant(2.e-3)
rhof = Constant(1.e-3)
kappa0 = Constant(2.5e-7)
phi0 = Constant(0.2)
alpha = Constant(0.3)
muf = Constant(1.e-3)
bb = Constant((0, 0, 0))

Dp = Constant(5.e-3/phi0)
Dl = Constant(1.e-2/phi0)
Lp0 = Constant(3.6e-8)
sigma0 = Constant(0.91)
vmax = Constant(20.)
p0 = Constant(10.9)
pic = Constant(20.)
pii = Constant(10.)
pc = Constant(20.)
km = Constant(0.5)
chi = Constant(0.05)
nHill = Constant(5.)
I = Identity(3)
Lbp = Constant(5000.)
Sv = Constant(174.)
l0 = Constant(6.82e-5)
gammap = Constant(0.13/phi0)
lmbdrp = Constant(1.8/phi0)
lmbdpr = Constant(7.1/phi0)

# Holzapfel-Ogden
a = Constant(0.496)
b = Constant(0.041)
a_f = Constant(0.193)
b_f = Constant(0.176)
a_s = Constant(0.123)
b_s = Constant(0.209)
a_fs = Constant(0.162)
b_fs = Constant(0.166)

# ********* Mesh and I/O ********* #

mesh = Mesh("meshes/newVentr.xml")
bdry = MeshFunction("size_t", mesh, "meshes/newVentr_facet_region.xml")
base = 8
epi = 7
endo = 6
ds = Measure("ds", subdomain_data=bdry)  # epi:7, endo:6, base: 8
nn = FacetNormal(mesh)


fileO = XDMFFile(mesh.mpi_comm(), "outputs/CoupledEdemaVentricle-NP.xdmf")
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ********* Finite dimensional spaces ********* #

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
Bub = FiniteElement("Bubble", mesh.ufl_cell(), 4)
P1b = VectorElement(P1 + Bub)
P2vec = VectorElement("CG", mesh.ufl_cell(), 1)
Hh = FunctionSpace(mesh, MixedElement([P1, P1, P2vec, P1, P1]))

print("**************** Total Dofs = ", Hh.dim())


Sol = Function(Hh)
dSol = TrialFunction(Hh)
cp, cl, u, phi, p = split(Sol)
wp, wl, v, psi, q = TestFunctions(Hh)


Mh = FunctionSpace(mesh, P1)
FiberSpace = VectorFunctionSpace(mesh, "CG", 1)
Vh = FunctionSpace(mesh, P1b)
etah = Function(Mh)

# ******* Generate fibre and sheet directions ********** #

# n0 fibre, s0 sheetlets, n0 normal directions

phif = TrialFunction(Mh)
psif = TestFunction(Mh)
aDif = Constant(10.)
AAf = aDif*dot(grad(phif), grad(psif))*dx
ggf = Constant(0.0)
BBf = ggf * psif * dx
bcphiEn = DirichletBC(Mh, Constant(0.0), bdry, endo)
bcphiEp = DirichletBC(Mh, Constant(1.0), bdry, epi)
bcphi = [bcphiEp, bcphiEn]

solve(AAf == BBf, etah, bcphi,
      solver_parameters={'linear_solver': 'mumps'})

s0 = normalizeAndProj_vec(grad(etah))

k0 = Constant((0, 0, 1))

kp = normalize_vec(k0 - dot(k0, s0) * s0)
f0flat = cross(s0, kp)

thetaEpi = Constant(-20.0)
thetaEndo = Constant(30.0)
f0 = normalizeAndProj_vec(f0flat*cos(varAngle(etah))
                          + cross(s0, f0flat)*sin(varAngle(etah))
                          + s0 * dot(s0, f0flat)*(1.0-cos(varAngle(etah))))
n0 = cross(f0, s0)

# ******* Boundary conditions ********** #

pendo = Expression('0.5*p0*sin(pi/40*t)*sin(pi/40*t)', p0=p0, t=t, degree=0)

u_D = project(Constant((0, 0, 0)), Hh.sub(2).collapse())
bcU = DirichletBC(Hh.sub(2), u_D, bdry, base)
bcP1 = DirichletBC(Hh.sub(4), pendo, bdry, endo)
bcP2 = DirichletBC(Hh.sub(4), p0, bdry, epi)
bcH = [bcU, bcP1, bcP2]

# ******* Initial conditions ********** #

cp0 = Expression(
    '(x[0]-0.13047)*(x[0]-0.13047)+(x[1]-3.05269)*(x[1]-3.05269)<= 0.24 ? 0.002:0.0', degree=1)

cpold = interpolate(cp0, Hh.sub(0).collapse())
clold = interpolate(Constant(0.003), Hh.sub(1).collapse())
uold = project(Constant((0, 0, 0)), Hh.sub(2).collapse())
phiold = project(phi0, Hh.sub(3).collapse())
pold = interpolate(p0, Hh.sub(4).collapse())

# ********  Weak form ********** #

ndim = u.geometric_dimension()

F = I + grad(u)
J = det(F)
invF = inv(F)

C = F.T*F
B = F*F.T
I1 = tr(C)
I8_fs = inner(f0, C*s0)
I4_f = inner(f0, C*f0)
I4_s = inner(s0, C*s0)

sig = a*exp(b*subplus(I1-ndim))*B \
    + 2*a_f*(I4_f-1)*exp(b_f*subplus(I4_f-1)**2)*outer(F*f0, F*f0) \
    + 2*a_s*(I4_s-1)*exp(b_s*subplus(I4_s-1)**2)*outer(F*s0, F*s0) \
    + a_fs*I8_fs*exp(b_fs*subplus(I8_fs)**2)*(outer(F*f0, F*s0)
                                              + outer(F*s0, F*f0))

Peff = J*sig*invF.T + lmbdas*ln(J)*invF.T

# Poromechanics
F1 = inner(Peff - alpha*p*J*invF.T, grad(v)) * dx \
    + psi*(J-1-phi+phi0)*dx \
    + rhof*J/dt*(phi-phiold)*q * dx \
    + dot(phi*J*invF*kappa(J, phi)/muf*invF.T*grad(p), grad(q)) * dx \
    - rhos*dot(bb, v) * dx \
    - rhof*J*ell(p, cp)*q*dx

# Chemotaxis
F2 = J/dt*((phi-phiold)*cp+phi*(cp-cpold))*wp*dx \
    + dot(phi*J*invF*Dp*invF.T*grad(cp), grad(wp)) * dx \
    - J*react_p(phi, cp, cl)*wp*dx \
    + J/dt*((phi-phiold)*cl+phi*(cl-clold))*wl*dx \
    + dot(phi*J*invF*Dl*invF.T*grad(cl), grad(wl)) * dx \
    - dot(phi*chi*cl*J*invF*invF.T*grad(cp), grad(wl)) * dx \
    - J*react_l(phi, cp, cl)*wl*dx

FF = F1 + F2
Tang = derivative(FF, Sol, dSol)

# ********* Time loop ************* #

while (t < Tfinal):

    t += dt

    pendo.t = t

    print("t=%.2f" % t)

    # ********* Solving ************* #
    
    solve(FF == 0, Sol, J=Tang, bcs=bcH,
          solver_parameters={'newton_solver': {'linear_solver': 'mumps',
                                               'absolute_tolerance': 1.0e-7,
                                               'relative_tolerance': 1.0e-7}})
    cph, clh, uh, phih, ph = Sol.split()

    # ********* Writing ************* #
    
    if (cont % freqsave == 0):

        cph.rename("cp", "cp")
        fileO.write(cph, t)
        clh.rename("cl", "cl")
        fileO.write(clh, t)
        ph.rename("p", "p")
        fileO.write(ph, t)
        uh.rename("u", "u")
        fileO.write(uh, t)
        phih.rename("phi", "phi")
        fileO.write(phih, t)

    cont += 1

    # ********* Updating ************* #
    
    assign(cpold, cph)
    assign(clold, clh)
    assign(uold, uh)
    assign(phiold, phih)
    assign(pold, ph)

# ************* End **************** #
