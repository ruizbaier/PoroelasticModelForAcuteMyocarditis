# -----------------------------------------------------------------------------
# Code: PoroelasticModelForAcuteMyocarditis
#
# @rticle{Barnafi21,
#   url     = {http://arxiv.org/abs/2111.04206},
#   doi     = {2111.04206xxx},
#   author  = {Barnafi, Nicolas and G\'omez-Vargas, Bryan  and Louren\c{c}o, Wesley de Jesus and 
#               Reis, Ruy Freitas and Rocha, Bernardo Martins and Lobosco, Marcelo and 
#               Ruiz-Baier, Ricardo and Weber dos Santos, Rodrigo},
#   title   = {Mixed methods for large-strain poroelasticity/chemotaxis models 
#               simulating the formation of myocardial oedema},
#   year    = {2021},
#   journal = {arXiv preprint}
# }

# @article{Lourenco22,
#   url     = {http://xxxx},
#   doi     = {101222.xxx},
#   year    = {2022},
#   volume  = {xxx}, 
#   pages   = {1--20},
#   author  = {Louren\c{c}o, Wesley de Jesus and Reis, Ruy Freitas and 
#               Ruiz-Baier, Ricardo and Rocha, Bernardo Martins and 
#               Weber dos Santos, Rodrigo and Lobosco, Marcelo},
#   title   = {A poroelastic approach for modelling myocardial oedema 
#               in acute myocarditis},
#   journal = {Frontiers in Physiology}
# }
# 
# -----------------------------------------------------------------------------

from __future__ import print_function
from numpy import *
from fenics import *
import time

# -----------------------------------------------------------------------------

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# -----------------------------------------------------------------------------
# Reaction terms and functions 
# -----------------------------------------------------------------------------

# Hydraulic conductivity
kappa   = lambda J,phi: kappa0

# Reaction term pathogens (Eq. 2.5)
react_p = lambda phi,cp,cl: phi*(gammap - lmbdarp*cl)*cp

# Reaction term leucocytes (Eq. 2.4)
react_l = lambda phi,cp,cl: phi*cp * cl * lmbdapr

# Coulping term for capillary exchange (Eq. 2.7)
ell     = lambda p, cp: Lp0*(1.0+Lbp*cp)*Sv * (pc - p - sigma0*(pic-pii)/(1.+Lbp*cp)) \
        - l0*(1.+vmax*(p-p0)**nHill/(km**nHill+(p-p0)**nHill))

# -----------------------------------------------------------------------------        
# Time constants
# -----------------------------------------------------------------------------

t  = 0.0   # time
dt = 0.1   # time step size
tf = 200.0 # final time
cont = 0   # counter
freq = 10  # rate for output data

# -----------------------------------------------------------------------------
# Model constants
# -----------------------------------------------------------------------------

# Solid constitutive model: Neo-Hookean
E       = Constant(60.) 
nu      = Constant(0.35)
lmbdas  = Constant(E*nu/((1. + nu)*(1. - 2.*nu)))
mus     = Constant(E/(2.*(1. + nu)))
pen     = Constant(lmbdas)
 
# Darcy / Poroelastic
rhos    = Constant(2.e-3)
rhof    = Constant(1.e-3)
kappa0  = Constant(2.5e-7)
phi0    = Constant(0.2)
alpha   = Constant(0.25) 
muf     = Constant(1.) 
bb      = Constant((0.,0.,0.))

# Immune system

Dp     = Constant(5.e-4)/phi0 # Local edema case
Dl     = Constant(5.e-1)/phi0 # Local edema case
Lp0     = Constant(3.6e-8)
sigma0  = Constant(0.91)
vmax    = Constant(200.) 
p0      = Constant(0.0)
pic     = Constant(20.)
pii     = Constant(10.)
pc      = Constant(20.)
km      = Constant(6.5)
chi     = Constant(1.e-2)/phi0 
nHill   = Constant(1.)
Lbp     = Constant(1.e4)
Sv      = Constant(174.) 
l0      = Constant( Sv * Lp0 * (pc - sigma0*(pic-pii)) ) #Constant(6.82776e-5)
gammap  = Constant(0.12)/phi0  # Local edema case
lmbdapr = Constant(9.00)/phi0  # Local edema case
lmbdarp = Constant(1.5)/phi0

# -----------------------------------------------------------------------------
# Local or diffuse dynamics (overwrites baseline parameters)
# -----------------------------------------------------------------------------

# Local edema parameters (Section 3.2, Fig. 3.2)
#Dp     = Constant(5.e-4)/phi0 # Local edema case
#Dl     = Constant(5.e-1)/phi0 # Local edema case
#gammap  = Constant(0.12)/phi0  # Local edema case
#lmbdapr = Constant(9.00)/phi0  # Local edema case

# Global edema parameters (Section 3.2, Fig. 3.3)
Dp     = Constant(1.e-3)/phi0
Dl     = Constant(3.e-2)/phi0
gammap = Constant(0.06)/phi0
lmbdapr = Constant(7.10)/phi0

# -----------------------------------------------------------------------------
# Mesh and output file
# -----------------------------------------------------------------------------

mesh = Mesh()
hdf  = HDF5File(mesh.mpi_comm(), "file.h5", "r")
hdf.read(mesh, "/mesh", False)
bdry = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
hdf.read(bdry, "/bdry")

nn = FacetNormal(mesh)

fileO = XDMFFile(mesh.mpi_comm(), "outputs/CoupledEdemaRectangle.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# -----------------------------------------------------------------------------
# Finite dimensional spaces
# -----------------------------------------------------------------------------

P1    = FiniteElement("CG", mesh.ufl_cell(), 1)
P2    = FiniteElement("CG", mesh.ufl_cell(), 2)
Bub   = FiniteElement("Bubble", mesh.ufl_cell(), 4)
P1b   = VectorElement(P1 + Bub)
P2vec = VectorElement("CG", mesh.ufl_cell(), 2)
Hh    = FunctionSpace(mesh, MixedElement([P2, P2, P1, P2vec, P1]))

print("Poroelastic Model For Acute Myocarditis")
print(" Total DoFs =", Hh.dim())
    
Sol  = Function(Hh)
dSol = TrialFunction(Hh)

cp,cl,p,u,phi = split(Sol)
wp,wl,q,v,psi = TestFunctions(Hh)

# -----------------------------------------------------------------------------
# Boundary and initial conditions
# -----------------------------------------------------------------------------

# markers
base = 8
epi  = 7
endo = 6

u_D  = project(Constant((0, 0, 0)), Hh.sub(3).collapse())
bcU  = DirichletBC(Hh.sub(3), u_D, bdry, base)
bcP2 = DirichletBC(Hh.sub(2), p0, bdry, epi)
bcH  = [bcU]

# Pathogen initial condition (3D case - LV)
cp0 = Expression('(x[0]-0.13047)*(x[0]-0.13047) + '
                 '(x[1]-3.05269)*(x[1]-3.05269) + '
                 '(x[2]-5.5)*(x[2]-5.5) <= 0.24 ? 0.002:0.0', degree=1)

cpold  = interpolate(cp0, Hh.sub(0).collapse())
clold  = interpolate(Constant(0.003), Hh.sub(1).collapse())
pold   = interpolate(p0, Hh.sub(2).collapse())
uold   = project(Constant((0.,0.,0.)), Hh.sub(3).collapse())
phiold = project(phi0, Hh.sub(4).collapse())

# -----------------------------------------------------------------------------
# Kinematics and Constitutive relations
# -----------------------------------------------------------------------------

ndim = u.geometric_dimension()
I = Identity(mesh.topology().dim())
F = I + grad(u)
J = det(F)
invF = inv(F)
C = F.T*F
B = F*F.T
I1 = tr(C)

# Neo-Hookean
Peff =  mus*(F - invF.T) + lmbdas*ln(J)*invF.T 

# -----------------------------------------------------------------------------
# Weak forms
# -----------------------------------------------------------------------------

# Poroelastic (Eq. 2.3)

F1 = inner(Peff - alpha*p*J*invF.T, grad(v)) * dx \
    + psi*(J-1.0-phi+phi0)*dx \
    + dot(phi*J*invF*kappa(J,phi)/muf*invF.T*grad(p),grad(q)) * dx \
    - rhos*dot(bb,v)* dx \
    + rhof*J/dt*(phi-phiold)*q * dx \
    - rhof*J*ell(p, cp)*q*dx

# Immuno (Eqs. 2.6, Reaction terms 2.4, 2.5 and 2.7)

F2 = J/dt*((phi-phiold)*cp+phi*(cp-cpold))*wp*dx \
    + dot(phi*J*invF*Dp*invF.T*grad(cp),grad(wp)) * dx \
    + J/dt*((phi-phiold)*cl+phi*(cl-clold))*wl*dx \
    + dot(phi*J*invF*Dl*invF.T*grad(cl),grad(wl)) * dx \
    - dot(chi*phi*cl*J*invF*invF.T*grad(cp),grad(wl)) * dx \
    - J*react_p(phi,cp,cl)*wp*dx \
    - J*react_l(phi,cp,cl)*wl*dx 

FF = F1 + F2 

Tang = derivative(FF,Sol,dSol)   

# -----------------------------------------------------------------------------
# Time loop  
# -----------------------------------------------------------------------------
 
wbegin = time.time() 
while (t < tf):
    t += dt
    print("Time = %.2f" % t)
        
    solve(FF == 0., Sol, bcs=bcH, J=Tang, \
          solver_parameters={'newton_solver':{'linear_solver':'mumps',\
                                              'absolute_tolerance':1.0e-10,\
                                              'relative_tolerance':1.0e-10}})
    cph,clh,ph,uh,phih = Sol.split()

    if (cont % freq == 0):
        cph.rename("Pathogen","Pathogen")
        fileO.write(cph,t)
        clh.rename("Leukocyte","Leukocyte")
        fileO.write(clh,t)
        ph.rename("Pressure","Pressure")
        fileO.write(ph,t)  
        uh.rename("Displacement","Displacement")
        fileO.write(uh,t)
        phih.rename("phase","Phase")
        fileO.write(phih,t)      

    cont += 1

    assign(pold,ph)
    assign(uold,uh)
    assign(cpold,cph)
    assign(clold,clh)
    assign(phiold,phih)

# -----------------------------------------------------------------------------
# Final settings
# -----------------------------------------------------------------------------

wstop = time.time()
wtime = (wstop-wbegin)/60.0
print("\nTotal simulation time = ", wtime, "seconds")

# End