'''

Coupled large strain poroelasticity and chemotaxis

Holzapfel material law. Fibers are rule-based generated

Replicating initial 2D test now in the ventricle

P1b-P1-P1 discretisation for displ-porosity-pressure
P1-P1 elements for the chemotaxis

Backward Euler scheme for the time discretisation

LOG:

* Diffusion needed to be larger (because of coarse mesh)
* no hope to run the monolithic with the fine mesh newVentr_fine
* maximum principle not respected (no hope to get it with Lagrangian elements for cp,cl)

* other things to play with: anisotropic kappa, ...

NB: This script is a copy of the original one, used mainly to test the splitting schemes.
'''
from fenics import *
# from AndersonAcceleration import AndersonAcceleration
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from itertools import chain
from numba import jit

import logging
logging.getLogger('FFC').setLevel(logging.FATAL)

opts = PETSc.Options()
#opts["ksp_monitor"] = None
monitor = True  # We use our homemade convergence monitor!
print_every = 1
opts["ksp_type"] = "fgmres"
opts["ksp_atol"] = 1e-10
opts["ksp_rtol"] = 1e-6
opts["ksp_norm_type"] = "unpreconditioned"
opts["ksp_pc_side"] = "right"
opts["ksp_gmres_restart"] = 1000
opts["ksp_gmres_modifiedgramschmidt"] = None

# First level options
opts["pc_fieldsplit_schur_fact_type"] = "schur"
opts["fieldsplit_upphi_ksp_type"] = "preonly"
opts["fieldsplit_upphi_ksp_gmres_restart"] = 100
opts["fieldsplit_upphi_ksp_gmres_modifiedgramschmidt"] = None
opts["fieldsplit_upphi_pc_type"] = "fieldsplit"
# opts["fieldsplit_upphi_pc_factor_mat_type"] = "mumps"
opts["fieldsplit_upphi_ksp_atol"] = 1e-14
opts["fieldsplit_upphi_ksp_rtol"] = 1e-1
opts["fieldsplit_upphi_ksp_max_it"] = 10
opts["fieldsplit_upphi_ksp_norm_type"] = "unpreconditioned"
opts["fieldsplit_upphi_ksp_pc_side"] = "right"
# opts["fieldsplit_upphi_ksp_monitor"] = None
opts["fieldsplit_cpcl_ksp_type"] = "preonly"  # HYPRE here is good enough
# opts["fieldsplit_cpcl_ksp_monitor"] = None 
opts["fieldsplit_cpcl_ksp_atol"] = 1e-14
opts["fieldsplit_cpcl_ksp_rtol"] = 0.1
opts["fieldsplit_cpcl_norm_type"] = "unpreconditioned"
opts["fieldsplit_cpcl_pc_side"] = "right"
opts["fieldsplit_cpcl_pc_type"] = "hypre"  # HYPRE here is almost as good as LU

# Second level options
opts["fieldsplit_upphi_pc_fieldsplit_schur_fact_type"] = "full"
opts["fieldsplit_upphi_pc_fieldsplit_schur_precondition"] = "selfp"
opts["fieldsplit_upphi_fieldsplit_u_ksp_type"] = "gmres"
opts["fieldsplit_upphi_fieldsplit_u_ksp_atol"] = 1e-12
opts["fieldsplit_upphi_fieldsplit_u_ksp_rtol"] = 1e-1
opts["fieldsplit_upphi_fieldsplit_u_ksp_max_it"] = 100
# opts["fieldsplit_upphi_fieldsplit_u_ksp_monitor"] = None
opts["fieldsplit_upphi_fieldsplit_u_ksp_gmres_restart"] = 100
opts["fieldsplit_upphi_fieldsplit_u_ksp_gmres_modifiedgramschmidt"] = None
opts["fieldsplit_upphi_fieldsplit_u_pc_type"] = "asm"
opts["fieldsplit_upphi_fieldsplit_u_pc_side"] = "right"
opts["fieldsplit_upphi_fieldsplit_u_pc_asm_local_type"] = "additive"
opts["fieldsplit_upphi_fieldsplit_u_sub_ksp_type"] = "preonly"
opts["fieldsplit_upphi_fieldsplit_u_sub_pc_type"] = "lu"
opts["fieldsplit_upphi_fieldsplit_u_sub_pc_factor_mat_solver_type"] = "umfpack"

opts["fieldsplit_upphi_fieldsplit_pphi_ksp_type"] = "gmres"
opts["fieldsplit_upphi_fieldsplit_pphi_ksp_atol"] = 1e-12
opts["fieldsplit_upphi_fieldsplit_pphi_ksp_rtol"] = 1e-1
opts["fieldsplit_upphi_fieldsplit_pphi_ksp_max_it"] = 100
opts["fieldsplit_upphi_fieldsplit_pphi_ksp_gmres_restart"] = 100
opts["fieldsplit_upphi_fieldsplit_pphi_ksp_gmres_modifiedgramschmidt"] = None
# opts["fieldsplit_upphi_fieldsplit_pphi_ksp_monitor"] = None
opts["fieldsplit_upphi_fieldsplit_pphi_pc_type"] = "asm"
opts["fieldsplit_upphi_fieldsplit_pphi_pc_side"] = "right"
opts["fieldsplit_upphi_fieldsplit_pphi_pc_asm_local_type"] = "additive"
opts["fieldsplit_upphi_fieldsplit_pphi_sub_ksp_type"] = "preonly"
opts["fieldsplit_upphi_fieldsplit_pphi_sub_pc_type"] = "lu"
opts["fieldsplit_upphi_fieldsplit_pphi_sub_pc_factor_mat_solver_type"] = "umfpack"


# All commented as HYPRE works fine for cpcl together
# opts["fieldsplit_cpcl_pc_fieldsplit_cp_ksp_type"] = "preonly"
# opts["fieldsplit_cpcl_pc_fieldsplit_cp_ksp_atol"] = 1e-12
# opts["fieldsplit_cpcl_pc_fieldsplit_cp_ksp_rtol"] = 0.5
# # opts["fieldsplit_cpcl_pc_fieldsplit_cp_ksp_constant_null_space"] = None
# # opts["fieldsplit_cpcl_fieldsplit_cp_ksp_monitor"] = None
# opts["fieldsplit_cpcl_fieldsplit_cp_pc_type"] = "hypre"
# opts["fieldsplit_cpcl_fieldsplit_cl_ksp_type"] = "preonly"
# opts["fieldsplit_cpcl_fieldsplit_cl_ksp_atol"] = 1e-12
# opts["fieldsplit_cpcl_fieldsplit_cl_ksp_rtol"] = 0.5
# # opts["fieldsplit_cpcl_pc_fieldsplit_cl_ksp_constant_null_space"] = None
# # opts["fieldsplit_cpcl_pc_fieldsplit_cl_ksp_monitor"] = None
# opts["fieldsplit_cpcl_fieldsplit_cl_pc_type"] = "hypre"

u_deg = 1
atol = 1e-6
rtol = 1e-6
maxit = 50


#parameters["form_compiler"]["representation"] = "uflacs"
#parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# parameter-dependent functions


def varAngle(s): return ((thetaEpi - thetaEndo) * s + thetaEndo) / 180.0 * pi


def normalizeAndProj_vec(u): return project(u/sqrt(dot(u, u)+0.001), FiberSpace, solver_type='cg')


def normalize_vec(u): return u/sqrt(dot(u, u))


def subplus(u): return conditional(ge(u, 0.0), u, 0.0)


def kappa(J, phi): return kappa0*(1 + (1-phi0)**2/phi0 **
                                  3*J*phi**3*(J-phi)**2)  # isotropic Kozeny-Carman


def react_p(phi, cp, cl): return phi*cp*(gammap - lmbdrp*cl)


def react_l(phi, cp, cl): return phi*cp*cl * lmbdpr


def ell(p, cp): return Sv*Lp0*(1.0+Lbp*cp) * (pc - p - sigma0*(pic-pii)/(1.+Lbp*cp)) \
    - l0*(1.+vmax*(p-p0)**nHill/(km**nHill+(p-p0)**nHill))


# time constants
t = 0.0
dt = 0.05
Tfinal = 0.05
freqsave = 1
cont = 0


# ********* model constants ******* #

# Poroelasticity
E = Constant(60)
nu = Constant(0.35)
lmbdas = Constant(E*nu/((1. + nu)*(1. - 2.*nu)))
lmbdas = Constant(27.293)  # kPa

rhos = Constant(2)  # gr / cm^3
rhof = Constant(1)  # gr / cm^3
kappa0 = Constant(2.5e-7)
phi0 = Constant(0.2)
alpha = Constant(0.3)
muf = Constant(0.03)  # 0.03 cm^2/s, blood viscosity
bb = Constant((0, 0, 0))

Dp = Constant(5.e-3/phi0)
Dl = Constant(1.e-2/phi0)
Lp0 = Constant(3.6e-8)
sigma0 = Constant(0.91)
vmax = Constant(20.)
p0 = Constant(0.133 * 10.9)  # 10.9 mmHg
pic = Constant(0.133 * 20.)  # 20 mmHg
pii = Constant(10.)
pc = Constant(0.133 * 20.)  # 20 mmHg
km = Constant(0.133 * 6.5)  # 6.5 mmHg
chi = Constant(0.05)
nHill = Constant(5.)
I = Identity(3)
Lbp = Constant(5000.)
Sv = Constant(174.)
l0 = Constant(6.82e-5)
gammap = Constant(0.13/phi0)
lmbdrp = Constant(1.8/phi0)
lmbdpr = Constant(7.1/phi0)

# Holzapfel-Ogden. All 'a' quantities are scaled from N/cm^2 as shown in the first one.
a = Constant(0.0496)  # 0.496 N/cm^2
b = Constant(0.041)
a_f = Constant(0.0193)
b_f = Constant(0.176)
a_s = Constant(0.123)
b_s = Constant(0.209)
a_fs = Constant(0.0162)
b_fs = Constant(0.0166)


# mesh = Mesh("meshes/newVentr.xml")
# bdry = MeshFunction("size_t", mesh, "meshes/newVentr_facet_region.xml")
# mesh = Mesh("meshes/newVentr_fine.xml")
# bdry = MeshFunction("size_t", mesh, "meshes/newVentr_fine_facet_region.xml")

# Like this for parallel execution
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "meshes/newVentr.h5", "r")
# hdf = HDF5File(mesh.mpi_comm(), "meshes/newVentr_fine.h5", "r")
hdf.read(mesh, "/mesh", False)
bdry = MeshFunction("size_t", mesh, 2)
hdf.read(bdry, "/facet_region")
base = 8
epi = 7
endo = 6
ds = Measure("ds", domain=mesh, subdomain_data=bdry)  # epi:7, endo:6, base: 8
nn = FacetNormal(mesh)


fileO = XDMFFile(mesh.mpi_comm(), "outputs/CoupledEdemaVentricle-NP.xdmf")
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ********* Finite dimensional spaces ********* #

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
Bub = FiniteElement("Bubble", mesh.ufl_cell(), 4)
P1b = VectorElement(P1 + Bub)
Pvec = VectorElement("CG", mesh.ufl_cell(), u_deg)
Hh = FunctionSpace(mesh, MixedElement([P1, P1, P1b, P1, P1]))

if MPI.COMM_WORLD.rank == 0:
    print("====== DOFS:", Hh.dim())


Sol = Function(Hh)
cp, cl, u, phi, p = split(Sol)
wp, wl, v, psi, q = TestFunctions(Hh)
dSol = Function(Hh)

Mh = FunctionSpace(mesh, P1)
FiberSpace = VectorFunctionSpace(mesh, "CG", 1)
Vh = FunctionSpace(mesh, P1b)
etah = Function(Mh)

# ******* Generate fibre and sheet directions ********** #

phif = TrialFunction(Mh)
psif = TestFunction(Mh)
aDif = Constant(10.)
AAf = aDif*dot(grad(phif), grad(psif))*dx
ggf = Constant(0.0)
BBf = ggf * psif * dx
bcphiEn = DirichletBC(Mh, Constant(0.0), bdry, endo)
bcphiEp = DirichletBC(Mh, Constant(1.0), bdry, epi)
bcphi = [bcphiEp, bcphiEn]

solve(AAf == BBf, etah, bcphi, solver_parameters={'linear_solver': 'gmres'})

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


# ******* Boundary and initial conditions ********** #

# tt0 = Constant(1.)
# ttendo= Expression('tt0*sin(pi/40*t)*sin(pi/40*t)', tt0 = tt0, t=t, degree = 1)
pendo = Expression('0.5*p0*sin(pi/40*t)*sin(pi/40*t)', p0=p0, t=t, degree=0)

u_D = project(Constant((0, 0, 0)), Hh.sub(2).collapse(), solver_type='gmres')
bcU = DirichletBC(Hh.sub(2), u_D, bdry, base)
bcP1 = DirichletBC(Hh.sub(4), pendo, bdry, endo)
bcP2 = DirichletBC(Hh.sub(4), p0, bdry, epi)
bcH = [bcU, bcP1, bcP2]

cp0 = Expression(
    '(x[0]-0.13047)*(x[0]-0.13047)+(x[1]-3.05269)*(x[1]-3.05269)<= 0.24 ? 0.002:0.0', degree=1)

cpold = interpolate(cp0, Hh.sub(0).collapse())
clold = interpolate(Constant(0.003), Hh.sub(1).collapse())
# uold = project(u_D, Hh.sub(2).collapse())
uold = u_D.copy(True)
phiold = interpolate(phi0, Hh.sub(3).collapse())
pold = interpolate(p0, Hh.sub(4).collapse())
# ********  Weak form ********** #

ndim = u.geometric_dimension()

F = I + grad(u)
J = det(F)
invF = inv(F)

# C = F.T*F
C = J**(-2./3.) * F.T*F
B = F*F.T
I1 = tr(C)
I8_fs = inner(f0, C*s0)
I4_f = inner(f0, C*f0)
I4_s = inner(s0, C*s0)

sig = a*exp(b*(I1-ndim))*B + 2*a_f*(I4_f-1)*exp(b_f*(I4_f-1)**2)*outer(F*f0, F*f0) + 2*a_s*(I4_s-1) * exp(b_s *
                                                                                                          (I4_s-1)**2)*outer(F*s0, F*s0) + a_fs*I8_fs * exp(b_fs*(I8_fs)**2)*(outer(F*f0, F*s0) + outer(F*s0, F*f0))


# Peff = J*sig*invF.T + lmbdas*ln(J)*invF.T
# Volumetric term lmbdas/2(J-1)ln(J), gives -p=lmbdas/2(ln(J) + (J-1)/J)
# Peff = J*sig*invF.T + 0.5*lmbdas*(ln(J)+1-1/J)*J*invF.T
Peff = J*sig*invF.T + lmbdas*(J*(ln(J)+1)-1)*invF.T


dtc = Constant(dt)
idt = Constant(1/dt)
# Poromechanics
Fu = inner(Peff - alpha*p*J*invF.T, grad(v)) * dx - rhos*dot(bb, v) * dx
# Fp = rhof*idt * (phi-phiold)*q * dx \
#     + dot(phi*J*invF*kappa(J, phi)/muf*invF.T*grad(p), grad(q)) * dx \
#     - rhof*J*ell(p, cp)*q*dx
# Fphi = psi*(J-1-phi+phi0)*dx
Fp = rhof * idt * J * (phi - phiold) * psi * dx \
    + dot(phi*J*invF*kappa(J, phi)/muf*invF.T*grad(p), grad(psi)) * dx \
    - rhof*J*ell(p, cp)*psi*dx
Fphi = q*(J-1-phi+phi0)*dx
# - dot(ttendo*J*invF.T*nn,v) * ds(endo)

# Chemotaxis
Fcp = J*idt*((phi-phiold)*cp+phi*(cp-cpold))*wp*dx \
    + dot(phi*J*invF*Dp*invF.T*grad(cp), grad(wp)) * dx \
    - J*react_p(phi, cp, cl)*wp*dx \


Fcl = J*idt*((phi-phiold)*cl+phi*(cl-clold))*wl*dx \
    + dot(phi*J*invF*Dl*invF.T*grad(cl), grad(wl)) * dx \
    - dot(phi*chi*cl*J*invF*invF.T*grad(cp), grad(wl)) * dx \
    - J*react_l(phi, cp, cl)*wl*dx

# Auxiliary forms
FF = Fu + Fp + Fphi + Fcp + Fcl
Tang = derivative(FF, Sol, TrialFunction(Hh))

# ********* Time loop ************* #

# Initialize solutions to avoid singularities
# assign(Sol.sub(0), cpold)
# assign(Sol.sub(1), clold)
# assign(Sol.sub(2), uold)
# assign(Sol.sub(3), phiold)
# assign(Sol.sub(4), pold)

res = PETScVector()
jac = PETScMatrix()
assemble(FF, tensor=res)
assemble(Tang, tensor=jac)

# Create solver
ksp = PETSc.KSP().create()
ksp.setOperators(jac.mat(), jac.mat())
PC = ksp.getPC()
PC.setType('fieldsplit')
cp_dofs = Hh.sub(0).dofmap().dofs()
cl_dofs = Hh.sub(1).dofmap().dofs()
cpcl_dofs = sorted(cp_dofs + cl_dofs)
u_dofs = Hh.sub(2).dofmap().dofs()
phi_dofs = Hh.sub(3).dofmap().dofs()
p_dofs = Hh.sub(4).dofmap().dofs()

is_cp = PETSc.IS().createGeneral(cp_dofs)
is_cl = PETSc.IS().createGeneral(cl_dofs)
is_p = PETSc.IS().createGeneral(p_dofs)
is_phi = PETSc.IS().createGeneral(phi_dofs)
is_u = PETSc.IS().createGeneral(u_dofs)

upphi_dofs = sorted(u_dofs + phi_dofs + p_dofs)
PC.setFieldSplitIS(("upphi",  PETSc.IS().createGeneral(upphi_dofs)),
                   ("cpcl", PETSc.IS().createGeneral(cpcl_dofs)))


@jit(nopython=True, cache=True)
def get_local_dofs(dofs_local, dofs_global):
    # Get dofs in subblocks
    # Find the corresponding local indexex in f-p subspace
    n_loc = len(dofs_local)
    n_glob = len(dofs_global)
    dofs = [0] * n_loc
    i_loc = i = 0
    for dof in dofs_global:
        if dof in dofs_local:
            dofs[i_loc] = i
            i_loc += 1
        i += 1
    return dofs


# sub-dof ordering is inconsistent unless seen from the entire dofs vector... so
# we gather all everywhere to see the order locally
dofs_upphi_global = upphi_dofs.copy()
dofs_upphi_global = MPI.COMM_WORLD.allgather(dofs_upphi_global)
dofs_upphi_global = np.array(tuple(chain(*dofs_upphi_global)))  # Concat and sort
dofs_cpcl_global = cpcl_dofs.copy()
dofs_cpcl_global = MPI.COMM_WORLD.allgather(dofs_cpcl_global)
dofs_cpcl_global = np.array(tuple(chain(*dofs_cpcl_global)))  # Concat and sort

u_dofs_local = get_local_dofs(u_dofs, dofs_upphi_global)
pphi_dofs_local = get_local_dofs(p_dofs+phi_dofs, dofs_upphi_global)
cp_dofs_local = get_local_dofs(cp_dofs, dofs_cpcl_global)
cl_dofs_local = get_local_dofs(cl_dofs, dofs_cpcl_global)

ksp_upphi, ksp_cpcl = PC.getFieldSplitSubKSP()

PC_upphi = ksp_upphi.getPC()
PC_upphi.setType('fieldsplit')
PC_upphi.setFieldSplitIS(("u", PETSc.IS().createGeneral(u_dofs_local)),
                         ("pphi", PETSc.IS().createGeneral(pphi_dofs_local)))
PC_upphi.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

PC_cpcl = ksp_cpcl.getPC()
PC_cpcl.setType('fieldsplit')
PC_cpcl.setFieldSplitIS(("cp", PETSc.IS().createGeneral(cp_dofs_local)),
                        ("cl", PETSc.IS().createGeneral(cl_dofs_local)))
PC_cpcl.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
ksp.setFromOptions()
PC.setFromOptions()
PC.setUp()

# This is complicated because fieldsplit requires 'setFromOptions' before 'setUp'.
if PC.getType() == "fieldsplit":
    for ksp_loc in PC.getFieldSplitSubKSP():  # upphi, cpcpl
        ksp_loc.setFromOptions()
        PC_loc = ksp_loc.getPC()
        PC_loc.setFromOptions()
        PC_loc.setUp()
        if PC_loc.getType == "fieldsplit":
            for ksp_loc_loc in PC_loc.getFieldSplitSubKSP():  # u, pphi; cp, cl
                ksp_loc_loc.setFromOptions()
                PC_loc_loc = ksp_loc_loc.getPC()
                PC_loc_loc.setFromOptions()
                PC_loc_loc.setUp()

# Initialize problem ?
# solve(FF == 0, Sol, bcH, J=Tang)
while (t < Tfinal):

    t += dt

    pendo.t = t  # ttendo.t = t

    print("t=%.3f" % t)

    # Iteration preparation
    # solve(FF == 0, Sol, bcH, J=Tang)
    assemble(-FF, tensor=res)
    assemble(Tang, tensor=jac)
    BCS0 = [DirichletBC(bc) for bc in bcH]
    # for bc in bcH:
    # bc.apply(Sol.vector())
    for bc in BCS0:
        bc.homogenize()
        bc.apply(jac, res)
    res0 = res.norm('l2')

    def converged(_ksp, _it, _rnorm, *args, **kwargs):
        """
        args must have: index_map, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p.
        dummy is used to avoid allocation of new vector for residual. [is is somewhere in PETSc...?]
        """

        _res = _ksp.buildResidual()
        # Get residual subcomponents
        _vecs = [_res.getSubVector(_is) for _is in iss]
        a_errs = [vec.norm(PETSc.NormType.NORM_INFINITY) for vec in _vecs]  # Norm 2: 1, Norm infty: 3
        r_errs = [a_err/res0 for a_err, res0 in zip(a_errs, kwargs['res0s'])]

        error_abs = max(a_errs)
        error_rel = max(r_errs)
        if MPI.COMM_WORLD.rank == 0:
            width = 9
            if _it == 0 and monitor:
                print("\tKSP errors : {} {}".format('abs'.rjust(width), 'rel'.rjust(width)), flush=True)

            if monitor and _it % print_every == 0:
                print("\tKSP it {:4}:   {:.3e}, {:.3e}".format(
                    _it, error_abs, error_rel), flush=True)
            # print("DEBUG", a_errs)
            # print("DEBUG", r_errs)

        if error_abs < _ksp.atol or error_rel < _ksp.rtol:
            # Convergence
            # parprint("---- [Solver] Converged")
            return 1
        elif _it > _ksp.max_it or error_abs > _ksp.divtol:
            # Divergence
            return -1
        else:
            # Continue
            return 0

    err_abs = res.norm('l2')
    err_rel = err_abs/res0
    ksp.setConvergenceTest(converged)
    it = 0
    if MPI.COMM_WORLD.rank == 0:
        print('\tit {:3}, err abs = {:.3e}  err rel = {:.3e}'.format(
            it, err_abs, err_rel), flush=True)
    k_its = 0
    while err_abs > atol and err_rel > rtol and it < maxit:

        # First solve mechanics
        # ksp.setOperators(jac.mat(), jac.mat())
        ksp.setUp()
        iss = [is_cp, is_cl, is_u, is_phi, is_p]
        vecs = [res.vec().getSubVector(_is) for _is in iss]
        res0s = [vec.norm() for vec in vecs]
        for i, res0_ in enumerate(res0s):
            if res0_ < 1e-14:
                res0s[i] = 1
        kwargs_ = {'res0s': res0s}
        ksp.setConvergenceTest(converged, kargs=kwargs_)
        ksp.solve(res.vec(), dSol.vector().vec())
        #dSol.vector().apply("")
        # solve(jac, dSol.vector(), res)
        Sol.vector().vec().axpy(1, dSol.vector().vec())
        Sol.vector().apply("")

        # Compute error
        assemble(-FF, tensor=res)
        assemble(Tang, tensor=jac)
        for bc in BCS0:
            bc.apply(jac, res)
        err_abs = res.norm('l2')
        err_rel = err_abs/res0
        it += 1
        k_its_local = ksp.getIterationNumber()
        k_its += k_its_local

        if MPI.COMM_WORLD.rank == 0:
            print('\tit {:3} in {:3} GMRES its, err abs = {:.3e}  err rel = {:.3e}'.format(
                it, k_its_local, err_abs, err_rel), flush=True)
        if min(err_abs, err_rel) > 1e14 or np.isnan(err_abs) or np.isnan(err_rel):
            if MPI.COMM_WORLD.rank == 0:
                print("\t Newton diverged")
                import sys
                sys.exit()
    if MPI.COMM_WORLD.rank == 0:
        print("Solved time {:.3f}s in {} nonlinear its, {} total krylov its and {:.2f} average krylov its".format(
            t, it, k_its, k_its/it))
    cph, clh, uh, phih, ph = Sol.split()

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

    assign(cpold, cph)
    assign(clold, clh)
    assign(uold, uh)
    assign(phiold, phih)
    assign(pold, ph)

# ************* End **************** #
