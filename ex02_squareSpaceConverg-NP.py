'''
Convergence-in-space test 
Coupled large strain poroelasticity and chemotaxis
NeoHookean material
 
P1b-P1-P1 discretisation for displ-porosity-pressure
P1-P1 elements for the chemotaxis

Backward Euler scheme for the time discretisation
Unit square, manufactured solutions
Mixed boundary conditions
'''


from fenics import *
import sympy2fenics as sf

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 3

strain = lambda v: sym(grad(v))
kappa   = lambda J,phi: kappa0*(1+ (1-phi0)**2/phi0**3*J*phi**3*(J-phi)**2) # Kozeny-Carman
react_p = lambda phi,cp,cl: phi*cp*(gammap - lmbdrp*cl)
react_l = lambda phi,cp,cl: phi*cp*cl * lmbdpr
ell = lambda p, cp: Sv*Lp0*(1.0+Lbp*cp) * (pc - p - sigma0*(pic-pii)/(1.+Lbp*cp)) \
      - l0*(1.+vmax*(p-p0)**nHill/(km**nHill+(p-p0)**nHill))

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# time constants
dt = 0.01; Tfinal = 3.*dt;

# ********* model constants setting to 1 most of them ******* #

# Poroelasticity
lmbdas = Constant(36.4)
mus    = Constant(2.1)
rhos   = Constant(1.)
rhof   = Constant(1.)
kappa0 = Constant(1.)
alpha = Constant(0.5)
muf = Constant(1.)

# parameters? 
Dp = Constant(0.9)
Dl = Constant(0.8)
Lp0 = Constant(1.)
Lbp = Constant(1.)
sigma0 = Constant(1.)
Lbr = Constant(1.)
vmax = Constant(1.)
gammap = Constant(1.)
gammal = Constant(1.)
lmbdrp = Constant(1.)
lmbdpr = Constant(1.)
clmax = Constant(1.)
mul = Constant(1.)
p0   = Constant(1.)
km   = Constant(1.)
pic  = Constant(1.)
pc   = Constant(1.)
l0   = Constant(1.)
pii  = Constant(1.)
nHill = Constant(2.)
chi = Constant(1.)
Sv = Constant(1.)

I    = Identity(2)
uinf = Constant(0.25)

# ******* Exact solutions for error analysis ****** #
u_str = '(uinf*(sin(pi*x)*cos(pi*y)+x*x*0.5/lmbdas)*t,uinf*(-cos(pi*x)*sin(pi*y)+y*y*0.5/lmbdas)*t)'
dt_u_str = '(uinf*(sin(pi*x)*cos(pi*y)+x*x*0.5/lmbdas),uinf*(-cos(pi*x)*sin(pi*y)+y*y*0.5/lmbdas))'

cp_str = 't*(0.3*exp(x)+0.1*cos(pi*x)*cos(pi*y))'
dt_cp_str = '(0.3*exp(x)+0.1*cos(pi*x)*cos(pi*y))'

cl_str = 't*(0.3*exp(x)+0.1*sin(pi*x)*sin(pi*y))'
dt_cl_str = '0.3*exp(x)+0.1*sin(pi*x)*sin(pi*y)'

p_str = 'sin(pi*x*y)*cos(pi*x*y)*t'
dt_p_str = 'sin(pi*x*y)*cos(pi*x*y)'

phi0_str = '0.6+0.1*sin(x*y)'

# phi depends on J and on phi0...

nkmax = 4

hh = []; nn = []; ephi = []; rphi = []; eu = []; ru = []; ep = []; rp = [];
ecp = []; rcp = []; ecl = []; rcl = [];
ru.append(0.0); rp.append(0.0); rphi.append(0.0); rcp.append(0.0); rcl.append(0.0)

# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1); mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh)

    hh.append(mesh.hmax())

    t = 0

    phi0 = Expression(str2exp(phi0_str), degree=6, domain=mesh)
    
    cp_ex = Expression(str2exp(cp_str), t = t, degree=6, domain=mesh)
    dt_cp_ex = Expression(str2exp(dt_cp_str), t = t, degree=6, domain=mesh)
    cl_ex = Expression(str2exp(cl_str), t = t, degree=6, domain=mesh)
    dt_cl_ex = Expression(str2exp(dt_cl_str), t = t, degree=6, domain=mesh)

    u_ex  = Expression(str2exp(u_str), t = t, uinf= uinf, lmbdas=lmbdas, degree=6, domain=mesh)
    dt_u_ex  = Expression(str2exp(dt_u_str), t = t, uinf= uinf, lmbdas=lmbdas, degree=6, domain=mesh)

    p_ex  = Expression(str2exp(p_str), t = t, degree=6, domain=mesh)
    dt_p_ex = Expression(str2exp(dt_p_str), t = t, degree=6, domain=mesh)
    
    F_ex = I + grad(u_ex)
    J_ex = det(F_ex)
    invF_ex = inv(F_ex)

    #NEO HOOKEAN
    PP_ex = mus*(F_ex - invF_ex.T) \
           + lmbdas*ln(J_ex)*invF_ex.T \
           - alpha*p_ex*J_ex*invF_ex.T 
    
    phi_ex = J_ex - 1 + phi0 
    dt_phi_ex = J_ex * inner(grad(dt_u_ex).T,invF_ex) # Zheng2020 eq. (43)


    bb_ex = - div(PP_ex) / rhos
    pflux_ex = phi_ex*J_ex*invF_ex*kappa(J_ex,phi_ex)/muf*invF_ex.T*grad(p_ex)
    ll_ex = rhof*J_ex*dt_phi_ex - rhof*J_ex*ell(p_ex,cp_ex) \
            - div(pflux_ex) 

    mp_ex = J_ex*(dt_phi_ex*cp_ex+phi_ex*dt_cp_ex) \
           - div(phi_ex*J_ex*invF_ex*Dp*invF_ex.T*grad(cp_ex)) \
           - J_ex* react_p(phi_ex,cp_ex,cl_ex)

    ml_ex = J_ex*(dt_phi_ex*cl_ex+phi_ex*dt_cl_ex) \
           - div(phi_ex*J_ex*invF_ex*Dl*invF_ex.T*grad(cl_ex) \
                 - phi_ex*chi*cl_ex*J_ex*invF_ex*invF_ex.T*grad(cp_ex)) \
           - J_ex * react_l(phi_ex,cp_ex,cl_ex)


    

    # ********* Finite dimensional spaces ********* #

    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    Bub  = FiniteElement("Bubble", mesh.ufl_cell(), 3)
    P1b  = VectorElement(P1 + Bub)
    Hh = FunctionSpace(mesh,MixedElement([P1,P1,P1b,P1,P1]))

    print("**************** Total Dofs = ", Hh.dim())

    nn.append(Hh.dim())
    
    Sol = Function(Hh); dSol = TrialFunction(Hh)
    cp,cl,u,phi,p = split(Sol)
    wp,wl,v,psi,q = TestFunctions(Hh)

    # Essential BCs - product space
    bdry = MeshFunction("size_t", mesh, 1)
    bdry.set_all(0)

    Gamma = CompiledSubDomain(" near(x[0],0.0) && on_boundary")
    Sigma = CompiledSubDomain("(near(x[0],1.0) || near(x[1],1.0) || near(x[1],0.0)) && on_boundary")

    Gamma.mark(bdry,91); Sigma.mark(bdry,92);
    ds = Measure("ds", subdomain_data=bdry)
        
    cpold = interpolate(cp_ex, Hh.sub(0).collapse())
    clold = interpolate(cl_ex, Hh.sub(1).collapse())
    uold   = project(u_ex, Hh.sub(2).collapse())
    phiold = project(phi_ex, Hh.sub(3).collapse())
    pold   = interpolate(p_ex, Hh.sub(4).collapse())

    # ********  Weak form ********** #


    F = I + grad(u); J = det(F); invF = inv(F)

    P_eff = mus*(F - invF.T) + lmbdas*ln(J)*invF.T


    F1 = inner(P_eff - alpha*p*J*invF.T, grad(v)) * dx \
         + psi*(J-1-phi+phi0)*dx \
         + rhof*J/dt*(phi-phiold)*q * dx \
         + dot(phi*J*invF*kappa(J,phi)/muf*invF.T*grad(p),grad(q)) * dx \
         - rhos*dot(bb_ex,v)* dx \
         - dot(PP_ex*n,v) * ds(92) \
         - rhof*J*ell(p,cp)*q*dx \
         - ll_ex * q * dx \
         - dot(pflux_ex,n) * q * ds(91)

    F2 = J/dt*((phi-phiold)*cp+phi*(cp-cpold))*wp*dx \
         + dot(phi*J*invF*Dp*invF.T*grad(cp),grad(wp)) * dx \
         - J*react_p(phi,cp,cl)*wp*dx \
         - mp_ex*wp*dx\
         + J/dt*((phi-phiold)*cl+phi*(cl-clold))*wl*dx \
         + dot(phi*J*invF*Dl*invF.T*grad(cl),grad(wl)) * dx \
         - dot(phi*chi*cl*J*invF*invF.T*grad(cp),grad(wl)) * dx \
         - J*react_l(phi,cp,cl)*wl*dx \
         - ml_ex*wl*dx

    FF = F1 + F2
    Tang = derivative(FF,Sol,dSol)
    
    # ********* Time loop ************* #
        
    while (t < Tfinal):

        t += dt
        
        print("t=%.2f" % t)

        cp_ex.t = t; dt_cp_ex.t = t;
        cl_ex.t = t; dt_cl_ex.t = t;
        u_ex.t = t; p_ex.t = t; dt_p_ex.t = t; 
        dt_u_ex.t = t; 

        bcCp = DirichletBC(Hh.sub(0), cp_ex, 'on_boundary')
        bcCl = DirichletBC(Hh.sub(1), cl_ex, 'on_boundary')
        
        bcP = DirichletBC(Hh.sub(4), p_ex, bdry, 92)
        u_D = project(u_ex, Hh.sub(2).collapse())
        bcU = DirichletBC(Hh.sub(2), u_D, bdry, 91)
        bcH = [bcCp,bcCl,bcU,bcP]
        
        solve(FF == 0, Sol, J=Tang, bcs=bcH, \
              solver_parameters={'newton_solver':{'linear_solver':'mumps',\
                                                  'absolute_tolerance':1.0e-6,\
                                                  'relative_tolerance':1.0e-6}})
        cph,clh,uh,phih,ph = Sol.split()

        assign(cpold,cph); assign(clold,clh); assign(uold,uh)
        assign(phiold,phih); assign(pold,ph)

    ecp.append(errornorm(cp_ex,cph,'H1'))
    ecl.append(errornorm(cl_ex,clh,'H1'))
    eu.append(errornorm(u_ex,uh,'H1'))
    ephi.append(pow(assemble((phi_ex-phih)**2*dx),0.5))
    ep.append(errornorm(p_ex,ph,'H1'))

    if(nk>0):
        rcp.append(ln(ecp[nk]/ecp[nk-1])/ln(hh[nk]/hh[nk-1]))
        rcl.append(ln(ecl[nk]/ecl[nk-1])/ln(hh[nk]/hh[nk-1]))
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rphi.append(ln(ephi[nk]/ephi[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********  Generating error history **** #
print('nn   &  hh  &   e(cp)  &   r(cp) &   e(cl)  &   r(cl)  &   e(u)  &   r(u)  &   e(phi)  &   r(phi)  &  e(p)  &  r(p) ')
print('==========================================================================')

for nk in range(nkmax):
    print('%d & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g \\\ ' % (nn[nk], hh[nk], ecp[nk], rcp[nk], ecl[nk], rcl[nk], eu[nk], ru[nk], ephi[nk], rphi[nk], ep[nk], rp[nk]))

# ************* End **************** #
