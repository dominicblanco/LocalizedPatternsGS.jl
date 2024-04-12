#Computer assisted proof of a spike solution for the 2D Gray Scott reduced equation (i.e. λγ = 1): λ₁Δu - u + u² - λ₁v³ = 0
# The following code computes the solution and rigorously proves the results given in section 6 of
# "Localized stationary patterns in the 2D Gray-Scott model: computer assisted proofs of existence"  M. Cadiot and D. Blanco

# First, the code computes the approximate solution using the initial guess described in section 3 of the aforementioned paper.
# From this we can check if the proof of the solution is verified or not. We essentially prove Theorem 6.1.

#####################################################################################################################################################################

# Choice of the parameters for the proof of spikes when 0 < γ < 2/9:
# λ₂ = 9
# λ₁ = 1/9

# Needed packages
using RadiiPolynomial, IntervalArithmetic, LinearAlgebra, FFTW

# Needed additional sequence structures for RadiiPolynomial (see Remark 2.3)
include("D4Fourier.jl")

#####################################################################################################################################################################


#################################### List of the needed functions : go directly to line 180 for the main code ################################################# 

# Converts a sequence to D₄Fourier
function Convert2D₄(a)
    N = order(a)[1]
    f = frequency(a)[1]
    anew = Sequence(D₄Fourier(N,f), zeros(dimension(D₄Fourier(N,f))))
    for k₂ = 0:N
        for k₁ = k₂:N
            anew[(k₁,k₂)] = a[(k₁,k₂)]
        end
    end
    return anew
end

# Equivalent of meshgrid function from Matlab
function _meshgrid(x,y)
    Nx = length(x)
    Ny = length(y)
    X = zeros(Ny,Nx)
    Y = zeros(Ny,Nx)
    for j = 1:Nx
        for i = 1:Ny
            X[i,j] = x[j]
            Y[i,j] = y[i]
        end
    end
    return X,Y
end

# Computes the result in Lemma 4.1
function φ(A,B,C,D)
    O₁ = max(A,D) + max(B,C)
    O₂ = sqrt(A^2 + D^2 + B^2 + C^2)
    return min(O₁,O₂)
end

# Computes the Fourier coefficients of 1_𝒟₀².
function _char_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f)⊗Fourier(N,f), Interval.(complex.(zeros((2N+1)^2))))
    for n₂ = -N:N
        for n₁ = -N:N
            char[(n₁,n₂)] = interval(1)/(interval(4)*d^2) * exp(1im*n₁*interval(π)*(interval(1)/d * interval(1/2) - interval(1)))*exp(1im*n₂*interval(π)*(interval(1)/d * interval(1/2) - interval(1))) * sinc(n₁/d*interval(1/2))*sinc(n₂/d*interval(1/2))
        end
    end
    rchar = Sequence(D₄Fourier(N,f), interval.(zeros(dimension(D₄Fourier(N,f)))))
    for n₂ = 0:N
        for n₁ = n₂:N
            rchar[(n₁,n₂)] = real(char[(n₁,n₂)] + char[(n₂,-n₁)] + char[(-n₁,-n₂)] + char[(-n₂,n₁)])
        end
    end
    return rchar
end

# Computes the sequence a(d,⋅) for a in D₄Fourier.
function _sequence_on_boundary(a)
    N = order(a)[1]
    f = frequency(a)[1]
    anew = Sequence(CosFourier(N,f), interval.(zeros(N+1)))
    for n₁ = 0:N
        for n₂ = -N:N
            anew[n₁] += a[(max(n₁,abs(n₂)),min(n₁,abs(n₂)))]*(-1)^n₂
        end
    end
    return anew
end

# Computes the Fourier coefficients of 1_𝒟₀
function _char_1D_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f), Interval.(complex.(zeros((2N+1)))))
    for n = -N:N
        char[n] = interval(1)/(interval(2)*d) * exp(1im*n*interval(π)*(interval(1)/d * interval(1/2) - interval(1))) * sinc(n/d*interval(1/2))
    end
    rchar = Sequence(CosFourier(N,f), interval.(zeros(N+1)))
    for n = 0:N
        rchar[n] = real(char[n] + char[-n])
    end
    return rchar
end

# Computes the function needed to take the convolution with ∂ₓ₁V₁ᴺ
# We denote by (Ṽⱼ)ₘ = |m̃₁| (Vⱼᴺ)ₘ 
function _Ṽⱼ_coeffs(Vⱼᴺ)
    N = order(Vⱼᴺ)[1]
    f = frequency(Vⱼᴺ)[1]
    Ṽⱼ = Sequence(CosFourier(N,f)⊗CosFourier(N,f), interval.(zeros((N+1)^2)))
    for n₁ = 0:N
        for n₂ = 0:N
            Ṽⱼ[(n₁,n₂)] = abs(n₁)*f*Vⱼᴺ[(max(n₁,n₂),min(n₁,n₂))]
        end
    end
    return Ṽⱼ
end

# Gray-Scott reduced equation
function Fᵣ!(Fᵣ,U,λ₁)
    project!(Fᵣ,Laplacian(2)*U*λ₁ + U^2 - λ₁*U^3 - U)
    return Fᵣ
end

# Derivative of the Gray-Scott reduced equation
function DFᵣ!(DFᵣ,U,λ₁)
    DFᵣ .= 0
    Δ = project(Laplacian(2),space(U),space(U),Float64)
    𝕌 = project(Multiplication(U),space(U),space(U),Float64)
    𝕌² = project(Multiplication(U^2),space(U),space(U),Float64)
    DFᵣ = Δ*λ₁ + 2𝕌 - 3λ₁*𝕌² - I
    return DFᵣ
end

# Newton function
function _newton_gs(Ū₀,jmax,λ₁)
    GS = similar(Ū₀)
    s = space(Ū₀)
    r = length(Ū₀)
    DGS = LinearOperator(s,s,similar(coefficients(Ū₀),r,r))
    j = 0
    ϵ = 1
    nv = 1
    while (ϵ > 1e-14) & (j < jmax)
        GS = Fᵣ!(GS,Ū₀,λ₁)
        DGS = DFᵣ!(DGS,Ū₀,λ₁)
        Ū₀ = Ū₀ - DGS\GS
        @show ϵ = norm(GS,Inf)
        nu = norm(Ū₀)
        if nu < 1e-5
            @show nu
            display("Newton may have converged to the 0 solution")
            return nv,j
            break
        end
        j += 1
    end
    return Ū₀,ϵ
end

# αₙ for the trace operator (see Section 3.2).
function αₙ(n)
    if n[1] == n[2] == 0
        return 1
    elseif n[1] == n[2] != 0
        return 4
    elseif (n[1] != 0) & (n[2] == 0)
        return 2
    else
        return 4
    end
end

# Computes the trace for a D₄Fourier sequence.
function _trace_D₄(N)
    M = dimension(D₄Fourier(N,1.0))
    S = zeros(N+1,M)
    for n₂ = 0:N
        for n₁ = 0:N
            m = (max(n₁,n₂),min(n₁,n₂))
            α = αₙ(m)
            S[n₁+1,m[1] + m[2]*N - div(((m[2]-2)^2 + 3*(m[2]-2)),2)] = α*(-1)^n₂
        end
    end
    return S
end

# Allows us to switch between D₄ and exponential Fourier series
function _exp2D₄!(D::Vector{Float64},s::D₄Fourier)
    k = 1
    ord = order(s)[1]
    for k₂ = 0:ord
        for k₁ = k₂:ord
            if k₁ == k₂ == 0
                D[k] = 1
                k += 1
            elseif k₁ == k₂ != 0
                D[k] = sqrt(4)
                k += 1
            elseif (k₁ != 0) & (k₂ == 0)
                D[k] = sqrt(4)
                k += 1
            else
                D[k] = sqrt(8)
                k +=1 
            end
        end
    end
    return D
end

# Allows us to switch between D₂ and exponential Fourier series
function exp2cos(N)

    d = 2*((ones((N+1)^2)))

    d[1] = 1;
    for n2=1:N
        d[n2+1] = sqrt(2);
    end

    for n1 = 1:N
        d[n1*(N+1)+1] = sqrt(2);
    end

    return d
end

# Computes convolution of D₄Fourier sequences up to order N
function _conv_small(u,v,N)
    #Computes u*v only up to order N
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(D₄Fourier(N,frequency(u)[1]), interval.(zeros(dimension(D₄Fourier(N,frequency(u)[1])))))
    for i₂ ∈ 0:N
        for i₁ ∈ i₂:N
            Cᵢ = interval(zero(Float64))
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Performs convolution up to order N of a D₄ and D₂ Fourier series
function __conv_small(u,v,N)
    #Computes u*v up to order N
    #u is a sequence in D₄Fourier
    #v is a sequence in CosFourier ⊗ CosFourier (D₂ symmetric)
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(CosFourier(N,frequency(u)[1])⊗CosFourier(N,frequency(u)[1]), interval.(zeros((N+1)^2)))
    for i₁ ∈ 0:N
        for i₂ ∈ 0:N
            Cᵢ = interval(zero(Float64))
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (abs(j₁),abs(j₂))
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Checks the conditions of the Radii-Polynomial Theorem (see Section 4).
function CAP(𝒴₀,𝒵₁,𝒵₂,s₀)
    if 𝒵₁ + 𝒵₂*s₀ < 1
        if interval(1/2)*𝒵₂*s₀^2 - (interval(1)-𝒵₁)*s₀ + 𝒴₀ < 0
          display("The proof was successful for s₀ = ")
          display(sup(s₀))  
        else
          display("The condition 2𝒴₀*𝒵₂ < (1-𝒵₁)² is not satisfied")
        end
    else
        if 𝒵₁ > 1
            display("𝒵₁ is too big")
        else
          display("failure: linear term is positive")
        end
      end
end

################### PROOF OF SPIKE SOLUTION : MAIN CODE #################################################################################################################################################
N = 20              # number of Fourier modes : 0 ≤ n₂ ≤ n₁ ≤ N for D₄ series
d = 4 ; di = interval(d)   # size of the domain = half period of the functions
λ₁ = 1/9 ; λ₁i = interval(λ₁)    # value of the parameter. λ₁ = 1/λ₂
Q = sqrt(1-9*λ₁/2)     # Quantity Q defined in "Exact Homoclinic and Heteroclinic Solutions of the Gray-Scott Model for Autocatalysis" J. K. Hale, L. A. Peletier and W. C. Troy
fourier = D₄Fourier(N,π/di)   # definition of the sequence space : D₄ series of frequency π/d
x = 2*d/(2*N+1)*(-N:N)
y = x
X,Y = _meshgrid(x,y)
Ū₀ = 3 ./(1 .+Q*cosh.(sqrt.((X.^2+Y.^2)/λ₁)))
s₀ = interval(0.0005) # Value of s₀ for 𝒵₂

# Constructing approximate solution via Newton's method
Û₀ = fftshift(FFTW.fft(ifftshift(Ū₀))/(2N+1)^2)
Ū₀_full = Sequence(Fourier(N,π/d)⊗Fourier(N,π/d), real(vec(Û₀)))
Ū₀ = Convert2D₄(Ū₀_full)
U₀,ϵ = _newton_gs(Ū₀,30,λ₁) 
U₀_interval = Sequence(fourier, coefficients(interval.(U₀)))

#################################################   Projection on X²₀(ℝ²)   ##################################################################################
# Projection of U₀ in X²₀(ℝ²) : U₀ needs to represent a function in H²₀(Ω₀)
# We define 𝒯 as the trace operator (𝒯U = 0 means that U ∈ X²₀(ℝ²)) and 𝒯ᵀ as its adjoint
𝒯 = interval.(_trace_D₄(N)) ; 𝒯ᵀ = 𝒯'

# We build the operator L and its inverse L⁻¹. 
Δ = project(Laplacian(2), fourier, fourier,Interval{Float64})
L₁₁ = -I + Δ*λ₁i
L₁₁⁻¹ = interval.(ones(dimension(fourier)))./diag(coefficients(L₁₁))
#Finally we can build the projection of U₀ on X²₀ that we denote U₀ again Doing U₀ = U₀ - L₁₁⁻¹𝒯ᵀ(𝒯L₁₁⁻¹𝒯ᵀ)⁻¹𝒯
U₀_interval = U₀_interval - Sequence(fourier, vec(L₁₁⁻¹.*Matrix(𝒯ᵀ)*inv(Matrix(𝒯*(L₁₁⁻¹.*𝒯ᵀ)))*Matrix(𝒯)*U₀_interval[:]))

# # We define an operator P that help us to switch between the D₄ and exponential series
# # (as the theoretical analysis is done in exponential series)
# # For a linear operator K between D₄ fourier series, P*K*inv(P) gives the equivalent operator
# # on exponential series for the D₄ modes (the other modes can be found by computing the orbits of the stored modes)
# # In particular, if K is diagonal, then P*K*inv(P) = K
P = interval.(vec(_exp2D₄!(zeros(dimension(fourier)),fourier)))
P⁻¹ = interval.(ones(dimension(fourier))./P)

# Computation of B and its norm
V₀_interval = interval(2)*U₀_interval - interval(3)*λ₁i*U₀_interval^2
DGᵣ = project(Multiplication(V₀_interval),fourier,fourier,Interval{Float64})
Bᵣ = interval.(inv((I + mid.(DGᵣ).*mid.(L₁₁⁻¹)')))
Bᵣ_adjoint = LinearOperator(fourier,fourier, coefficients(Bᵣ)')
norm_Bᵣ = sqrt(opnorm(LinearOperator(coefficients(P.*(Bᵣ*Bᵣ_adjoint).*P⁻¹')),2))

# ################ 𝒴₀ BOUND ######################################################
# Computation of the 𝒴₀ bound for the reduced equation, defined in Lemma 5.3.
Ω₀ = (2*di)^2
tail_Gᵣ = U₀_interval^2 - λ₁i*U₀_interval^3
Gᵣ = project(tail_Gᵣ,fourier)
𝒴₀ = sqrt(Ω₀)*sqrt(norm(Bᵣ*(L₁₁*U₀_interval+Gᵣ),2)^2 + norm((tail_Gᵣ-Gᵣ),2)^2)
@show 𝒴₀

################################ 𝒵₂ BOUND ######################################################
# Computation of the 𝒵₂ bound for the reduced equation, defined in Lemma 5.4.
κ₂ = interval(1)/(interval(2)*sqrt(λ₁i*interval(π)))
𝕌₀² = project(Multiplication(U₀_interval*U₀_interval),fourier,fourier,Interval{Float64})
𝒵₂ = interval(6)*λ₁i*κ₂*sqrt(opnorm(LinearOperator(coefficients(P.*(Bᵣ_adjoint*𝕌₀²*Bᵣ).*P⁻¹')),2)+norm(U₀_interval,1)^2) + norm_Bᵣ*(interval(2)*κ₂ + interval(3)*λ₁i*κ₂^2*s₀)
@show 𝒵₂

################################ 𝒵ᵤ BOUND ######################################################
# Computation of the 𝒵ᵤ bound for the reduced equation, defined in Lemma 5.7.
a₁ = sqrt(1/λ₁i)

################################ 𝒵ᵤ₁ BOUND ######################################################
C₀f₁₁ = max(a₁^2*interval(interval(2)*exp(interval(5/4)))*(interval(2)/a₁)^(interval(1/4)),a₁^2*sqrt(interval(π)/(interval(2)*sqrt(a₁))))

# Computing the fourier series of E₁
E₁ = Sequence(D₄Fourier(4N,π/di), interval.(zeros(dimension(D₄Fourier(4N,π/di)))))
for n₂ = 0:4N
    for n₁ = n₂:4N
        E₁[(n₁,n₂)] = real(interval(1)/(interval(8)*di) * ((interval(-1))^n₁*sinc(n₂)*(interval(1)/(interval(2)*a₁-im*n₁*interval(π)/di) + interval(1)/(interval(2)*a₁ + im*n₁*interval(π)/di)) + (interval(-1))^n₂*sinc(n₁)*(interval(1)/(interval(2)*a₁-im*n₂*interval(π)/di) + interval(1)/(interval(2)*a₁ + im*n₂*interval(π)/di))))
    end
end

# Computes a larger operator to convert from D₄ to exponential since inner products will be of size 2N.
P2 = interval.(vec(_exp2D₄!(zeros(dimension(D₄Fourier(2N,π/di))),D₄Fourier(2N,π/di))))

#Inner Products
E₁V₀ = _conv_small(E₁,V₀_interval,2N)
_inner_prod_E₁V₀ = abs(coefficients(P2.*V₀_interval)'*coefficients(P2.*E₁V₀))

𝒵ᵤ₁ = sqrt(interval(2))*C₀f₁₁*(interval(1)-exp(-interval(4)*a₁*di)) * (interval(2π))^(interval(1/4))/a₁^(3/4) * sqrt(Ω₀) * sqrt(_inner_prod_E₁V₀)

################################ 𝒵ᵤ₂ BOUND ######################################################
# We begin by computing all the necessary constants.
# We start with C₁₁f₁₁ and C₁₂f₁₁ defined in Lemma 4.10.

C₁₁f₁₁ = a₁^3*sqrt(interval(π/2))*interval(1)/sqrt(a₁ + interval(1))*(interval(1) + interval(1)/a₁)
C₁₂f₁₁ = a₁^2*sqrt(interval(π/2))*(sqrt(interval(2))*a₁ + interval(1))

# Then, we define the constants C₁,𝒞₁₁, and 𝒞₂₁ defined in Lemma 4.10.
C₁ = sqrt(di^2/(interval(16)*a₁^2*interval(π)^5) + interval(1)/a₁^4 + di/a₁^3)
𝒞₁₁ = interval(2)*sqrt(Ω₀)*exp(-a₁*di)*(C₁₁f₁₁*exp(-a₁) + C₁₂f₁₁)/(a₁)
𝒞₂₁ = interval(2)*sqrt(Ω₀)*C₁₁f₁₁*sqrt(log(interval(2))^2 + interval(2)*log(interval(2)) + interval(2))

# Now, we compute 1_𝒟₀² and 1_𝒟₀, the Fourier series representations of the
# characteristic functions on 𝒟₀² and 𝒟₀ respectively. We do these computations
# using the functions _char_boundary_coeffs and _char_1D_boundary_coeffs.
print("Computing coefficients of characteristic function")
setprecision(80)
char = _char_boundary_coeffs(4N,frequency(U₀_interval)[1],di)
setprecision(80)
char1D = _char_1D_boundary_coeffs(4N,frequency(U₀_interval)[1],di)

P3 = interval.(exp2cos(2N))
P4 = interval.([1 ; sqrt(2)*ones(2N)])

Ṽ₀_interval = _Ṽⱼ_coeffs(V₀_interval)
V₀d_interval = _sequence_on_boundary(V₀_interval)

char∂ₓ₁V₀ = __conv_small(char,Ṽ₀_interval,2N)
_boundary_inner_prod∂ₓ₁V₀ = abs(coefficients(P3.*Ṽ₀_interval)'*coefficients(P3.*char∂ₓ₁V₀))

charV₀ = _conv_small(char,V₀_interval,2N)
_boundary_inner_prodV₀ = abs(coefficients(P2.*charV₀)'*coefficients(P2.*V₀_interval))

char1DV₀d = project(char1D*V₀d_interval,space(V₀d_interval))
_boundary_inner_prodV₀d = abs(coefficients(P4.*char1DV₀d)'*coefficients(P4.*V₀d_interval))

CV₀ = sqrt(interval(1/8) * sqrt(_boundary_inner_prod∂ₓ₁V₀)*sqrt(_boundary_inner_prodV₀) + interval(1)/(interval(2)*di) * interval(1/4) * _boundary_inner_prodV₀d)

𝒵ᵤ₂ = interval(4)/sqrt(Ω₀) * C₁ * (𝒞₁₁ * sqrt(_inner_prod_E₁V₀) + 𝒞₂₁*CV₀)
@show 𝒵ᵤ₂

𝒵ᵤ = sqrt(𝒵ᵤ₁^2 + 𝒵ᵤ₂^2)
@show 𝒵ᵤ

################################ Z₁ BOUND ######################################################
# Computation of the Z₁ bound for the reduced equation, defined in Lemma 5.6.
𝕍₀² = project(Multiplication(V₀_interval^2),fourier,fourier,Interval{Float64})
l₁₁ₙ = ((interval((N+1)π)/di)^2*λ₁i+interval(1))
Mᵣ = I + DGᵣ.*L₁₁⁻¹'
Mᵣ_adjoint = LinearOperator(fourier,fourier,coefficients(Mᵣ)')
Z₁₃ = sqrt(opnorm(LinearOperator(coefficients(P.*(L₁₁⁻¹.*(𝕍₀² - DGᵣ^2).*L₁₁⁻¹').*P⁻¹')),2))
Z₁₁ = sqrt(opnorm(LinearOperator(coefficients(P.*((I-Bᵣ*Mᵣ)*(I-Mᵣ_adjoint*Bᵣ_adjoint)).*P⁻¹')),2))
Z₁₂ = interval(1)/l₁₁ₙ*sqrt(opnorm(LinearOperator(coefficients(P.*(Bᵣ*(𝕍₀² - DGᵣ^2)*Bᵣ_adjoint).*P⁻¹')),2))
Z₁₄ = interval(1)/l₁₁ₙ*norm(V₀_interval,1)
Z₁ = φ(Z₁₁,Z₁₂,Z₁₃,Z₁₄)

𝒵₁ = Z₁ + norm_Bᵣ*𝒵ᵤ
@show 𝒵₁

#Perform the Computer Assisted Proof
s_min = sup((interval(1) - 𝒵₁ - sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂)
s_max = min(inf((interval(1) - 𝒵₁ + sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂), inf((interval(1)-𝒵₁)/𝒵₂))
CAP(𝒴₀,𝒵₁,𝒵₂,s₀)

################################ Proof of Periodic Solution #################################################
# The value of κ̂₂ defined in Theorem 5.8
κ̂₂ = sqrt(interval(1)/(interval(4π)*λ₁i) + interval(1)/(interval(4)*di^2) + interval(1)/(interval(2)*di) * interval(π)/sqrt(λ₁i))

# We can now perform the computer assisted proof for the branch of periodic solutions
𝒵₁_hat = 𝒵₁+norm_Bᵣ*𝒵ᵤ
𝒵₂_hat = interval(6)*λ₁i*κ̂₂*sqrt(opnorm(LinearOperator(coefficients(P.*(Bᵣ_adjoint*𝕌₀²*Bᵣ).*P⁻¹')),2)+norm(U₀_interval,1)^2) + norm_Bᵣ*(interval(2)*κ̂₂ + interval(3)*λ₁i*κ̂₂^2*s₀)
ŝ_min = sup((interval(1) - 𝒵₁_hat - sqrt((interval(1) - 𝒵₁_hat)^2 - interval(2)*𝒴₀*𝒵₂_hat))/𝒵₂_hat)
ŝ_max = min(inf((interval(1) - 𝒵₁_hat + sqrt((interval(1) - 𝒵₁_hat)^2 - interval(2)*𝒴₀*𝒵₂_hat))/𝒵₂_hat), inf((interval(1)-𝒵₁_hat)/𝒵₂_hat))
CAP(𝒴₀,𝒵₁_hat,𝒵₂_hat,s₀)