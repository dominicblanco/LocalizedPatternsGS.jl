#Computer assisted proof of solutions for the 2D Gray Scott system of equations
# λ₁Δu₁ - u₁ + (u₂ + 1 - λ₁u₁)u₁² = 0
# Δu₂ - λ₂u₂ + (λ₁λ₂ - 1)u₁ = 0
# The following code computes the solution and rigorously proves the results given in section 6 of
# "The 2D Gray-Scott sysem of equations: constructive proofs of existence of localized stationary patterns"  M. Cadiot and D. Blanco

# We computed the approximate solution using continuation. We provide the data here.
# From this we can check if the proof of the solution is verified or not. We essentially prove Theorems 6.1, 6.2, 6.3, and 6.4.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, IntervalArithmetic, IntervalLinearAlgebra, LinearAlgebra, JLD2

# Needed additional sequence structures for RadiiPolynomial (see Section 6)
# You can download this file from Github and put it in the same folder as this one. Then, it will include automatically.
include("D4Fourier.jl")

#####################################################################################################################################################################


#################################### List of the needed functions : go directly to line 245 for the main code ################################################# 

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

# Computes convolution of D₄Fourier sequences up to order N
function _conv_smallbig(u,v,N)
    #Computes u*v only up to order N
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(D₄Fourier(N,frequency(u)[1]), interval.(big.(zeros(dimension(D₄Fourier(N,frequency(u)[1]))))))
    for i₂ ∈ 0:N
        for i₁ ∈ i₂:N
            Cᵢ = interval(zero(BigFloat))
            setprecision(80)
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
                    setprecision(80)
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Performs the estimate of Lemma 4.1
function φ(A,B,C,D)
    O₁ = max(A,D) + max(B,C)
    O₂ = sqrt(A^2 + D^2 + B^2 + C^2)
    return min(O₁,O₂)
end

# Computes the Fourier coefficients of 1_𝒟₀²
function _char_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f)⊗Fourier(N,f), Interval.(complex.(big.(zeros((2N+1)^2)))))
    for n₂ = -N:N
        for n₁ = -N:N
            char[(n₁,n₂)] = interval(big(1))/(interval(big(4))*d^2) * exp(1im*n₁*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1))))*exp(1im*n₂*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1)))) * sinc(n₁/d*interval(big(1/2)))*sinc(n₂/d*interval(big(1/2)))
        end
    end
    rchar = Sequence(D₄Fourier(N,f), interval.(big.(zeros(dimension(D₄Fourier(N,f))))))
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
    anew = Sequence(CosFourier(N,f), interval.(big.(zeros(N+1))))
    for n₁ = 0:N
        for n₂ = -N:N
            anew[n₁] += a[(max(n₁,abs(n₂)),min(n₁,abs(n₂)))]*(-1)^n₂
        end
    end
    return anew
end

# Computes the Fourier coefficients of 1_𝒟₀
function _char_1D_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f), Interval.(complex.(big.(zeros((2N+1))))))
    for n = -N:N
        char[n] = interval(big(1))/(interval(big(2))*d) * exp(1im*n*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1)))) * sinc(n/d*interval(big(1/2)))
    end
    rchar = Sequence(CosFourier(N,f), interval.(big.(zeros(N+1))))
    for n = 0:N
        rchar[n] = real(char[n] + char[-n])
    end
    return rchar
end

# Computes the function needed to take the convolution with ∂ₓ₁V₁ᴺ
# We denote by (Ṽⱼ)ₘ = |m̃₁| (Vⱼᴺ)ₘ 
function _Ṽⱼ_coeffs(Vⱼᴺ)
    N = order(Vⱼᴺ)[1]
    f = frequency(Vⱼᴺ)[1]
    Ṽⱼ = Sequence(CosFourier(N,f)⊗CosFourier(N,f), interval.(big.(zeros((N+1)^2))))
    for n₁ = 0:N
        for n₂ = 0:N
            Ṽⱼ[(n₁,n₂)] = abs(n₁)*f*Vⱼᴺ[(max(n₁,n₂),min(n₁,n₂))]
        end
    end
    return Ṽⱼ
end

# Checks the conditions of the Radii-Polynomial Theorem 3.1.
function CAP(𝒴₀,𝒵₁,𝒵₂,r₀)
    if 𝒵₁ + 𝒵₂*r₀ < 1
        if interval(1/2)*𝒵₂*r₀^2 - (interval(1)-𝒵₁)*r₀ + 𝒴₀ < 0
          display("The proof was successful for r₀ = ")
          display(sup(r₀))  
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

################### PROOF OF SOLUTIONS : MAIN CODE #################################################################################################################################################
# To run the code, click the run button in the terminal.
# Below is the data for three solutions. To prove one of them, comment the others. 
# You can write a comment using the # sign. To comment multiple lines, use #= and end the comment with =#.
# Make sure to download the necessary file that includes the data you wish to prove!
setprecision(80) #Sets the precision for the proof.
U₀ = load("U0_leaf","U₀") #Leaf solution
N₀ = 240    # number of Fourier modes for leaf: 0 ≤ n₂ ≤ n₁ ≤ N₀ for D₄ series
N = 180     # number of Fourier modes for operators for the leaf.
d = 22  ; di = interval(d) ; dbig = interval(big(d))   # size of the domain for the leaf
λ₂ = 3.74 ; λ₂i = interval(λ₂) ; λ₂big = interval(big(λ₂)) # values of parameters for the leaf
λ₁ = 0.0566 ; λ₁i = interval(λ₁) ; λ₁big = interval(big(λ₁))
r₀ = interval(6e-6) # value of r₀ for 𝒵₂

#=U₀ = load("U0_ring","U₀") #Ring solution 
N₀ = 80   # number of Fourier modes for the ring
N = 60    # number of Fourier modes for operators for the ring
setprecision(80)
d = 10 ; di = interval(d) ; dbig = interval(big(d))   # size of the domain for the ring
λ₂ = 3.73  ; λ₂i = interval(λ₂) ; λ₂big = interval(big(λ₂)) # values of parameters for the ring
λ₁ = 0.0567 ; λ₁i = interval(λ₁) ; λ₁big = interval(big(λ₁))
r₀ = interval(6e-6) # value of r₀ for 𝒵₂=#

#=U₀ = load("U0_spikeaway","U₀") #Spike solution away from λ₁λ₂ = 1
N₀ = 50   # number of Fourier modes for the ring
N = 20    # number of Fourier modes for operators for the ring
setprecision(80)
d = 8 ; di = interval(d) ; dbig = interval(big(d))   # size of the domain for the spike away from λ₁λ₂ = 1
λ₂ = 10 ; λ₂i = interval(λ₂) ; λ₂big = interval(big(λ₂)) # values of parameters for the spike away from λ₁λ₂ = 1
λ₁ = 1/9 ; λ₁i = interval(λ₁) ; λ₁big = interval(big(λ₁))
r₀ = interval(6e-6) # value of r₀ for 𝒵₂=#
U₀₁ = component(U₀,1)
U₀₂ = component(U₀,2)
fourier_long = D₄Fourier(N₀,π/di)
fourier = D₄Fourier(N,π/di)
U₀₁ = project(U₀₁,D₄Fourier(N₀,π/d))
U₀₂ = project(U₀₂,D₄Fourier(N₀,π/d))
print("Creating intervals")
U₀₁_interval = Sequence(fourier_long, interval.(coefficients(U₀₁)))
setprecision(80)
U₀₁big = Sequence(fourier_long, interval.(big.(coefficients(U₀₁))))
U₀₂_interval = Sequence(fourier_long, interval.(coefficients(U₀₂)))
setprecision(80)
U₀₂big = Sequence(fourier_long, interval.(big.(coefficients(U₀₂))))
#################################################   Projection on X²₀   ##################################################################################
# Projection of U₀ in X²₀ : X₀ needs to represent a function in H²₀(Ω₀) × H²₀(Ω₀)
# We define 𝒯 as the trace operator (𝒯U = 0 means that U ∈ X²₀) and Sᵀ as its adjoint
setprecision(80)
𝒯 = interval.(big.(_trace_D₄(N₀))) ; 𝒯ᵀ = 𝒯'
# We build the operators Lᵢⱼ and their inverses. We do this as we break things down
# into blocks to avoid memory issues. 
L₁₁ = diag(coefficients(project(Laplacian(2), fourier, fourier,Interval{Float64})*λ₁i - I))
L₁₁_long = diag(coefficients(project(Laplacian(2), fourier_long, fourier_long,Interval{Float64})*λ₁i - I))
setprecision(80)
L₁₁big_long = diag(coefficients(project(Laplacian(2), fourier_long, fourier_long,Interval{BigFloat})*λ₁big - I))

L₁₁⁻¹ = interval.(ones(dimension(fourier)))./L₁₁
L₁₁⁻¹_long = interval.(ones(dimension(fourier_long)))./L₁₁_long
L₁₁ = Nothing
setprecision(80)
L₁₁⁻¹big_long = interval.(big.(ones(dimension(fourier_long))))./L₁₁big_long
L₁₁big_long = Nothing

L₂₁ = (λ₁i*λ₂i-interval(1))*interval.(ones(dimension(fourier)))
L₂₁_long = (λ₁i*λ₂i-interval(1))*interval.(ones(dimension(fourier_long)))
setprecision(80)
L₂₁big_long = (λ₁big*λ₂big-interval(1))*interval.(big.(ones(dimension(fourier_long))))

L₂₂ = diag(coefficients(project(Laplacian(2), fourier, fourier,Interval{Float64}) - λ₂i*I))
L₂₂_long = diag(coefficients(project(Laplacian(2), fourier_long, fourier_long,Interval{Float64}) - λ₂i*I))
setprecision(80)
L₂₂big_long = diag(coefficients(project(Laplacian(2), fourier_long, fourier_long,Interval{BigFloat}) - λ₂big*I))

L₂₂⁻¹ = ones(dimension(fourier))./L₂₂
L₂₂⁻¹_long = ones(dimension(fourier_long))./L₂₂_long
L₂₂ = Nothing
setprecision(80)
L₂₂⁻¹big_long = interval.(big.(ones(dimension(fourier_long))))./L₂₂big_long
L₂₂big_long = Nothing

# Finally we can build the projection of U₀ on X²₀ that we denote U₀ again Doing U₀ = U₀ - L⁻¹Sᵀ(SL⁻¹Sᵀ)⁻¹S
# Note that by expanding the expression above into blocks, we obtain the quantites below
# results for U₀₁ and U₀₂
print("Computing the trace")
setprecision(80)
M = solve(Matrix(𝒯*((L₁₁⁻¹big_long .- (L₂₂⁻¹big_long.*L₂₁big_long.*L₁₁⁻¹big_long) .+ L₂₂⁻¹big_long).*𝒯ᵀ)),𝒯*coefficients(U₀₁big+U₀₂big))
setprecision(80)
U₀₁big = U₀₁big - Sequence(fourier_long,L₁₁⁻¹big_long.*𝒯ᵀ*M)
setprecision(80)
U₀₂big = U₀₂big - Sequence(fourier_long,(-(L₂₂⁻¹big_long.*L₂₁big_long.*L₁₁⁻¹big_long).*𝒯'+ L₂₂⁻¹big_long.*𝒯')*M)
U₀₁_interval = Interval.(Float64.(inf.(U₀₁big),RoundDown),Float64.(sup.(U₀₁big),RoundUp) )
U₀₂_interval = Interval.(Float64.(inf.(U₀₂big),RoundDown),Float64.(sup.(U₀₂big),RoundUp) )
𝒯 = Nothing
𝒯ᵀ = Nothing
L₁₁⁻¹big_long = Nothing
L₂₁big_long = Nothing
L₂₂⁻¹big_long = Nothing
# # We define an operator P that help us to switch between the D₄ and exponential series
# # (as the theoretical analysis is done in exponential series)
# # For a linear operator B between D₄ fourier series, P*B*inv(P) gives the equivalent operator
# # on exponential series for the D₄ modes (the other modes can be found by computing the orbits of the stored modes)
# # In particular, if B is diagonal, then P*B*inv(P) = B
P = vec(_exp2D₄!(zeros(dimension(fourier)),fourier))
P⁻¹ = ones(dimension(fourier))./P
P = interval.(P)
P⁻¹ = interval.(P⁻¹)
# Computation of B₁₁,B₁₂ and the norm of B₁₁.
print("Computing U₀₁²")
U₀₁²big = U₀₁big*U₀₁big
print("Computing U₀₂U₀₁")
U₀₂U₀₁big = U₀₂big*U₀₁big
V₁big = interval(2)*U₀₂U₀₁big + interval(2)*U₀₁big - interval(3)*λ₁i*U₀₁²big
V₂big = U₀₁²big
U₀₁²_interval = Interval.(Float64.(inf.(U₀₁²big),RoundDown),Float64.(sup.(U₀₁²big),RoundUp) )
V₁_interval = Interval.(Float64.(inf.(V₁big),RoundDown),Float64.(sup.(V₁big),RoundUp) )
V₂_interval = Interval.(Float64.(inf.(V₂big),RoundDown),Float64.(sup.(V₂big),RoundUp) )

DG₁₁ = project(Multiplication(V₁_interval),fourier,fourier,Interval{Float64})
DG₁₂ = project(Multiplication(V₂_interval),fourier,fourier,Interval{Float64})
print("Computing B")
B₁₁ = interval.(inv(mid.(I + DG₁₁.*L₁₁⁻¹' - DG₁₂.*(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹)')))
B₁₂ = -B₁₁*DG₁₂.*L₂₂⁻¹'
B₁₁_adjoint = LinearOperator(fourier,fourier, coefficients(B₁₁)')
print("Computing norm of B₁₁")
norm_B₁₁ = (opnorm(LinearOperator(coefficients(P.*((B₁₁_adjoint*B₁₁)^2).*P⁻¹')),2))^(interval(1/4))
@show norm_B₁₁

# ################ 𝒴₀ BOUND ######################################################
# Computation of the 𝒴₀ bound, defined in Lemma 4.3.
print("Computing tail_G₁")
tail_G₁ = (U₀₂_interval + interval(1) - λ₁i*U₀₁_interval)*U₀₁²_interval
G₁ = project(tail_G₁,fourier)

# These are the components of 𝒴₀ expanded. 
# That is, the components of ||(πᴺ + BΩ₀)(LU₀ + G(U₀))||₂².
print("Computing 𝒴₀ components")
𝒴₀¹ = B₁₁*project(L₁₁_long.*U₀₁_interval + G₁,fourier) + B₁₂*project(L₂₁_long.*U₀₁_interval + L₂₂_long.*U₀₂_interval,fourier)
𝒴₀² = project(L₂₁_long.*U₀₁_interval + L₂₂_long.*U₀₂_interval,fourier)

# These are the tail components of 𝒴₀ as a result of choosing N ≠ N₀ and having a nonlinear term
# That is the components of ||(π³ᴺ⁰ - πᴺ)LU₀ + (π³ᴺ⁰ - πᴺ)G(U₀)||₂²
𝒴₀¹∞ = L₁₁_long.*(U₀₁_interval - project(U₀₁_interval,fourier))  + tail_G₁ - G₁ + L₂₁_long.*(U₀₁_interval - project(U₀₁_interval,fourier)) + L₂₂_long.*(U₀₂_interval - project(U₀₂_interval,fourier))
𝒴₀²∞ = L₂₁_long.*(U₀₁_interval - project(U₀₁_interval,fourier)) +L₂₂_long.*(U₀₂_interval - project(U₀₂_interval,fourier))

L₁₁_long = Nothing
L₂₁_long = Nothing
L₂₂_long = Nothing

Ω₀ = (2di)^2
𝒴₀ = sqrt(Ω₀)*sqrt(norm(𝒴₀¹,2)^2 + norm(𝒴₀²,2)^2 + norm(𝒴₀¹∞,2)^2 + norm(𝒴₀²∞,2)^2)
@show 𝒴₀

################################ 𝒵₂ BOUND ######################################################
# Computation of the 𝒵₂ bound defined in Lemma 4.5.
# Computation of the constants κ₂,κ₃, and κ₀
print("Computing 𝒵₂")
κ₂ = interval(1)/(interval(2)*sqrt(λ₁i*interval(π)))
@show κ₂
κ₃ = sqrt(interval(2))/(interval(4π)) * min(interval(1)/(λ₁i*λ₂i),interval(1)/sqrt(λ₁i*λ₂i))
@show κ₃
κ₀ = min(max(((λ₁i*κ₂ + interval(1)/(interval(2)*sqrt(λ₂i*interval(π))))^2 + interval(1)/(interval(4π)*λ₂i))^(interval(1/2)), sqrt(interval(2))/(interval(2)*sqrt(λ₂i*interval(π)))), κ₂*interval(1)/λ₂i * ((interval(1)-λ₁i*λ₂i)^2 + interval(1))^(interval(1/2)))
@show κ₀
Q = U₀₂_interval + interval(1) -interval(3)*λ₁i*U₀₁_interval
ℚ = project(Multiplication(Q),fourier,fourier,Interval{Float64})
𝕌₀₁ = project(Multiplication(U₀₁_interval),fourier,fourier,Interval{Float64})
Q² = Q*Q
ℚ² = project(Multiplication(Q²),fourier,fourier,Interval{Float64})
𝕌₀₁² = project(Multiplication(U₀₁²_interval),fourier,fourier,Interval{Float64})
print("Computing 𝒵₂ⱼ for j = 1,2,3")
𝒵₂₁ = opnorm(LinearOperator(P.*coefficients(B₁₁*(ℚ² + 𝕌₀₁²)*B₁₁_adjoint).*P⁻¹'),2)
𝒵₂₂ = sqrt(opnorm(LinearOperator(P.*coefficients(B₁₁*((ℚ²+𝕌₀₁²) - (ℚ^2 + 𝕌₀₁^2))*B₁₁_adjoint).*P⁻¹'),2))
𝒵₂₃ = norm(Q² + U₀₁²_interval,1)

𝒵₂ = interval(2)*(sqrt(φ(𝒵₂₁,𝒵₂₂,𝒵₂₂,𝒵₂₃))*sqrt(κ₂^2+interval(4)*κ₀^2)) + norm_B₁₁*interval(3)*κ₃*r₀
@show 𝒵₂

################################ 𝒵ᵤ₁ BOUND ######################################################
# Computation of the 𝒵ᵤ₁ bound defined in Lemma 4.9.
print("Starting 𝒵ᵤ")
setprecision(80)
a₁big = sqrt(interval(1)/λ₁big)
setprecision(80)
a₂big = sqrt(λ₂big)
a₁ = sqrt(interval(1)/λ₁i)
a₂ = sqrt(λ₂i)
# The constants C₀f₁₁ and C₀f₂₂ in Lemma 4.8
C₀f₁₁ = max(a₁^2*interval(interval(2)*exp(interval(5/4)))*(interval(2)/a₁)^(interval(1/4)),a₁^2*sqrt(interval(π)/(interval(2)*sqrt(a₁))))
C₀f₂₂ = max(interval(interval(2)*exp(interval(5/4)))*(interval(2)/a₂)^(1/4),sqrt(interval(π)/(interval(2)*sqrt(a₂))))
# Computing the fourier series of E₁ and E₂ defined in Lemma 4.9.
setprecision(80)
E₁big = Sequence(D₄Fourier(4N,π/di), interval.(big.(zeros(dimension(D₄Fourier(4N,π/di))))))
setprecision(80)
E₂big = Sequence(D₄Fourier(4N,π/di), interval.(big.(zeros(dimension(D₄Fourier(4N,π/di))))))
for n₂ = 0:4N
    for n₁ = n₂:4N
        setprecision(80)
        E₁big[(n₁,n₂)] = real(interval(big(1))/(interval(big(8))*dbig) * ((-interval(big(1)))^n₁*sinc(n₂)*(interval(big(1))/(interval(big(2))*a₁big-im*n₁*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₁big + im*n₁*interval(big(π))/dbig)) + (-interval(big(1)))^n₂*sinc(n₁)*(interval(big(1))/(interval(big(2))*a₁big-im*n₂*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₁big + im*n₂*interval(big(π))/dbig))))
        setprecision(80)
        E₂big[(n₁,n₂)] = real(interval(big(1))/(interval(big(8))*dbig) * ((-interval(big(1)))^n₁*sinc(n₂)*(interval(big(1))/(interval(big(2))*a₂big-im*n₁*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₂big + im*n₁*interval(big(π))/dbig)) + (-interval(big(1)))^n₂*sinc(n₁)*(interval(big(1))/(interval(big(2))*a₂big-im*n₂*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₂big + im*n₂*interval(big(π))/dbig))))
    end
end
E₁ = Interval.(Float64.(inf.(E₁big),RoundDown),Float64.(sup.(E₁big),RoundUp) )
E₂ = Interval.(Float64.(inf.(E₂big),RoundDown),Float64.(sup.(E₂big),RoundUp) )
# Computes a larger operator to convert from D₄ to exponential since inner products will be of size 2N.
P2 = interval.(vec(_exp2D₄!(zeros(dimension(D₄Fourier(2N,π/di))),D₄Fourier(2N,π/di))))

setprecision(80)
P2big = interval.(big.(vec(_exp2D₄!(zeros(dimension(D₄Fourier(2N,π/di))),D₄Fourier(2N,π/di)))))

# Computation of the 𝒵ᵤ₁₁ bound, the first quantity defined in Lemma 4.9.
print("Computing 𝒵ᵤ₁₁")
V₁ᴺ_interval = project(V₁_interval,D₄Fourier(2N,π/di))
V₂ᴺ_interval = project(V₂_interval,D₄Fourier(2N,π/di))

#For spike and ring, use lines 457 through 461
E₁V₁ = _conv_small(E₁,V₁ᴺ_interval, 2N)
_inner_prod_E₁V₁ = abs(coefficients(P2.*V₁ᴺ_interval)'*coefficients(P2.*E₁V₁))
@show _inner_prod_E₁V₁
𝒵ᵤ₁₁ = sqrt(interval(2))*C₀f₁₁*(1-exp(-4a₁*di)) * (interval(2π))^(interval(1/4))/a₁^(interval(3/4))*sqrt(Ω₀) *sqrt(_inner_prod_E₁V₁)  
@show 𝒵ᵤ₁₁

#For the leaf, use lines 465 through 472
# More specifically, comment lines 457 through 461 and uncommnet lines 465 through 471.
#=setprecision(80)
E₁V₁big = _conv_smallbig(E₁big,V₁ᴺbig,2N)
setprecision(80)
_inner_prod_E₁V₁big = abs(coefficients(P2big.*V₁ᴺbig)'*coefficients(P2big.*E₁V₁big))
@show _inner_prod_E₁V₁big
𝒵ᵤ₁₁big = sqrt(interval(2))*C₀f₁₁*(1-exp(-4a₁*di)) * (interval(2π))^(1/4)/a₁^(3/4)*sqrt(Ω₀) *sqrt(_inner_prod_E₁V₁big)  
𝒵ᵤ₁₁ = Interval(Float64(inf(𝒵ᵤ₁₁big),RoundDown),Float64(sup(𝒵ᵤ₁₁big),RoundUp) )
@show 𝒵ᵤ₁₁=#

# Computation of the 𝒵ᵤ₁₂ bound, the second quantity defined in Lemma 4.9.
print("Computing 𝒵ᵤ₁₂")
# For spike and ring, use lines 477 through 481
E₂V₂ = _conv_small(E₂,V₂ᴺ_interval, 2N)
_inner_prod_E₂V₂ = abs(coefficients(P2.*V₂ᴺ_interval)'*coefficients(P2.*E₂V₂))
@show _inner_prod_E₂V₂
𝒵ᵤ₁₂ = sqrt(interval(2))*C₀f₂₂*(interval(1)-exp(-4a₂*di)) * (interval(2π))^(interval(1/4))/a₂^(interval(3/4))*sqrt(Ω₀)*sqrt(_inner_prod_E₂V₂)  
@show 𝒵ᵤ₁₂

# For the leaf, use lines 485 through 492
# More specifically, comment lines 477 through 481 and uncomment 485 through 492.
#=setprecision(80)
E₂V₂big = _conv_smallbig(E₂big,V₂ᴺbig,2N)
setprecision(80)
_inner_prod_E₂V₂big = abs(coefficients(P2big.*V₂ᴺbig)'*coefficients(P2big.*E₂V₂big))
@show _inner_prod_E₂V₂big
𝒵ᵤ₁₂big = sqrt(interval(2))*C₀f₂₂*(1-exp(-4a₂*di)) * (interval(2π))^(interval(1/4))/a₂^(interval(3/4))*sqrt(Ω₀)*sqrt(_inner_prod_E₂V₂big)  
𝒵ᵤ₁₂ = Interval(Float64(inf(𝒵ᵤ₁₂big),RoundDown),Float64(sup(𝒵ᵤ₁₂big),RoundUp) )
@show 𝒵ᵤ₁₂=#

#Since 1/λ₁i > λ₂i in all cases we prove, we are in the first case for 𝒵ᵤ₁₃, the third expression defined in Lemma 4.8.
𝒵ᵤ₁₃ = 𝒵ᵤ₁₂  

#Now, we compute the full 𝒵ᵤ₁ bound concluding the computation of Lemma 4.9.
𝒵ᵤ₁ = sqrt((𝒵ᵤ₁₁ + 𝒵ᵤ₁₃)^2 + 𝒵ᵤ₁₂^2)
@show 𝒵ᵤ₁
################################ 𝒵ᵤ₂ BOUND ######################################################
# Computation of the 𝒵ᵤ₂ bound defined in Lemma 4.10.
# We begin by computing all the necessary constants.
# We start with C₁₁f₁₁,C₁₂f₁₁,C₁₁f₂₂, and C₁₂f₂₂ defined in Lemma 4.10.
print("Computing 𝒵ᵤ₂")
C₁₁f₁₁ = a₁^3*sqrt(interval(π/2))*interval(1)/sqrt(a₁ + interval(1))*(interval(1) + interval(1)/a₁)
C₁₂f₁₁ = a₁^2*sqrt(interval(π/2))*(sqrt(interval(2))*a₁ + interval(1))

C₁₁f₂₂ = a₂*sqrt(interval(π/2))*1/sqrt(a₂ + interval(1))*(interval(1)+interval(1)/a₂)
C₁₂f₂₂ = sqrt(interval(π/2))*(sqrt(interval(2))*a₂ + interval(1))

# Next, we define the constants Cⱼ,𝒞₁ⱼ, and 𝒞₂ⱼ for j = 1,2 defined in Lemma 4.10.
C₁ = sqrt(di^2/(interval(16)*a₁^2*interval(π)^5) + interval(1)/a₁^4 + di/a₁^3)
C₂ = sqrt(di^2/(interval(16)*a₂^2*interval(π)^5) + interval(1)/a₂^4 + di/a₂^3)
𝒞₁₁ = interval(2)*sqrt(Ω₀)*exp(-a₁*di)*(C₁₁f₁₁*exp(-a₁) + C₁₂f₁₁)/a₁
𝒞₂₁ = interval(2)*sqrt(Ω₀)*C₁₁f₁₁*sqrt(log(interval(2))^2 + interval(2)*log(interval(2)) + interval(2))
𝒞₁₂ = interval(2)*sqrt(Ω₀)*exp(-a₂*di)*(C₁₁f₂₂*exp(-a₂) + C₁₂f₂₂)/a₂
𝒞₂₂ = interval(2)*sqrt(Ω₀)*C₁₁f₂₂*sqrt(log(interval(2))^2 + interval(2)*log(interval(2)) + interval(2))

# Now, we compute 1_𝒟₀² and 1_𝒟₀, the Fourier series representations of the
# characteristic functions on 𝒟₀² and 𝒟₀ respectively. We do these computations
# using the functions _char_boundary_coeffs and _char_1D_boundary_coeffs.
print("Computing coefficients of characteristic function")
setprecision(80)
char = _char_boundary_coeffs(4N,frequency(U₀₁_interval)[1],dbig)
setprecision(80)
char1D = _char_1D_boundary_coeffs(4N,frequency(U₀₁_interval)[1],dbig)
#Note that the function char is the characteristic function on all four corners.
# Indeed, since Vⱼᴺ is D₄-symmetric, we can compute the norm of Vⱼᴺ on all four corners
# and divide by 1/4 to obtain the result. For the norm involving ∂ₓ₁v₁ᴺ, we upper bound
# by the norm in the upper right corner by the norm on all four corners. This allows us
# to compute the convolution of a D₄ and D₂ sequence, which is less computationally expensive.
# Indeed, (∂ₓ₁v₁ᴺ)² is an even (D₂) function.

# Similarly, char1D is the characteristic function of 𝒟₀ ∪ (-d,-d+1). Since v₁ᴺ(d,⋅)
# is even, we can take the norm on this domain and multiply by 1/2.
P3 = interval.(exp2cos(2N))
P4 = interval.([1 ; sqrt(2)*ones(2N)])

setprecision(80)
V₁ᴺbig = project(V₁big,D₄Fourier(2N,π/di))
setprecision(80)
Ṽ₁big = _Ṽⱼ_coeffs(V₁ᴺbig)
setprecision(80)
V₁ᴺdbig = _sequence_on_boundary(V₁ᴺbig)

setprecision(80)
V₂ᴺbig = project(V₂big,D₄Fourier(2N,π/di))
setprecision(80)
Ṽ₂big = _Ṽⱼ_coeffs(V₂ᴺbig)
setprecision(80)
V₂ᴺdbig = _sequence_on_boundary(V₂ᴺbig)

char = Interval.(Float64.(inf.(char),RoundDown),Float64.(sup.(char),RoundUp) ) 
char1D = Interval.(Float64.(inf.(char1D),RoundDown),Float64.(sup.(char1D),RoundUp) ) 
Ṽ₁_interval = Interval.(Float64.(inf.(Ṽ₁big),RoundDown),Float64.(sup.(Ṽ₁big),RoundUp) ) 
V₁ᴺd_interval = Interval.(Float64.(inf.(V₁ᴺdbig),RoundDown),Float64.(sup.(V₁ᴺdbig),RoundUp) ) 
Ṽ₂_interval = Interval.(Float64.(inf.(Ṽ₂big),RoundDown),Float64.(sup.(Ṽ₂big),RoundUp) ) 
V₂ᴺd_interval = Interval.(Float64.(inf.(V₂ᴺdbig),RoundDown),Float64.(sup.(V₂ᴺdbig),RoundUp) ) 


# We now compute each 𝒵ᵤ₂ⱼ bound for  j = 1,2,3. Beginning with 𝒵ᵤ₂₁,
print("Computing 𝒵ᵤ₂₁")
charṼ₁ = __conv_small(char,Ṽ₁_interval,2N)
_boundary_inner_prod∂ₓ₁V₁ = abs(coefficients(P3.*charṼ₁)'*coefficients(P3.*Ṽ₁_interval))
@show _boundary_inner_prod∂ₓ₁V₁

charV₁ = _conv_small(char,V₁ᴺ_interval,2N)
_boundary_inner_prodV₁ = abs(coefficients(P2.*charV₁)'*coefficients(P2.*V₁ᴺ_interval))
@show _boundary_inner_prodV₁

char1DV₁d = project(char1D*V₁ᴺd_interval,space(V₁ᴺd_interval))
_boundary_inner_prodV₁d = abs(coefficients(P4.*char1DV₁d)'*coefficients(P4.*V₁ᴺd_interval))
@show _boundary_inner_prodV₁d

CV₁ᴺ = sqrt(interval(1/8) * sqrt(_boundary_inner_prod∂ₓ₁V₁)*sqrt(_boundary_inner_prodV₁) + interval(1)/(2di) * interval(1/4) * _boundary_inner_prodV₁d)

𝒵ᵤ₂₁ = interval(4)/sqrt(Ω₀) * C₁ * (𝒞₁₁ * sqrt(_inner_prod_E₁V₁) + 𝒞₂₁*CV₁ᴺ)
@show 𝒵ᵤ₂₁

# Next, we compute 𝒵ᵤ₂₂
print("Computing 𝒵ᵤ₂₂")
charṼ₂ = __conv_small(char,Ṽ₂_interval,2N)
_boundary_inner_prod∂ₓ₁V₂ = abs(coefficients(P3.*charṼ₂)'*coefficients(P3.*Ṽ₂_interval))
@show _boundary_inner_prod∂ₓ₁V₂

charV₂ = _conv_small(char,V₂ᴺ_interval,2N)
_boundary_inner_prodV₂ = abs(coefficients(P2.*charV₂)'*coefficients(P2.*V₂ᴺ_interval))
@show _boundary_inner_prodV₂

char1DV₂d = project(char1D*V₂ᴺd_interval,space(V₂ᴺd_interval))
_boundary_inner_prodV₂d = abs(coefficients(P4.*char1DV₂d)'*coefficients(P4.*V₂ᴺd_interval))
@show _boundary_inner_prodV₂d

CV₂ᴺ = sqrt(interval(1/8) * sqrt(_boundary_inner_prod∂ₓ₁V₂)*sqrt(_boundary_inner_prodV₂) + interval(1)/(2di) * interval(1/4) * _boundary_inner_prodV₂d)

𝒵ᵤ₂₂ = interval(4)/sqrt(Ω₀) * C₂ * (𝒞₁₂ * sqrt(_inner_prod_E₂V₂) + 𝒞₂₂*CV₂ᴺ)
@show 𝒵ᵤ₂₂

# Finally, we compute 𝒵ᵤ₂₃. Note that we require an additional inner product for its computation.
print("Computing 𝒵ᵤ₂₃")
E₁V₂ = _conv_small(E₁,V₂ᴺ_interval, 2N)
_inner_prod_E₁V₂ = abs(coefficients(P2.*V₂ᴺ_interval)'*coefficients(P2.*E₁V₂))
@show _inner_prod_E₁V₂

𝒵ᵤ₂₃ =  min(C₁,C₂) *(𝒵ᵤ₂₂/C₂ + interval(4)*λ₁i/sqrt(Ω₀)*(𝒞₁₁ * sqrt(_inner_prod_E₁V₂) + 𝒞₂₁*CV₂ᴺ))
@show 𝒵ᵤ₂₃

# Finally, we can compute 𝒵ᵤ₂
𝒵ᵤ₂ = sqrt((𝒵ᵤ₂₁ + 𝒵ᵤ₂₃)^2 + 𝒵ᵤ₂₂^2)
@show 𝒵ᵤ₂

# Now, we define 𝒵ᵤ as
𝒵ᵤ = sqrt(𝒵ᵤ₁^2 + 𝒵ᵤ₂^2)
@show 𝒵ᵤ
################################ Z₁ BOUND ######################################################
# Computation of the Z₁ bound defined in Lemma 4.7.
# The Jᵢ's are the various computations of the form DG₁ⱼπₙDG₁ₖ.
# We compute them in advance as we need them multiple times.
print("Computing Jᵢ's")
V₁ᴺ²_interval = V₁ᴺ_interval*V₁ᴺ_interval
J₁ = project(Multiplication(V₁ᴺ²_interval),fourier,fourier) - DG₁₁^2

V₂ᴺV₁ᴺ_interval = V₂ᴺ_interval*V₁ᴺ_interval
J₂ = project(Multiplication(V₂ᴺV₁ᴺ_interval),fourier,fourier) - DG₁₁*DG₁₂

J₃ = project(Multiplication(V₂ᴺV₁ᴺ_interval),fourier,fourier) - DG₁₂*DG₁₁

V₂ᴺ²_interval = V₂ᴺ_interval*V₂ᴺ_interval
J₄ = project(Multiplication(V₂ᴺ²_interval),fourier,fourier) - DG₁₂^2

# Let lᵢⱼₙ = min_{n ∈ J_red(D₄)\Iᴺ} |lᵢⱼ(ñ)|. Then,
l₁₁ₙ = ((interval((N+1)π)/di)^2*λ₁i+interval(1))
l₂₂ₙ = ((interval((N+1)π)/di)^2+λ₂i)
l₂₁ₙ = abs(λ₁i*λ₂i-interval(1))

# Computation of Z₁₃. Let M₃ = πᴺ(L⁻¹)⋆DG(U₀)⋆πₙDG(U₀)*L⁻¹πᴺ
# We fully expand M₃ so we can use Lemma 4.1 to estimate its norm.
print("Computing M₃")
M₃₁₁ = L₁₁⁻¹.*J₁.*L₁₁⁻¹' -(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹).*J₃.*L₁₁⁻¹' - L₁₁⁻¹.*J₂.*(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹)' + (L₂₂⁻¹.*L₂₁.*L₁₁⁻¹).*J₄.*(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹)'
norm_M₃₁₁ = opnorm(LinearOperator(coefficients(P.*M₃₁₁.*P⁻¹')),2)
M₃₁₁ = Nothing

M₃₁₂ = L₁₁⁻¹.*J₂.*L₂₂⁻¹' - (L₂₂⁻¹.*L₂₁.*L₁₁⁻¹).*J₄.*L₂₂⁻¹'
norm_M₃₁₂ = opnorm(LinearOperator(coefficients(P.*M₃₁₂.*P⁻¹')),2)
M₃₁₂ = Nothing

M₃₂₁ = L₂₂⁻¹.*J₃.*L₁₁⁻¹' - L₂₂⁻¹.*J₄.*(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹)'
norm_M₃₂₁ = opnorm(LinearOperator(coefficients(P.*M₃₂₁.*P⁻¹')),2)
M₃₂₁ = Nothing

M₃₂₂ = L₂₂⁻¹.*J₄.*L₂₂⁻¹'
norm_M₃₂₂ = opnorm(LinearOperator(coefficients(P.*M₃₂₂.*P⁻¹')),2)
M₃₂₂ = Nothing

Z₁₃ = sqrt(φ(norm_M₃₁₁,norm_M₃₁₂,norm_M₃₂₁,norm_M₃₂₂))
J₂ = Nothing
J₃ = Nothing

# Computation of Z₁₁ where M₄ = πᴺ(I - BM)πᴺ(I - M⋆B⋆)πᴺ and M = I + DG*L⁻¹
# By the structure of the system, we only have M₁₁ and M₁₂ to compute.
print("Computing M₄")
M₁₁ = I + DG₁₁.*L₁₁⁻¹' - DG₁₂.*(L₂₂⁻¹.*L₂₁.*L₁₁⁻¹)'
DG₁₁ = Nothing
M₁₂ = DG₁₂.*L₂₂⁻¹'
DG₁₂ = Nothing
M₁₁_adjoint = LinearOperator(fourier,fourier,coefficients(M₁₁)')
M₁₂_adjoint = LinearOperator(fourier,fourier,coefficients(M₁₂)')
B₁₁_adjoint = LinearOperator(fourier,fourier,coefficients(B₁₁)')
B₁₂_adjoint = LinearOperator(fourier,fourier,coefficients(B₁₂)')

# Let M₄ = I - BM. Again using the structure of the system, the only nonzero block of M₄ is M₄₁₁
M₄₁₁ = (I-B₁₁*M₁₁)*(I-M₁₁_adjoint*B₁₁_adjoint) + (B₁₁*M₁₂+B₁₂)*(M₁₂_adjoint*B₁₁_adjoint+B₁₂_adjoint) 
M₁₁ = Nothing
M₁₁_adjoint = Nothing
M₁₂ = Nothing
M₁₂_adjoint = Nothing
B₁₂ = Nothing
B₁₂_adjoint = Nothing

Z₁₁ = sqrt(opnorm(LinearOperator(coefficients(P.*M₄₁₁.*P⁻¹')),2))
M₄₁₁ = Nothing

# Computation of Z₁₂ where 
# M₁ = πᴺB₁₁DG₁₁(U₀)πₙDG₁₁(U₀)⋆B₁₁⋆πᴺ 
# M₂ = πᴺB₁₁DG₁₂(U₀)πₙDG₁₂(U₀)⋆B₁₁⋆πᴺ

print("Computing M₁")
M₁ = B₁₁*J₁*B₁₁_adjoint
J₁ = Nothing
norm_M₁ = opnorm(LinearOperator(coefficients(P.*M₁.*P⁻¹')),2)
M₁ = Nothing

print("Computing M₂")
M₂ = B₁₁*J₄*B₁₁_adjoint
J₄ = Nothing
norm_M₂ = opnorm(LinearOperator(coefficients(P.*M₂.*P⁻¹')),2)
M₂ = Nothing

Z₁₂ = sqrt((interval(1)/l₁₁ₙ * sqrt(norm_M₁) + (l₂₁ₙ/(l₂₂ₙ*l₁₁ₙ))*sqrt(norm_M₂))^2 + interval(1)/l₂₂ₙ^2*norm_M₂)

# Computation of Z₁₄
print("Computing Z₁₄")
Z₁₄ = sqrt((interval(1)/l₁₁ₙ * norm(V₁ᴺ_interval,1) + (l₂₁ₙ/(l₂₂ₙ*l₁₁ₙ))*norm(V₂ᴺ_interval,1))^2 + interval(1)/l₂₂ₙ^2*norm(V₂ᴺ_interval,1)^2)

Z₁ = φ(Z₁₁,Z₁₂,Z₁₃,Z₁₄)

# This computes the error due to taking N ≠ N₀
inf_error = norm_B₁₁*φ(interval(1),0,abs(λ₁i*λ₂i - 1)/λ₂i,1/λ₂i)*sqrt(norm(V₁_interval - V₁ᴺ_interval,1)^2 + norm(V₂_interval - V₂ᴺ_interval,1)^2)

𝒵₁ = Z₁ + norm_B₁₁*𝒵ᵤ + inf_error

r_min = sup((interval(1) - 𝒵₁ - sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂)
r_max = min(inf((interval(1) - 𝒵₁ + sqrt((interval(1) - 𝒵₁)^2 - interval(2)*𝒴₀*𝒵₂))/𝒵₂), inf((interval(1)-𝒵₁)/𝒵₂))
CAP(𝒴₀,𝒵₁,𝒵₂,r₀)


################################ Proof of Periodic Solution #################################################
# The values of κ̂₂,κ̂₃, and κ̂₃ defined in Theorem 4.11
κ̂₂ = sqrt(interval(1)/(interval(4π)*λ₁i) + interval(1)/(interval(4)*di^2) + interval(1)/(interval(2)*di) * interval(π)/sqrt(λ₁i))
κ̂₃ = sqrt(interval(2))*min(κ̂₂^2/λ₂i,κ̂₂*sqrt(interval(1)/(interval(4π)*λ₂i) + interval(1)/(interval(4)*di^2*λ₂i^2) + interval(1)/(interval(2)*di)*interval(π)/sqrt(λ₂i)))
κ̂₀₁ = sqrt((λ₁i*κ̂₂ + sqrt(interval(1)/(interval(4π)*λ₂i) + interval(1)/(interval(4)*di^2*λ₂i^2) + interval(1)/(interval(2)*di)*interval(π)/sqrt(λ₂i)))^2 + interval(1)/(interval(4π)*λ₂i) + interval(1)/(interval(4)*di^2*λ₂i^2) + interval(1)/(interval(2)*di)*interval(π)/sqrt(λ₂i))
κ̂₀₂ = sqrt(interval(2))*sqrt(interval(1)/(interval(4π)*λ₂i) + interval(1)/(interval(4)*di^2*λ₂i^2) + interval(1)/(interval(2)*di)*interval(π)/sqrt(λ₂i))
κ̂₀ = min(max(κ̂₀₁,κ̂₀₂),κ̂₂*interval(1)/λ₂i * ((interval(1)-λ₁i*λ₂i)^2 + interval(1))^(interval(1/2)))

# We can now perform the computer assisted proof for the branch of periodic solutions
𝒵₁_hat = 𝒵₁+norm_B₁₁*𝒵ᵤ
𝒵₂_hat = interval(2)*(sqrt(φ(𝒵₂₁,𝒵₂₂,𝒵₂₂,𝒵₂₃))*sqrt(κ̂₂^2 + interval(4)*κ̂₀^2)) + interval(3)*norm_B₁₁*κ̂₃*r₀
r̂_min = sup((interval(1) - 𝒵₁_hat - sqrt((interval(1) - 𝒵₁_hat)^2 - interval(2)*𝒴₀*𝒵₂_hat))/𝒵₂_hat)
r̂_max = min(inf((interval(1) - 𝒵₁_hat + sqrt((interval(1) - 𝒵₁_hat)^2 - interval(2)*𝒴₀*𝒵₂_hat))/𝒵₂_hat), inf((interval(1)-𝒵₁_hat)/𝒵₂_hat))
CAP(𝒴₀,𝒵₁_hat,𝒵₂_hat,r₀)
