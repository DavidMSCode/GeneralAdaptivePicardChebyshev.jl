using LinearAlgebra

"""
	interpolate(τs::AbstractVector{<:Real}, N::Integer; recursive::Bool = false)

Computes a matrix of Chebyshev polynomials of the first kind ``T_n(τ)`` for 
``τ ∈ τs`` and ``n = 0,1,...,N``. Used with Chebyshev coefficients to 
calculate the value of the interpolating polynomials at the given values of 
``τ``.

# Arguments
- `τs::AbstractVector{<:Real}`: The points at which to evaluate the Chebyshev
 polynomials.
- `N::Integer`: The polynomial degree.
- `recursive::Bool`: If true, use the recursive formula to compute the
Chebyshev polynomials within the domain [-1,1]. If false, use the
trigonometric formulation. Default is false.

# Returns
- `Ts`: A matrix of Chebyshev polynomial values at the given values of τ.
"""
function interpolate(τs::AbstractVector{<:Real}, N::Integer; recursive::Bool = false)
	#find the indeces of the τs that are outside the domain
	idx_outside = findall(x -> abs(x) > 1, τs)
	if !isempty(idx_outside)
		@warn ("The input values at the following indices are outside the
		domain [-1,1]: $idx_outside. Extrapolated values may be innacurate.")
	end

	#Get the uneweighted Chebyshev polynomial values at each τ
	js = 0:N
	if !recursive
		#Default behaviour: Use the trig formulation and only use the recursive
		#formula if τ is outside the domain [-1,1]
		Ts = [abs(τ) <= 1 ? trig_chebyshev(τ, j) :
			  recursive_chebyshev(τ, j) for τ in τs, j in js]
	else
		#always use the recursive formula if recursive is true
		Ts = [recursive_chebyshev(τ, j) for τ in τs, j in js]
	end
	return Ts
end

@doc (@doc interpolate)
function interpolate(τ::Real, N::Integer; recursive = false)
	return interpolate([τ], N; recursive = recursive)
end

"""
	trig_chebyshev(τ::Real, N::Integer)

Computes the Chebyshev polynomial of the first kind using the trigonometric
form ``T_N(τ) = cos(N * acos(τ))`` for some ``τ ∈ [-1,1]`` and ``N = 0,1,...``.

# Arguments
- `τ::Real`: The point at which to evaluate the Chebyshev polynomial.
- `N::Integer`: The polynomial degree.

# Returns
- `T`: The Chebyshev polynomial value at the given point.
"""
function trig_chebyshev(τ::Real, N::Integer)
	return cos(N * acos(τ))
end

"""
	recursive_chebyshev(τ::Real, N::Integer)

Computes the Chebyshev polynomial of the first kind using the recursive formula
``T_N(τ) = 2 * τ * T_{N-1}(τ) - T_{N-2}(τ)`` for some ``τ ∈ ℝ`` and
``N = 0,1,...``.

# Arguments
- `τ::Real`: The point at which to evaluate the Chebyshev polynomial.
- `N::Integer`: The polynomial degree.

# Returns
- `T`: The Chebyshev polynomial value at the given point.
"""
function recursive_chebyshev(τ::Real, N::Integer)
	if N == 0
		return 1
	elseif N == 1
		return τ
	else
		return 2 * τ * recursive_chebyshev(τ, N - 1) - recursive_chebyshev(τ, N - 2)
	end
end

"""
	chebyshev(N::Integer; M::Integer=N)

Computes the value of the N+1 Chebyshev polynomials at the M+1 points cosine
spaced nodes on the interval [-1,1].

# Arguments
- `N::Integer`: The polynomial degree.
- `M::Integer`: The sampling degree. Must be greater than or equal to the
polynomial degree. This is equal to the total number of function sampling
points minus 1. Defaults to `N`.

# Returns
- `Ts`: The Chebyshev polynomial values at the M+1 points.

# Description
This function computes the value of the N+1 Chebyshev polynomials at the M+1
cosine spaced nodes. It first generates the cosine spaced sample points from
[-1,1] using the `cosineSamplePoints` function. Then, it computes the
unweighted Chebyshev polynomial values at each τ.
"""
function chebyshev(N::Integer, M::Integer=N)
	# Get the uneweighted Chebyshev polynomials at each of the M+1 points
	τs = cosineSamplePoints(M)
	Ts = interpolate(τs, N)
	return Ts
end

"""
	cosineSamplePoints(M::Integer)

Computes `M+1` cosine spaced sample points from [-1,1].

# Returns
- `τs`: The cosine spaced sample points from [-1,1].
"""
function cosineSamplePoints(M::Integer)
	# Cosine Sample points (M+1 points) from [-1,1]
	τs = 0:M
	return -cos.(pi * τs / M)
end

"""
	lsq_chebyshev_fit(N::Integer, M::Integer=N)

Compute the Chebyshev polynomials and the Least Squares Operator matrix.

# Arguments
- `N::Integer`: The polynomial degree.
- `M::Integer`: The sampling degree. Must be greater than or equal to the
polynomial degree. This is equal to the total number of function sampling
points minus 1. Defaults to `N`.

# Returns
- `T`: The Chebyshev polynomial values at the M+1 points.
- `A`: The Least Squares Operator matrix.

# Description
This function computes the Chebyshev polynomials at the M+1 points and the
Least Squares Operator matrix. It first generates the Chebyshev polynomials
using the `chebyshev` function. Then, it constructs the weights matrix, W,
and the V matrix. Finally, it computes the Least Squares Operator matrix, A.

# Example
```julia
N = 5
M = 5
T, A = lsq_chebyshev_fit(N, M)
```
"""
function lsq_chebyshev_fit(N::Integer, M::Integer=N)
	#generate the Chebyshev polynomials at the nodes
	T = chebyshev(N, M)

	#weights matrix
	wdiag = [0.5; ones(M - 1); 0.5]
	W = Diagonal(wdiag)

	#V matrix
	vdiag = [1 / M; 2 .* ones(N) ./ M]
	if M == N
		vdiag[end] = 1 / M
	end
	V = Diagonal(vdiag)

	#transpose T
	Tt = transpose(T)

	#Least Squares Operator
	A = V * Tt * W

	return T, A
end

"""
	clenshaw_curtis_nested_ivpd(d::Integer, N::Integer, M::Integer=N)

Compute the Clenshaw-Curtis quadrature and Chebyshev basis function matrices
for an Nth degree polynomial and for a `d`-th order integral on the interval [-1,1].

# Arguments
- `d::Integer`: The integral order.
- `N::Integer`: The polynomial degree.
- `M::Integer`: The sampling degree. Must be greater than or equal to the
polynomial degree. The integrand is evaluated at `M+1` cosine spaced nodes.
Defaults to `N`.


# Returns
- `A`: The Least Squares Operator matrix.
- `P`: The Quadrature Matrix.
- `T`: The Chebyshev Matrix.
"""
function clenshaw_curtis_nested_ivpd(d::Integer, N::Integer, M::Integer=N-d)
	if M < N-d
		throw(ArgumentError("The number of sampling nodes must be greater than
		the polynomial order, N."))
	end
	if d > N
		throw(ArgumentError("The polynomial order N must be greater than or
		equal to the integral order d."))
	end

	# Least Squares Operator for "acceleration"
	Ta, A = lsq_chebyshev_fit(N - d, M)

	# Constants of Integration
	ks = 0:N
	Lrow = cos.(ks * pi)
	L = vcat(Lrow', zeros(N, N + 1))

	#S matrix
	tempdiag = [1; [1 / (2 * i) for i in 1:N]]
	temp = Diagonal(tempdiag)
	temp2 = diagm(N + 1, N, -1 => ones(N), 1 => [0; -ones(N - 2)])
	S = temp * temp2
	S[1, 1] = 0.25
	S[2, 1] = 1.0

	#Integration Operator
	temp3 = -L + Diagonal(ones(N + 1))
	P = temp3 * S

	# Chebyshev Matrix
	T = chebyshev(N, M)

	return A, P, T
	return
end

"""
	clenshaw_curtis_ivpii(N::Integer, M::Integer=N)

Compute the Clenshaw-Curtis quadrature and Cebyshev basis function matrices for
a second order initial value problem on the interval [-1,1].

# Arguments
- `N::Integer`: The polynomial degree.
- `M::Integer`: The sampling degree. Must bes greater than or equal to the
polynomial degree. This is equal to the total number of function sampling
points minus 1. Defaults to `N`.

# Returns
- `A`: The Least Squares Operator matrix.
- `Ta`: The "acceleration" Chebyshev Matrix.
- `P1`: The Quadrature Matrix for acceleration to velocity.
- `T1`: The "Velocity" Chebyshev Matrix.
- `P2`: The Quadrature Matrix for velocity to position.
- `T2`: The "Position" Chebyshev Matrix.

# Example
```julia
N = 5
M = 5
A, Ta, P1, T1, P2, T2 = clenshaw_curtis_ivpii(N, M)
```

"""
function clenshaw_curtis_ivpii(N::Integer, M::Integer=N)
	# if M < N
	# 	throw(ArgumentError("The number of sampling nodes must be greater than
	# 	the polynomial order, N."))
	# end

	# Least Squares Operator for "acceleration"
	Ta, A = lsq_chebyshev_fit(N - 2, M)

	# "Position" Constants of Integration
	ks = 0:N
	Lprow = cos.(ks * pi)
	Lp = vcat(Lprow', zeros(N, N + 1))

	#S matrix for "position"
	temp4diag = [1; [1 / (2 * i) for i in 1:N]]
	temp4 = Diagonal(temp4diag)
	temp5 = diagm(N + 1, N, -1 => ones(N), 1 => [0; -ones(N - 2)])
	Sp = temp4 * temp5
	Sp[1, 1] = 0.25
	Sp[2, 1] = 1.0

	#Picard Integration Operator for velocity to position
	temp6 = -Lp + Diagonal(ones(N + 1))
	P2 = temp6 * Sp

	# "Velocity" Quadrature matrix is subset of the "Position" Quadrature matrix
	P1 = P2[1:end-1, 1:end-1]

	# "Position" Chebyshev Matrix
	T2 = chebyshev(N, M)
	# "Velocity" Chebyshev Matrix is subset of the "Position" Chebyshev Matrix
	T1 = T2[:, 1:end-1]

	return A, Ta, P1, T1, P2, T2
end

"""
	clenshaw_curtis_ivpi(N::Integer, M::Integer=N)

Compute the Clenshaw-Curtis quadrature and Cebyshev basis function matrices for
a first order initial value problem on the interval [-1,1].

# Arguments
- `N::Integer`: The polynomial degree.
- `M::Integer`: The sampling degree. Must bes greater than or equal to the
polynomial degree. This is equal to the total number of function sampling
points minus 1. Defaults to `N`.

# Returns
- `A`: The Least Squares Operator matrix.
- `Ta`: The "acceleration" Chebyshev Matrix.
- `P1`: The Quadrature Matrix for acceleration to velocity.
- `T1`: The "Velocity" Chebyshev Matrix.
"""
function clenshaw_curtis_ivpi(N::Integer, M::Integer=N)
	# if M < N
	# 	throw(ArgumentError("The number of sampling nodes, M, must be greater
	# 	than or equal to the polynomial order, N."))
	# end

	# Least Squares Operator for "acceleration"
	Ta, A = lsq_chebyshev_fit(N - 1, M)

	# "Velocity" Constants of Integration
	ks = 0:N
	Lvrow = cos.(ks * pi)
	Lv = vcat(Lvrow', zeros(N, N + 1))

	#S matrix for "velocity"
	temp1diag = [1; [1 / (2 * i) for i in 1:N]]
	temp1 = Diagonal(temp1diag)
	temp2 = diagm(N + 1, N, -1 => ones(N), 1 => [0; -ones(N - 2)])
	Sv = temp1 * temp2
	Sv[1, 1] = 0.25
	Sv[2, 1] = 1.0

	#Picard Integration Operator for accleration to velocity
	temp3 = -Lv + Diagonal(ones(N + 1))
	P1 = temp3 * Sv

	# "Velocity" Chebyshev Matrix
	T1 = chebyshev(N, M)

	return A, Ta, P1, T1
end

