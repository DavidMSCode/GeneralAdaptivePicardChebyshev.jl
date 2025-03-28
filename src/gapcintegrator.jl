
mutable struct Stats
	feval::Int
	accepted::Int
	rejected::Int
end

function Base.show(io::IO, stats::Stats)
	println("Accepted segments      : ", rpad(stats.accepted, 10))
	println("Rejected segments      : ", rpad(stats.rejected, 10))
	println("Function evaluations   : ", rpad(stats.feval, 10))
end

function constant_accel_guess(ts, y0, dy0, ddy0, params)
	#return parabolic trajectory
	t0 = ts[1]
	dims = length(y0)
	ys = zeros(length(ts), dims)
	dys = zeros(length(ts), dims)
	for i in 1:dims
		ys[:, i] .= y0[i] .+ dy0[i] * (ts .- t0) .+ 0.5 * ddy0[i] * (ts .- t0) .^ 2
		dys[:, i] .= dy0[i] .+ ddy0[i] * (ts .- t0)
	end
	return ys, dys
end

function zero_accel_guess(ts, y0, dy0, ddy0, params)
	#guess a solution with zero acceleration but constant velocity
	t0 = ts[1]
	dims = length(y0)
	ys = zeros(length(ts), dims)
	dys = zeros(length(ts), dims)
	for i in 1:dims
		ys[:, i] .= y0[i] .+ dy0[i] * (ts .- t0)
		dys[:, i] .= dy0[i]
	end
	return ys, dys
end

function stationary_guess(ts, y0, dy0, ddy0, params)
	t0 = ts[1]
	dims = length(y0)
	ys = zeros(length(ts), dims)
	dys = zeros(length(ts), dims)
	for i in 1:dims
		ys[:, i] .= y0[i]
		dys[:, i] .= dy0[i]
	end
	return ys, dys
end

function initial_time_step(y0, dy0, t0, order, ode, params, tol)
	#from Gauss-Radau code: Moving Planets Around chapter 8
	#integration order
	p = order
	#eval right hand side
	f0 = ode(t0, y0, dy0, params)
	d0 = maximum(abs.(y0))
	d1 = maximum(abs.(f0))

	if (d0 < 1e-5) || (d1 < 1e-5)
		dt0 = 1e-6
	else
		dt0 = 0.01 * (d0 / d1)
	end # if

	# perform 1 step of Euler
	y1 = y0 + dt0 * dy0
	dy1 = dy0 + dt0 * f0
	#call function
	f1 = ode(t0 + dt0, y1, dy1, params)
	d2 = maximum(abs.(f1 - f0)) / dt0

	if maximum([d1, d2]) <= 1e-15
		dt1 = maximum([1e-6, dt0 * 1e-3])
	else
		dt1 = (0.01 / maximum([d1, d2]))^(1.0 / (p / 2 + 1))
	end # if

	dt = minimum([100 * dt0, dt1]) * 3
	return dt, f0
end # function


function step(y0, dy0, ddy0, gamma, beta, alpha, dt, t, tf, N, M, A, Ta, P1, T1, P2, T2, tol, exponent, fac, iseg, ode, params, verbose, maxIters, itol, analytic_guess, apcstats)
	#Hardcoded max segment iterations (with different dts)
	maxSegIters = 40
	#initialize the chebyshev nodes and solution vectors
	Ms = 0:M
	#tau time [-1,1] nodes that chebyshev functions are valid with cosine spacing
	taus = -cos.(π * (Ms ./ M))
	new_a = gamma * 0
	ys = gamma * 0
	dys = gamma * 0
	ys[1, :] = y0
	dys[1, :] = dy0
	times = zeros(N + 1)

	#calculate the time transformation for first guess
	w1 = (2 * t + dt) / 2#time average (value at tau=0) 
	w2 = (dt) / 2#time scaling factor	(tf-t0)/2
	times = w1 .+ w2 * taus#real times at chebyshev nodes

	#if user specified analytic_guess use it for initial trajectory,
	#otherwise use constant initial conditions
	ys, dys = analytic_guess(times, y0, dy0, ddy0, params)
	#Initialize the picard iteration loop
	istat = 0
	segItr = 0
	while istat == 0 && segItr < maxSegIters
		if verbose #print segment information
			println("Segment: ", iseg, " Time: ", t, " dt: ", dt)
		end
		#begin picard iteration
		ierr = 1
		itr = 0
		old_gammas = zeros(size(A)[1], size(y0)[1])
		new_a = ys * 0.0
		new_a[1, :] = ddy0 #initial acceleration (should not change in the iteration)
		#keep history of ierr in 10 length empty array
		ierrors = fill(NaN, 10)
		while ierr > itol && itr < maxIters
			#calculate the new acceleration along the entire trajectory
			new_a[2:end, :] = stack([ode(state..., params) for state in zip(times[2:end], eachrow(ys[2:end, :]), eachrow(dys[2:end, :]))], dims = 1)
			apcstats.feval += (M - 1)
			#calculate the least squares coefficients for the acceleration
			#polynomial
			gamma = A * new_a
			#calculate velocity coefficients (and multipley time scale to get
			#units of real time)
			beta = w2 * P1 * gamma
			beta[1, :] += dy0 #add initial velocity
			new_dys = T1[2:end, :] * beta #integrate new velocity at the chebyshev nodes (>1)
			#calculate the position coefficients (and multiply by time scale to
			#get units of real time)
			alpha = w2 * P2 * beta
			alpha[1, :] += y0#add initial position
			new_ys = T2[2:end, :] * alpha#calculate the new positions at the chebyshev nodes
			Y0 = 0*alpha
			Y0[1, :] = y0
			#estimate the convergence by checking if the last three coefficients
			#of the acceleration polynomials are less than the iteration
			#tolerance (scaled by the max acceleration over the trajectory)

			#difference in last coefficients between iterations
			da = gamma[:, :] - old_gammas[:, :]
			ierr = (maximum(abs.(da)) / maximum(abs.(new_a)))

			#update the solution vectors for nodes > 1
			ys[2:end, :], dys[2:end, :] = new_ys, new_dys
			#store old acceleration coefficients
			old_gammas = gamma
			itr += 1
			if verbose
				println("\t\tIteration: ", itr, " Convergence Error: ", ierr)
			end
			ierrors[mod(itr, 10)+1] = ierr
			#break if ierr has repeated values in last 10 iterations
			if count(x -> x == ierr, ierrors) > 2
				if verbose
					println("\t\tError has repeated values, breaking iteration")
				end
				break
			end
		end

		#compute global error estimate for the segment

		estim_a_end = maximum(abs.(gamma[end, :])) / maximum(abs.(new_a))

		err = (estim_a_end / tol)^(exponent)

		#calculate next step size with safety factor
		dtreq = dt / err * 0.9

		if err <= 1
			#accept the solution
			#set flag to exit while loop
			istat = 1
			#update the time by current dt
			t += dt
			#update the statistics
			apcstats.accepted += 1
			#check if we have reached the final time within one epsilon. This
			#prevents up to two extra segments from being calculated
			#unecessarily.
			if nextfloat(t) >= tf
				#set flag to 2 for reached final time
				t = nextfloat(t)
				istat = 2
			end
		else
			if verbose
				println("\t Error too large (", err, ") retrying with smaller timestep")
			end
			apcstats.rejected += 1
		end

		#next iteration timestep
		if dtreq / dt > 1 / fac
			dt = dt / fac
		elseif dtreq < 1e-10
			dt = dt * fac
		else
			dt = dtreq
		end

		if (t + dt) > tf
			dt = tf - t
		end
		if dt < eps(t + dt)
			dt = 1e-13
		end

		if istat == 0
			if verbose
				println("\t\tReinterpolating solution for new guess with smaller timestep")
			end
			#use current solution for initial guess but interpolated to new
			#nodes due to shorter segment length
			w1_new = (2 * t + dt) / 2#new time average (value at tau=0) 
			w2_new = (dt) / 2#new time scaling factor	(tf-t0)/2
			times = w1_new .+ w2_new * taus#new real times chebyshev nodes

			#convert to taus of old solution
			taus_old = (times .- w1) ./ w2
			#interpolate the old solution to the new nodes
			Tx = interpolate(taus_old, N, supress_warning = true)
			Tv = Tx[:, 1:end-1]
			ys[2:end, :] = Tx[2:end, :] * alpha
			dys[2:end, :] = Tv[2:end, :] * beta

			w1 = w1_new
			w2 = w2_new
		end
		segItr += 1
	end

	if istat == 0
		#set flag to -1 for reached max segment iterations without accepted solution
		istat = -1
	end

	return ys, dys, new_a, times, dt, gamma, alpha, beta, istat, apcstats
end



"""
	integrate(y0, dy0, t0, tf, tol, ode, params; N = 32, verbose = false, dt = nothing, maxIters = 20, itol = 1e-15, exponent = nothing, analytic_guess = nothing)

Integrates a system of ordinary differential equations (ODEs) using a Chebyshev polynomial-based method.

# Arguments
- `y0::Vector{Float64}`: Initial state vector.
- `dy0::Vector{Float64}`: Initial derivative of the state vector.
- `t0::Float64`: Initial time.
- `tf::Float64`: Final time.
- `tol::Float64`: Tolerance for the integration.
- `ode::Function`: Function representing the ODE system.
- `params::Any`: Additional parameters for the ODE function.
- `N::Int`: Number of Chebyshev nodes (default: 20).
- `verbose::Bool`: Flag for verbose output (default: false).
- `dt::Union{Nothing, Float64}`: Initial time step (default: nothing).
- `maxIters::Int`: Maximum number of iterations for the Picard iteration (default: 20).
- `itol::Float64`: Iteration tolerance (default: 1e-15).
- `exponent::Union{Nothing, Float64}`: Exponent for global error estimation (default: nothing).
- `analytic_guess::Union{Nothing, Function}`: Function for the initial analytic guess (default: nothing).

# Returns
- `sol_time::Vector{Float64}`: Vector of time points.
- `sol_pos::Matrix{Float64}`: Matrix of state vectors at each time point.
- `sol_vel::Matrix{Float64}`: Matrix of state derivatives at each time point.
- `sol_acc::Matrix{Float64}`: Matrix of state second derivatives at each time point.
"""
function integrate_ivp2(y0, dy0, t0, tf, tol, ode, params; N = 20, verbose = false, dt = nothing, maxIters = 20, itol = 1e-15, exponent = nothing, analytic_guess = nothing, times=nothing)
	#Number of nodes to sample solution at. Since this is a 2nd order
	#integrator, the acceleration polynomial is of order N-2 this means the 
	M = N - 2

	#maximum timestep change factor
	fac = 0.5

	#initialize the timestep with logic for user specified timestep and timescale exponent
	if isnothing(dt)
		dt, ddy0 = initial_time_step(y0, dy0, t0, N, ode, params, tol)
		if isnothing(exponent)
			#timescale exponent for global error estimation
			exp = 1 / N
		else
			exp = exponent #user specified exponent
		end
	else
		if !isnothing(exponent)
			println("Warning: user provided exponent ignored when fixed timestep dt is specified")
		end
		exp = 0
		ddy0 = ode(t0, y0, dy0, params)
	end

	#set the initial analytic guess function if provided. Otherwise use constant
	#acceleration guess
	if isnothing(analytic_guess)
		analytic_guess = constant_accel_guess
	end

	#statistics
	apcstats = Stats(0, 0, 0)

	#initialize solution at all nodes
	nstore = 10000
	cstore = 500
	dim = length(y0)

	sol_y = zeros(nstore, dim)
	sol_dy = zeros(nstore, dim)
	sol_ddy = zeros(nstore, dim)
	sol_time = zeros(nstore)

	sol_y[1, :] = y0
	sol_dy[1, :] = dy0
	sol_ddy[1, :] = ddy0
	sol_time[1] = t0

	#initialize polynomial coefficient storage
	ddy_coeffs = zeros(cstore, M + 1, dim)
	dy_coeffs = zeros(cstore, M + 2, dim)
	y_coeffs = zeros(cstore, M + 3, dim)
	segment_times = zeros(cstore)

	segment_times[1] = t0

	#pre compute the quadrature and chebyshev matrices
	A, Ta, P1, T1, P2, T2 = clenshaw_curtis_ivpii(N, M)

	#initialize the chebyshev coefficients
	gammas = zeros(M + 1, dim)
	betas = zeros(M + 2, dim)
	alphas = zeros(M + 3, dim)
	t = t0

	if verbose
		println("Initial timestep=", dt)
	end

	integrate_ivp2 = true
	iseg = 0
	istat = nothing
	while integrate_ivp2
		#update segment count
		iseg += 1
		#advance solution  by 1 segment
		ys, dys, ddys, ts, dt, gammas, alphas, betas, istat, apcstats = step(y0, dy0, ddy0, gammas, betas, alphas, dt, t, tf, N, M, A, Ta, P1, T1, P2, T2, tol, exp, fac, iseg, ode, params, verbose, maxIters, itol, analytic_guess, apcstats)
		if istat == -1
			#check if the iteration failed to converge
			println("Warning: Picard iteration failed to converge on segment ", iseg)
			println("Returning partial solution from t=", t0, " to t=", t)
			integrate_ivp2 = false
		end
		t = ts[end]
		if istat == 2
			integrate_ivp2 = false
		end

		y0 = ys[end, :]
		dy0 = dys[end, :]
		ddy0 = ddys[end, :]

		#check if solution vectors need to be expanded
		if iseg * N + 1 > size(sol_y, 1)
			sol_y = vcat(sol_y, zeros(nstore, dim))
			sol_dy = vcat(sol_dy, zeros(nstore, dim))
			sol_ddy = vcat(sol_ddy, zeros(nstore, dim))
			sol_time = vcat(sol_time, zeros(nstore))
		end

		#store the solution
		range = (iseg-1)*M+2:(iseg)*M+1
		sol_y[range, :] = ys[2:end, :]
		sol_dy[range, :] = dys[2:end, :]
		sol_ddy[range, :] = ddys[2:end, :]
		sol_time[range] = ts[2:end]

		#check if the polynomial coefficient storage needs to be expanded
		if iseg+1 > size(ddy_coeffs, 1)
			ddy_coeffs = vcat(ddy_coeffs, zeros(cstore, M + 1, dim))
			dy_coeffs = vcat(dy_coeffs, zeros(cstore, M + 2, dim))
			y_coeffs = vcat(y_coeffs, zeros(cstore, M + 3, dim))
			segment_times = vcat(segment_times, zeros(cstore))
		end

		#store the polynomial coefficients
		ddy_coeffs[iseg, :, :] = gammas
		dy_coeffs[iseg, :, :] = betas
		y_coeffs[iseg, :, :] = alphas
		segment_times[iseg+1] = t
	end #while

	if isnothing(times)
		#return the solution at chebyshev nodes
		t_out = sol_time[1:iseg*M+1]
		y_out = sol_y[1:iseg*M+1, :]
		dy_out = sol_dy[1:iseg*M+1, :]
		ddy_out = sol_ddy[1:iseg*M+1, :]
	else
		#interpolate is a vector of times to interpolate the solution to
		t_out,y_out,dy_out,ddy_out = interpolate_solution(times, segment_times[1:iseg+1], y_coeffs[1:iseg,:,:], dy_coeffs[1:iseg,:,:], ddy_coeffs[1:iseg,:,:])
	end

	return t_out,y_out,dy_out,ddy_out, istat, apcstats
end

function integrate_ivp1(dy0, t0, tf, tol, ode, params; N = 20, verbose = false, dt = nothing, maxIters = 20, itol = 1e-15, exponent = nothing, analytic_guess = nothing, times=nothing)
	dummy_y0 = 0 * dy0

	#convert first order oder
	function second_order_ode(t, dummy_y, dy, params)
		return ode(t, dy, params)
	end

	t, y, dy, ddy, istat, apcstats = integrate_ivp2(dummy_y0, dy0, t0, tf, tol, second_order_ode, params, N = N, verbose = verbose, dt = dt, maxIters = maxIters, itol = itol, exponent = exponent, analytic_guess = analytic_guess,times=times)
	return t, dy, ddy, istat, apcstats
end

function interpolate_solution(times::AbstractArray, segment_times, y_coeffs, dy_coeffs, ddy_coeffs)
	debug=false
	"""This function takes a list of times and interpolates the solution to those times."""
	num_segs = length(segment_times)-1 #total number of solution segments
	N = size(y_coeffs, 2) - 1 #order of the y polynomial
	dim = size(y_coeffs, 3) #dimension of the system
	y_out = zeros(length(times), dim)
	dy_out = zeros(length(times), dim)
	ddy_out = zeros(length(times), dim)
	t_out = zeros(length(times), 1)
	#make a list of the times that fall within this segment

	for i in 1:num_segs
		ind=[]
		if times[end] < segment_times[i]
			#no need to check further
			break
		end
		t0 = segment_times[i]
		t1 = segment_times[i+1]
		#calculate the time transformation for first guess
		w1 = (2 * t0 + (t1-t0)) / 2#time average (value at tau=0) 
		w2 = (t1-t0) / 2#time scaling factor	(tf-t0)/2
		ind = findall((times .>= t0) .& (times .<= t1))

		if debug
			println("Segment: ", i, " Times: ", t0, " to ", t1, " Indices: ", ind)
		end

		if length(ind) > 0
			#interpolate the solution to these times
			taus = (times[ind].-w1) ./ w2
			#print warning if any taus are outside of [-1,1]
			if any(taus .< -1) || any(taus .> 1)
				println("Warning: Some taus are outside of [-1,1]")
			end
			Ty = interpolate(taus, N)
			Tdy = Ty[:, 1:end-1]
			Tddy = Tdy[:, 1:end-1]
			y_out[ind,:] = Ty * y_coeffs[i, :, :]
			dy_out[ind,:] = Tdy * dy_coeffs[i, :, :]
			ddy_out[ind,:] = Tddy * ddy_coeffs[i, :, :]
			t_out[ind] = times[ind]
		end
	end
	return times, y_out, dy_out, ddy_out
end