function zero_accel_guess(ts, y0, dy0, params)
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

function stationary_guess(ts, y0, dy0, params)
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

function initial_time_step(y0, dy0, t0, order, ode, params)
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
		dt1 = (0.01 / maximum([d1, d2]))^(1.0 / (p/2 + 1))
	end # if

	dt = minimum([100 * dt0, dt1])
	return dt, f0
end # function


function step(y0, dy0, ddy0, gammas, betas, alphas, dt, t, tf, N, M, A, Ta, P1, T1, P2, T2, tol, exponent, fac, iseg, ode, params, verbose, maxIters, itol, analytic_guess,feval)
	#Hardcoded max segment iterations (with different dts)
	maxSegIters = 40

	#initialize the chebyshev nodes and solution vectors
	Ms = 0:M
	#tau time [-1,1] nodes that chebyshev functions are valid with cosine spacing
	taus = -cos.(π * (Ms ./ M))
	new_a = gammas * 0
	ys = gammas * 0
	dys = gammas * 0
	times = zeros(N + 1)

	#Initialize the picard iteration loop
	istat = 0
	segItr = 0
	while istat == 0 && segItr < maxSegIters
		if verbose #print segment information
			println("Segment: ", iseg, " Time: ", t, " dt: ", dt)
		end
		#calculate the time transformation
		w1 = (2 * t + dt) / 2#time average (value at tau=0) 
		w2 = (dt) / 2#time scaling factor	(tf-t0)/2
		times = w1 .+ w2 * taus#real times at chebyshev nodes
		#if user specified analytic_guess use it for initial trajectory,
		#otherwise use constant initial conditions
		ys, dys = analytic_guess(times, y0, dy0, params)

		#begin picard iteration
		ierr = 1
		itr = 0
		old_gammas = zeros(size(A)[1], size(y0)[1])
		new_a = ys*0.0
		new_a[1, :] = ddy0 #initial acceleration (should not change in the iteration)
		while ierr > itol && itr < maxIters
			#calculate the new guess along the entire trajectory
			new_a[2:end,:] = stack([ode(state..., params) for state in zip(times[2:end], eachrow(ys[2:end,:]), eachrow(dys[2:end,:]))], dims = 1)
			feval = feval + (M-1)
			#calculate the least squares coefficients for the acceleration
			#polynomial
			gammas = A * new_a
			#calculate velocity coefficients (and multipley time scale to get
			#units of real time)
			beta = w2 * P1 * gammas
			beta[1, :] += dy0 #add initial velocity
			new_dys = T1[2:end,:] * beta #integrate new velocity at the chebyshev nodes (>1)
			#calculate the position coefficients (and multiply by time scale to
			#get units of real time)
			alpha = w2 * P2 * beta
			alpha[1, :] += y0#add initial position
			new_ys = T2[2:end,:] * alpha#calculate the new positions at the chebyshev nodes

			#estimate the convergence by checking if the last three coefficients
			#of the acceleration polynomials are less than the iteration
			#tolerance (scaled by the max acceleration over the trajectory)

			#difference in last coefficients between iterations
			da = gammas[:, :] - old_gammas[:, :] 
			ierr = (maximum(abs.(da)) / maximum(abs.(new_a)))

			#update the solution vectors for nodes > 1
			ys[2:end,:], dys[2:end,:] = new_ys, new_dys
			#store old acceleration coefficients
			old_gammas = gammas
			itr += 1
			if verbose
				println("\t\tIteration: ", itr, " Convergence Error: ", ierr)
			end
		end

		#compute global error estimate for the segment
		estim_a_end = maximum(abs.(gammas[end, :])) / maximum(abs.(new_a))
		err = (estim_a_end / tol)^(exponent)

		#calculate next step size
		dtreq = dt / err

		if err <= 1
			#accept the solution
			#set flag to exit while loop
			istat = 1
			#update the time by current dt
			t += dt
			#check if we have reached the final time within one epsilon. This
			#prevents up to two extra segments from being calculated
			#unecessarily.
			if nextfloat(t) >= tf
				#set flag to 2 for reached final time
				t = nextfloat(t)
				istat = 2
			end
		elseif verbose
			println("\t Error too large (", err, ") retrying with smaller timestep")
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

		segItr += 1
	end

	if istat ==0
		#set flag to -1 for reached max segment iterations without accepted solution
		istat = -1
	end

	return ys, dys, new_a, times, dt, gammas, alphas, betas, istat, feval
end



"""
    integrate(y0, dy0, t0, tf, tol, ode, params; N = 32, verbose = false, dt = nothing, maxIters = 20, itol = 1e-19, exponent = nothing, analytic_guess = nothing)

Integrates a system of ordinary differential equations (ODEs) using a Chebyshev polynomial-based method.

# Arguments
- `y0::Vector{Float64}`: Initial state vector.
- `dy0::Vector{Float64}`: Initial derivative of the state vector.
- `t0::Float64`: Initial time.
- `tf::Float64`: Final time.
- `tol::Float64`: Tolerance for the integration.
- `ode::Function`: Function representing the ODE system.
- `params::Any`: Additional parameters for the ODE function.
- `N::Int`: Number of Chebyshev nodes (default: 32).
- `verbose::Bool`: Flag for verbose output (default: false).
- `dt::Union{Nothing, Float64}`: Initial time step (default: nothing).
- `maxIters::Int`: Maximum number of iterations for the Picard iteration (default: 20).
- `itol::Float64`: Iteration tolerance (default: 1e-19).
- `exponent::Union{Nothing, Float64}`: Exponent for global error estimation (default: nothing).
- `analytic_guess::Union{Nothing, Function}`: Function for the initial analytic guess (default: nothing).

# Returns
- `sol_time::Vector{Float64}`: Vector of time points.
- `sol_pos::Matrix{Float64}`: Matrix of state vectors at each time point.
- `sol_vel::Matrix{Float64}`: Matrix of state derivatives at each time point.
- `sol_acc::Matrix{Float64}`: Matrix of state second derivatives at each time point.
"""
function integrate_ivp2(y0, dy0, t0, tf, tol, ode, params; N = 32, verbose = false, dt = nothing, maxIters = 20, itol = 1e-19, exponent = nothing, analytic_guess = nothing)
	#Number of nodes to sample solution at. Since this is a 2nd order
	#integrator, the acceleration polynomial is of order N-2 this means the 
	M = N-2

	#maximum timestep change factor
	fac = 0.25

	#initialize the timestep with logic for user specified timestep and timescale exponent
	if isnothing(dt)
		dt, ddy0 = initial_time_step(y0, dy0, t0, N, ode, params)
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

	#set the initial analytic guess function if provided. Otherwise use zero
	#acceleration guess
	if isnothing(analytic_guess)
		analytic_guess = zero_accel_guess
	end

	#statistics
	feval = 0

	#initialize solution at all nodes
	nstore = 10000
	dim = length(y0)

	sol_pos = zeros(nstore, dim)
	sol_vel = zeros(nstore, dim)
	sol_acc = zeros(nstore, dim)
	sol_time = zeros(nstore)

	sol_pos[1, :] = y0
	sol_vel[1, :] = dy0
	sol_acc[1, :] = ddy0
	sol_time[1] = t0

	#pre compute the quadrature and chebyshev matrices
	A, Ta, P1, T1, P2, T2 = clenshaw_curtis_ivpii(N,M)

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
		ys, dys, ddys, ts, dt, gammas, alphas, betas, istat,feval = step(y0, dy0, ddy0, gammas, betas, alphas, dt, t, tf, N, M, A, Ta, P1, T1, P2, T2, tol, exp, fac, iseg, ode, params, verbose, maxIters, itol, analytic_guess,feval)
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
		if iseg * N + 1 > size(sol_pos, 1)
			sol_pos = vcat(sol_pos, zeros(nstore, dim))
			sol_vel = vcat(sol_vel, zeros(nstore, dim))
			sol_acc = vcat(sol_acc, zeros(nstore, dim))
			sol_time = vcat(sol_time, zeros(nstore))
		end

		#store the solution
		range = (iseg-1)*M+2:(iseg)*M+1
		sol_pos[range, :] = ys[2:end, :]
		sol_vel[range, :] = dys[2:end, :]
		sol_acc[range, :] = ddys[2:end, :]
		sol_time[range] = ts[2:end]
	end #while
	return sol_time[1:iseg*M+1], sol_pos[1:iseg*M+1, :], sol_vel[1:iseg*M+1, :], sol_acc[1:iseg*M+1, :], istat, feval
end