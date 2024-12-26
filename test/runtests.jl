using TAPC
using Test

@testset "TAPC.jl" begin
    #simple 2nd order ode
    ode = f(t,y,dy,params) = [-(2*pi)^2*y[1]]
    y0 = [1]
    dy0 = [0]
    t0 = 0.0
    tf = 1
    params = Dict()
    tol = 1e-14
    
    
    t,y,dy,ddy = integrate_ivp2(y0,dy0,t0,tf,tol,ode,params,N=20,exponent=1/5)
    @test isapprox(y[end],cos(2*pi),rtol=1e-7)
end
