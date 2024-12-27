module TAPC

export integrate_ivp2, clenshaw_curtis_nested_ivpd, interpolate, clenshaw_curtis_ivpi, clenshaw_curtis_ivpii

include("tapcintegrator.jl")
include("clenshawcurtisivp.jl")

end
