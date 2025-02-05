module GeneralAdaptivePicardChebyshev

export integrate_ivp2, integrate_ivp1, clenshaw_curtis_nested_ivpd, interpolate, clenshaw_curtis_ivpi, clenshaw_curtis_ivpii

include("gapcintegrator.jl")
include("clenshawcurtisivp.jl")

end
