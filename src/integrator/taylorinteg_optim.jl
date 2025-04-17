# This file is part of the TaylorIntegration.jl package; MIT licensed


# taylorinteg
function taylorinteg_optim!(f, x0::U, t0::T, tmax::T, order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true, dense::Bool=true) where {T<:Real, U<:Number}

    # Initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f, t, x, params)

    # Re-initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )
    _taylorinteg_optim!(Val(dense), f, t, x, x0, t0, tmax, abstol, rv, params; parse_eqs, maxsteps)
end

function _taylorinteg_optim!(dense::Val{D}, f, t::Taylor1{T}, x::Taylor1{U},
        x0::U, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, params;
        parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real, U<:Number, D}

    # Allocation
    # tv = Array{T}(undef, maxsteps+1)
    # xv = Array{U}(undef, maxsteps+1)
    # psol = init_psol(dense, xv, x)

    # Initial conditions
    nsteps = 1
    @inbounds t[0] = t0
    # @inbounds tv[1] = t0
    # @inbounds xv[1] = x0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f, t, x, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        # set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds x[0] = x0
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        # @inbounds tv[nsteps] = t0
        # @inbounds xv[nsteps] = x0
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    # return build_solution(tv, xv, psol, nsteps)
end


function taylorinteg_optim!(f!, q0::Array{U,1}, t0::T, tmax::T, order::Int, abstol::T, 
                            t::Taylor1{T}, x::Array{Taylor1{U},1}, dx::Array{Taylor1{U},1}, xaux::Array{Taylor1{U},1},
                            params = nothing;
                            maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Initialize the vector of Taylor1 expansions
    # dof = length(q0)
    # t = t0 + Taylor1( T, order )
    # x = Array{Taylor1{U}}(undef, dof)
    # dx = Array{Taylor1{U}}(undef, dof)
    # @inbounds for i in eachindex(q0)
    #     x[i] = Taylor1( q0[i], order )
    #     dx[i] = Taylor1( zero(q0[i]), order )
    # end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    # Re-initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x .= Taylor1.( q0, order )
    dx .= Taylor1.( zero.(q0), order)
    _taylorinteg_optim!(f!, t, x, dx, xaux, q0, t0, tmax, abstol, rv,
                        params; parse_eqs, maxsteps)
    
end

function _taylorinteg_optim!(f!, t::Taylor1{T}, x::Array{Taylor1{U},1}, dx::Array{Taylor1{U},1}, xaux::Array{Taylor1{U},1},
        q0::Array{U,1}, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, params;
        parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real, U<:Number}

    # Initialize the vector of Taylor1 expansions
    # dof = length(q0)

    # Allocation of output
    # tv = Array{T}(undef, maxsteps+1)
    # xv = Array{U}(undef, dof, maxsteps+1)
    # psol = init_psol(dense, xv, x)
    # xaux = Array{Taylor1{U}}(undef, dof)

    # Initial conditions
    @inbounds t[0] = t0
    # x0 = deepcopy(q0)
    # @inbounds tv[1] = t0
    # @inbounds xv[:,1] .= q0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, q0) # new initial condition
        # set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds for i in eachindex(q0)
            x[i][0] = q0[i]
            TaylorSeries.zero!(dx[i], 0)
        end
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        # @inbounds tv[nsteps] = t0
        # @inbounds xv[:,nsteps] .= deepcopy.(x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    # q0 .= deepcopy.(x0)

    # return build_solution(tv, xv, psol, nsteps)
end

@doc doc"""
    taylorinteg(f, x0, t0, tmax, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE ``\dot{x}=f(x, p, t)``,
where `p` are the parameters encoded in `params`.
The initial conditions are specified by `x0` at time `t0`; `x0` may be of type `T<:Number`
or `Vector{T}`, with `T` including `TaylorN{T}`; the latter case
is of interest for [jet transport applications](@ref jettransport).

The equations of motion are specified by the function `f`; we follow the same
convention of `DifferentialEquations.jl` to define this function, i.e.,
`f(x, p, t)` or `f!(dx, x, p, t)`; see the examples below.

The functions returns a `TaylorSolution`, whose fields are `t` and `x`; they represent,
respectively, a vector with the values of time (independent variable),
and a vector with the computed values of
the dependent variable(s). When the keyword argument `dense` is set to `true`, it also
outputs in the field `p` the Taylor polynomial expansion computed at each time step.
The integration stops when time is larger than `tmax`, in which case the last returned
value(s) correspond to `tmax`, or when the number of saved steps is larger
than `maxsteps`.

The integration method uses polynomial expansions on the independent variable
of order `order`; the parameter `abstol` serves to define the
time step using the last two Taylor coefficients of the expansions.
Make sure you use a *large enough* `order` to assure convergence.

Currently, the recognized keyword arguments are:
- `maxsteps[=500]`: maximum number of integration steps.
- `parse_eqs[=true]`: use the specialized method of `jetcoeffs!` created
    with [`@taylorize`](@ref).
- `dense[=true]`: output the Taylor polynomial expansion at each time step.

## Examples

For one dependent variable the function `f` defines the RHS of the equation of
motion, returning the value of ``\dot{x}``. The arguments of
this function are `(x, p, t)`, where `x` are the dependent variables, `p` are
the paremeters and `t` is the independent variable.

For several (two or more) dependent variables, the function `f!` defines
the RHS of the equations of motion, mutating (in-place) the (preallocated) vector
with components of ``\dot{x}``. The arguments of this function are `(dx, x, p, t)`,
where `dx` is the preallocated vector of ``\dot{x}``, `x` are the dependent
variables, `p` are the paremeters entering the ODEs and `t` is the independent
variable. The function may return this vector or simply `nothing`.

```julia
using TaylorIntegration

f(x, p, t) = x^2

sol = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

function f!(dx, x, p, t)
    for i in eachindex(x)
        dx[i] = x[i]^2
    end
    return nothing
end

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100, dense=true )
```

""" taylorinteg_optim!

# Generic functions
for R in (:Number, :Integer)
    @eval begin

        function taylorinteg_optim!(f, xx0::S, tt0::T, ttmax::U, order::Int, aabstol::V,
                params = nothing; dense=false, maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            # In order to handle mixed input types, we promote types before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg_optim!(f, x0, t0, tmax, order, abstol, params,
                dense=dense, maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg_optim!(f, q0::Array{S,1}, tt0::T, ttmax::U, order::Int, aabstol::V,
                params = nothing; dense=false, maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            #promote to common type before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            #convert the elements of q0 to the common, promoted type:
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg_optim!(f, q0_, t0, tmax, order, abstol, params,
                dense=dense, maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

    end
end
