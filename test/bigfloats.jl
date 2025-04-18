using TaylorIntegration, Elliptic
using Test
using LinearAlgebra: norm
using Logging
import Logging: Warn

@testset "Testing `bigfloats.jl`" begin

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    local _order = 90
    local _abstol = 1.0E-77

    function pendulum!(dx, x, p, t) #the simple pendulum ODE
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end

    @testset "Test ODE integration with BigFloats: simple pendulum" begin
        q0 = [big"1.3", 0.0] #the initial condition as a Vector{BigFloat}
        # T is the pendulum's librational period == 4Elliptic.K(sin(q0[1]/2)^2)
        # we will evaluate the elliptic integral K using TaylorIntegration.jl:
        G(x, p, t) = 1 / sqrt(1 - ((sin(big"1.3" / 2))^2) * (sin(t)^2)) # K elliptic integral kernel
        solk = (@test_logs min_level = Logging.Warn taylorinteg(
            G,
            0.0,
            0.0,
            BigFloat(π) / 2,
            _order,
            _abstol,
        ))
        tvk = solk.t
        xvk = solk.x
        @test eltype(tvk) == BigFloat
        @test eltype(xvk) == BigFloat
        T = 4xvk[end] # T = 4Elliptic.K(sin(q0[1]/2)^2)
        @test typeof(T) == BigFloat
        @test T ≈ 4Elliptic.K(sin(q0[1] / 2)^2) atol = eps(2.0) rtol = 0.0

        t0 = 0.0 #the initial time

        sol = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            pendulum!,
            q0,
            t0,
            T,
            _order,
            _abstol;
            maxsteps = 1,
        ))
        tv = sol.t
        xv = sol.x
        @test eltype(tv) == BigFloat
        @test eltype(xv) == BigFloat
        @test length(tv) == 2
        @test length(xv[:, 1]) == 2
        @test length(xv[:, 2]) == 2

        #note that T is a BigFloat
        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            pendulum!,
            q0,
            t0,
            T,
            _order,
            _abstol,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test length(xv[:, 2]) < 501
        #the line below implies that we've evaluated the pendulum's period
        #up to an accuracy comparable to eps(BigFloat) ~ 1e-77!!!
        @test norm(xv[end, :] .- q0, Inf) < 100eps(BigFloat)
    end
end
