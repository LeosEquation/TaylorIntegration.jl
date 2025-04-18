# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: Transpose, norm
using Logging
import Logging: Warn

@testset "Testing `many_ode.jl`" begin

    local _order = 28
    local _abstol = 1.0E-20
    local tT = Taylor1(_order)

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    @testset "Tests: dot{x}=x.^2, x(0) = [3, 1]" begin
        function eqs_mov!(Dx, x, p, t)
            for i in eachindex(x)
                Dx[i] = x[i]^2
            end
            nothing
        end
        exactsol(t, x0) = x0 / (1.0 - x0 * t)
        t0 = 0.0
        q0 = [3, 1]
        q0T = Taylor1.(Float64.(q0), _order)
        xdotT = Array{Taylor1{Float64}}(undef, length(q0))
        xaux = Array{Taylor1{Float64}}(undef, length(q0))
        tT[1] = t0
        TaylorIntegration.jetcoeffs!(eqs_mov!, tT, q0T, xdotT, xaux, nothing)
        @test q0T[1].coeffs[end] == 3.0^(_order + 1)
        @test q0T[2].coeffs[end] == 1.0
        δt = (_abstol / q0T[1].coeffs[end-1])^inv(_order - 1)
        @test TaylorIntegration.stepsize(q0T, _abstol) == δt

        sol = @test_logs(
            (Warn, max_iters_reached()),
            @inferred(
                TaylorSolution{Float64,Float64,2},
                taylorinteg(eqs_mov!, q0, 0.0, 0.5, _order, _abstol, nothing)
            )
        )
        @test_throws ErrorException sol(0.25)
        tv = sol.t
        xv = sol.x
        @test length(tv) == 501
        @test isa(xv, SubArray)
        @test xv[1, :] == q0
        @test tv[end] < 1 / 3

        trange = 0.0:1/8:1.0
        sol = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            eqs_mov!,
            q0,
            trange,
            _order,
            _abstol,
        ))
        xv = sol.x
        @test size(xv) == (9, 2)
        @test q0 == [3.0, 1.0]
        @test typeof(xv) == Transpose{Float64,Array{Float64,2}}
        @test xv[1, 1:end] == q0
        @test (isnan(xv[4, 1]) && isnan(xv[4, 2]))
        @test (isnan(xv[end, 1]) && isnan(xv[end, 2]))
        @test abs(xv[3, 2] - 4 / 3) ≤ eps(4 / 3)
        @test abs(xv[2, 1] - 4.8) ≤ eps(4.8)

        tarray = vec(trange)
        sol2 = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            eqs_mov!,
            q0,
            tarray,
            _order,
            _abstol,
            nothing,
        ))
        xv2 = sol2.x
        @test xv[1:3, :] == xv2[1:3, :]
        @test xv2[1:3, :] ≈ xv[1:3, :] atol = eps() rtol = 0.0
        @test size(xv2) == (9, 2)
        @test q0 == [3.0, 1.0]
        @test typeof(xv2) == Transpose{Float64,Array{Float64,2}}
        @test xv2[1, 1:end] == q0
        @test (isnan(xv2[4, 1]) && isnan(xv2[4, 2]))
        @test (isnan(xv2[end, 1]) && isnan(xv2[end, 2]))
        @test abs(xv2[3, 2] - 4 / 3) ≤ eps(4 / 3)
        @test abs(xv2[2, 1] - 4.8) ≤ eps(4.8)

        # Output includes Taylor polynomial solution
        sol = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            eqs_mov!,
            q0,
            0,
            0.5,
            _order,
            _abstol,
            nothing,
            dense = true,
            maxsteps = 2,
        ))
        tv = sol.t
        xv = sol.x
        psol = sol.p
        @test size(psol) == (2, 2)
        @test xv[1, :] == q0
        @test xv[2, :] == evaluate.(psol[1, :], tv[2] - tv[1])
        @test psol[1, 1] == Taylor1([3.0^i for i = 1:_order+1])
        @test psol[1, 2] == Taylor1(ones(_order + 1))
    end

    @testset "Tests: dot{x}=x.^2, x(0) = [3, 3]" begin
        function eqs_mov!(Dx, x, p, t)
            for i in eachindex(x)
                Dx[i] = x[i]^2
            end
            nothing
        end
        exactsol(t, x0) = x0 / (1.0 - x0 * t)

        q0 = [3.0, 3.0]
        tmax = 0.3
        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            eqs_mov!,
            q0,
            0,
            tmax,
            _order,
            _abstol,
            nothing,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test length(xv[:, 2]) < 501
        @test length(tv) == 14
        @test length(xv[:, 1]) == 14
        @test length(xv[:, 2]) == 14
        @test xv[1, 1:end] == q0
        @test tv[end] < 1 / 3
        @test tv[end] == tmax
        @test xv[end, 1] == xv[end, 2]
        @test abs(xv[end, 1] - exactsol(tv[end], xv[1, 1])) < 5e-14
        @test abs(xv[end, 2] - exactsol(tv[end], xv[1, 2])) < 5e-14

        tmax = 0.33
        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            eqs_mov!,
            [3, 3],
            0.0,
            tmax,
            _order,
            _abstol,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test length(xv[:, 2]) < 501
        @test length(tv) == 28
        @test length(xv[:, 1]) == 28
        @test length(xv[:, 2]) == 28
        @test xv[1, 1:end] == q0
        @test tv[end] < 1 / 3
        @test tv[end] == tmax
        @test xv[end, 1] == xv[end, 2]
        @test abs(xv[end, 1] - exactsol(tv[end], xv[1, 1])) < 1.0e-11
        @test abs(xv[end, 2] - exactsol(tv[end], xv[1, 2])) < 1.0e-11

        # Output includes Taylor polynomial solution
        sol = taylorinteg(eqs_mov!, q0, 0, tmax, _order, _abstol, nothing, dense = true)
        tv = sol.t
        xv = sol.x
        psol = sol.p
        @test size(psol) == (size(xv, 1) - 1, size(xv, 2))
        @test psol[1, :] == fill(Taylor1([3.0^i for i = 1:_order+1]), 2)
        @test xv[end, 1] == evaluate.(psol[end, 1], tv[end] - tv[end-1])
        @test xv[end, :] == sol(tv[end])
        @test psol[:, 1] == psol[:, 2]
    end

    @testset "Test non-autonomous ODE (2): dot{x}=cos(t)" begin
        function f!(Dx, x, p, t)
            Dx[1] = one(t)
            Dx[2] = cos(t)
            nothing
        end
        t0 = 0 // 1
        tmax = 10.25 * (2pi)
        abstol = 1e-20
        order = 25
        x0 = [t0, 0.0] #initial conditions such that x(t)=sin(t)

        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            f!,
            x0,
            t0,
            tmax,
            order,
            abstol,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test length(xv[:, 2]) < 501
        @test xv[1, 1:end] == x0
        @test tv[1] < tv[end]
        @test tv[end] == xv[end, 1]
        @test abs(sin(tmax) - xv[end, 2]) < 1e-14

        # Backward integration
        solb = (@test_logs min_level = Logging.Warn taylorinteg(
            f!,
            [tmax, sin(tmax)],
            tmax,
            t0,
            order,
            abstol,
            dense = true,
        ))
        tb = solb.t
        xb = solb.x
        @test length(tb) < 501
        @test length(xb[:, 1]) < 501
        @test length(xb[:, 2]) < 501
        @test tb[1] > tb[end]
        @test xb[1, 1:end] == [tmax, sin(tmax)]
        @test tb[end] == xb[end, 1]
        @test abs(sin(t0) - xb[end, 2]) < 5e-14
        @test solb(tmax) == [tmax, sin(tmax)]
        @test solb(x0[1]) == xb[end, :]
        tmidT = (tmax + t0) / 2 + Taylor1(order)
        @test solb(tmidT)[1] == tmidT
        @test norm(solb(tmidT)[2] - sin(tmidT), Inf) < 1e-13

        # Tests with a range, for comparison with backward integration
        tmax = 15 * (2pi)
        Δt = (tmax - t0) / 1024
        tspan = t0:Δt:tmax
        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            f!,
            x0,
            tspan,
            order,
            abstol,
            nothing,
        ))
        xv = sol.x
        @test xv[1, 1:end] == x0
        @test tmax == xv[end, 1]
        @test abs(sin(tmax) - xv[end, 2]) < 1e-14

        # Backward integration
        solback = (@test_logs min_level = Logging.Warn taylorinteg(
            f!,
            xv[end, :],
            reverse(tspan),
            order,
            abstol,
            nothing,
        ))
        xback = solback.x
        @test xback[1, :] == xv[end, :]
        @test abs(xback[end, 1] - x0[1]) < 5.0e-14
        @test abs(xback[end, 2] - x0[2]) < 5.0e-14
        @test abs(sin(t0) - xback[end, 2]) < 5.0e-14
        @test norm((xv[:, :] - xback[end:-1:1, :]), Inf) < 5.0e-14

        # Tests if trange is properly sorted
        @test_throws AssertionError taylorinteg(
            f!,
            x0,
            rand(t0:Δt:tmax, 100),
            order,
            abstol,
        )
    end

    @testset "Falling ball (stepsize)" begin
        function fallball!(dx, x, p, t)
            dx[1] = x[2]
            dx[2] = -one(x[1])
            nothing
        end
        exactsol(t, x0, v0) = x0 + v0 * t - 0.5 * t^2
        t0 = 0.0
        tmax = 5.0
        abstol = 1e-20
        order = 10
        x0 = [10.0, 0.0] #initial conditions such that x(t)=sin(t)
        sol = (@test_logs min_level = Logging.Warn taylorinteg(
            fallball!,
            x0,
            t0,
            tmax,
            order,
            abstol,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test exactsol.(tv, x0[1], x0[2]) ≈ xv[:, 1]
    end

end
