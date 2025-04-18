# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: norm
using Logging
import Logging: Warn

@testset "Testing `rootfinding.jl`" begin

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    err_newton_raphson() =
        "Newton-Raphson did not converge for prescribed tolerance and maximum allowed iterations.\n"
    local _order = 28
    local _abstol = 1.0E-20

    function pendulum!(dx, x, p, t)
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end

    g(dx, x, p, t) = (true, x[2])

    t0 = 0.0
    x0 = [1.3, 0.0]
    Tend = 7.019250311844546

    p = set_variables("ξ", numvars = length(x0), order = 2)
    x0N = x0 + p
    ξ = Taylor1(2)
    x01 = x0 + [ξ, ξ]

    #warm-up lap and preliminary tests
    sol = (@test_logs (Warn, max_iters_reached()) taylorinteg(
        pendulum!,
        g,
        x0,
        t0,
        Tend,
        _order,
        _abstol,
        maxsteps = 1,
    ))
    tv, xv, tvS, xvS, gvS = sol.t, sol.x, sol.tevents, sol.xevents, sol.gresids
    @test size(tv) == (2,)
    @test tv[1] == t0
    @test size(xv) == (2, 2)
    @test xv[1, :] == x0
    @test size(tvS) == (0,)
    @test size(xvS) == (0, 2)
    @test size(gvS) == (0,)

    solN = (@test_logs (Warn, max_iters_reached()) taylorinteg(
        pendulum!,
        g,
        x0N,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1,
    ))
    tvN, xvN, tvSN, xvSN, gvSN = solN.t, solN.x, solN.tevents, solN.xevents, solN.gresids
    @test eltype(tvN) == Float64
    @test eltype(xvN) == TaylorN{Float64}
    @test eltype(tvSN) == TaylorN{Float64}
    @test eltype(xvSN) == TaylorN{Float64}
    @test eltype(gvSN) == TaylorN{Float64}
    @test size(tvN) == (2,)
    @test tvN[1] == t0
    @test size(xvN) == (2, 2)
    @test xvN[1, :] == x0N
    @test size(tvSN) == (0,)
    @test size(xvSN) == (0, 2)
    @test size(gvSN) == (0,)

    sol1 = (@test_logs (Warn, max_iters_reached()) taylorinteg(
        pendulum!,
        g,
        x01,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1,
    ))
    tv1, xv1, tvS1, xvS1, gvS1 = sol1.t, sol1.x, sol1.tevents, sol1.xevents, sol1.gresids
    @test eltype(tv1) == Float64
    @test eltype(xv1) == Taylor1{Float64}
    @test eltype(tvS1) == Taylor1{Float64}
    @test eltype(xvS1) == Taylor1{Float64}
    @test eltype(gvS1) == Taylor1{Float64}
    @test size(tv1) == (2,)
    @test tv1[1] == t0
    @test size(xv1) == (2, 2)
    @test xv1[1, :] == x01
    @test size(tvS1) == (0,)
    @test size(xvS1) == (0, 2)
    @test size(gvS1) == (0,)

    #testing 0-th order root-finding
    sol = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
    ))
    tv, xv, tvS, xvS, gvS = sol.t, sol.x, sol.tevents, sol.xevents, sol.gresids
    @test tv[1] == t0
    @test xv[1, :] == x0
    @test size(tvS) == (5,)
    @test norm(tvS - [Tend / 2, Tend, 3Tend / 2, 2Tend, 5Tend / 2], Inf) < 1E-13
    @test norm(gvS, Inf) < eps()

    #testing 0-th order root-finding with dense output
    sol = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        dense = true,
    ))
    tv, xv, psol, tvS, xvS, gvS = sol.t, sol.x, sol.p, sol.tevents, sol.xevents, sol.gresids
    @test tv[1] == t0
    @test xv[1, :] == x0
    @test size(tvS) == (5,)
    @test norm(tvS - [Tend / 2, Tend, 3Tend / 2, 2Tend, 5Tend / 2], Inf) < 1E-13
    @test norm(gvS, Inf) < eps()
    for i = 1:length(tv)-1
        @test norm(psol[i, :](tv[i+1] - tv[i]) - xv[i+1, :], Inf) < 1e-13
        @test norm(sol(tv[i+1]) - xv[i+1, :], Inf) < 1e-13
    end

    #testing 0-th order root-finding with time ranges/vectors
    tvr = [t0, Tend / 2, Tend, 3Tend / 2, 2Tend, 5Tend / 2, 3Tend]
    @test_throws AssertionError taylorinteg(
        pendulum!,
        g,
        x0,
        view(tvr, :),
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = _order + 1,
    )
    solr = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0,
        view(tvr, :),
        _order,
        _abstol,
        maxsteps = 1000,
    ))
    xvr, tvSr, xvSr, gvSr = solr.x, solr.tevents, solr.xevents, solr.gresids
    @test xvr[1, :] == x0
    @test size(tvSr) == (5,)
    @test size(tvSr) == size(tvr[2:end-1])
    @test norm(tvSr - tvr[2:end-1], Inf) < 1E-13
    @test norm(tvr[2:end-1] - tvSr, Inf) < 1E-14
    @test norm(xvr[2:end-1, :] - xvSr, Inf) < 1E-14
    @test norm(gvSr[:]) < eps()
    @test norm(tvS - tvSr, Inf) < 5E-15

    #testing 0-th order root-finding + TaylorN jet transport
    solN = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0N,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
    ))
    tvN, xvN, tvSN, xvSN, gvSN = solN.t, solN.x, solN.tevents, solN.xevents, solN.gresids
    @test size(tvSN) == size(tvS)
    @test size(xvSN) == size(xvS)
    @test size(gvSN) == size(gvS)
    @test norm(gvSN[:]) < 1E-15
    @test norm(tvSN() - tvr[2:end-1], Inf) < 1E-13
    @test norm(xvSN() - xvS, Inf) < 1E-14

    #testing 0-th order root-finding + TaylorN jet transport + dense output
    solN = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0N,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        dense = true,
    ))
    tvN, xvN, psolN, tvSN, xvSN, gvSN =
        solN.t, solN.x, solN.p, solN.tevents, solN.xevents, solN.gresids
    @test size(tvSN) == size(tvS)
    @test size(xvSN) == size(xvS)
    @test size(gvSN) == size(gvS)
    @test norm(gvSN[:]) < 1E-15
    @test norm(tvSN() - tvr[2:end-1], Inf) < 1E-13
    @test norm(xvSN() - xvS, Inf) < 1E-14
    for i = 1:length(tvN)-1
        @test norm(psolN[i, :](tvN[i+1] - tvN[i]) - xvN[i+1, :], Inf) < 1e-12
        @test norm(solN(tvN[i+1]) - xvN[i+1, :], Inf) < 1e-12
    end

    #testing 0-th root-finding + Taylor1 jet transport
    sol1 = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x01,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
    ))
    tv1, xv1, tvS1, xvS1, gvS1 = sol1.t, sol1.x, sol1.tevents, sol1.xevents, sol1.gresids
    @test size(tvS1) == size(tvS)
    @test size(xvS1) == size(xvS)
    @test size(gvS1) == size(gvS)
    @test norm(gvS1[:]) < 1E-14
    @test norm(tvS1() - tvS, Inf) < 1E-13
    @test norm(xvS1() - xvS, Inf) < 1E-14

    #testing surface higher order crossing detections and root-finding
    @test_throws AssertionError taylorinteg(
        pendulum!,
        g,
        x0,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = _order + 1,
        newtoniter = 2,
    )

    sol = (@test_logs (Warn, err_newton_raphson()) match_mode = :any taylorinteg(
        pendulum!,
        g,
        x0,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
        newtoniter = 2,
    ))
    tv, xv, tvS, xvS, gvS = sol.t, sol.x, sol.tevents, sol.xevents, sol.gresids
    @test tv[1] < tv[end]
    @test tv[1] == t0
    @test xv[1, :] == x0
    @test size(tvS) == (5,)
    @test norm(tvS - tvr[2:end-1], Inf) < 1E-13
    @test norm(gvS[:]) < 1E-15

    # testing backward integrations
    solb = (@test_logs (Warn, err_newton_raphson()) match_mode = :any taylorinteg(
        pendulum!,
        g,
        xv[end, :],
        3Tend,
        t0,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
        newtoniter = 2,
    ))
    tvb, xvb, tvSb, xvSb, gvSb = solb.t, solb.x, solb.tevents, solb.xevents, solb.gresids
    @test tvb[1] > tvb[end]
    @test tvSb[1] > tvSb[end]
    @test norm(gvSb[:]) < 1E-14
    @test norm(tvSb[2:end] - reverse(tvr[2:end-1]), Inf) < 1E-13
    @test norm(xvSb[2:end, :] .- xvS[end:-1:1, :], Inf) < 1E-13

    #testing higher order root-finding + TaylorN jet transport
    solN = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x0N,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
    ))
    tvN, xvN, tvSN, xvSN, gvSN = solN.t, solN.x, solN.tevents, solN.xevents, solN.gresids
    @test size(tvSN) == size(tvS)
    @test size(xvSN) == size(xvS)
    @test size(gvSN) == size(gvS)
    @test norm(gvSN[:]) < 1E-13
    @test norm(tvSN() - tvr[2:end-1], Inf) < 1E-13
    @test norm(xvSN() - xvS, Inf) < 1E-14

    #testing higher root-finding + Taylor1 jet transport
    sol1 = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x01,
        t0,
        3Tend,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
    ))
    tv1, xv1, tvS1, xvS1, gvS1 = sol1.t, sol1.x, sol1.tevents, sol1.xevents, sol1.gresids
    @test size(tvS1) == size(tvS)
    @test size(xvS1) == size(xvS)
    @test size(gvS1) == size(gvS)
    @test norm(gvS1[:]) < 1E-14
    @test norm(tvS1() - tvr[2:end-1], Inf) < 1E-13
    @test norm(xvS1() - xvS, Inf) < 1E-14

    # Tests if trange is properly sorted
    Δt = (3Tend - t0) / 1000
    tspan = t0:Δt:(3Tend-0.125)
    sol1r = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        x01,
        tspan,
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
    ))
    xv1r, tvS1r, xvS1r, gvS1r = sol1r.x, sol1r.tevents, sol1r.xevents, sol1r.gresids
    sol1rb = (@test_logs min_level = Logging.Warn taylorinteg(
        pendulum!,
        g,
        xv1r[end, :],
        reverse(tspan),
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
    ))
    xv1rb, tvS1rb, xvS1rb, gvS1rb = sol1rb.x, sol1rb.tevents, sol1rb.xevents, sol1rb.gresids
    @test size(xv1r) == size(xv1rb)
    @test size(tvS1r) == size(tvS1rb)
    @test size(xvS1r) == size(xvS1rb)
    @test size(gvS1r) == size(gvS1rb)
    @test norm(gvS1r[:], Inf) < 1E-14
    @test norm(gvS1rb[:], Inf) < 1E-13
    @test tvS1r[1]() < tvS1r[end]()
    @test tvS1rb[1]() > tvS1rb[end]()
    @test norm(tvS1r() - reverse(tvS1rb()), Inf) < 5e-14
    @test norm(xv1r[:, :]() - xv1rb[end:-1:1, :](), Inf) < 5e-14

    @test_throws AssertionError taylorinteg(
        pendulum!,
        g,
        x0,
        rand(t0:Δt:3Tend, 100),
        _order,
        _abstol,
        maxsteps = 1000,
        eventorder = 2,
        newtoniter = 2,
    )
end
