"""
    Sensors

Provides efficient algorithms for information-theoretic sensor placement.

See the paper "Near-Optimal Sensor Placements in Gaussian Processes:
Theory, Efficient Algorithms and Empirical Studies" by Krause et al., 2008
"""
module Sensors

export entropynaive, entropy, miprec, mi

using LinearAlgebra: LinearAlgebra, Cholesky, mul!

using PyPrint: pprint, @pprint
using KernelFunctions:
    kernelmatrix,
    kernelmatrix!,
    kernelmatrix_diag,
    kernelmatrix_diag!

"""
    pddot(theta, x)

Compute the quadratic form ``x^{\top} \\Theta^{-1} x = b``.
"""
pddot(theta::Cholesky, x) = (y = theta.U' \ x; vec(sum(y .* y; dims=1)))
pddot(theta, x) = pddot(LinearAlgebra.cholesky(theta), x)

"""
    choldowndate!(L, u, j)

Computes the rank-one downdate in-place, L -> chol(L L^T - u u^T).
"""
function choldowndate!(L, u, j)
    for i in 1:j
        c1, c2 = L[i, i], u[i]
        dp = sqrt(c1^2 - c2^2)
        c1, c2 = c1/dp, c2/dp
        # @. @views L[:, i] = c1*L[:, i] - c2*u
        # @. @views u = (1/c1)*u - (c2/c1)*L[:, i]
        @views LinearAlgebra.axpby!(-c2, u, c1, L[:, i])
        @views LinearAlgebra.axpby!(-c2/c1, L[:, i], 1/c1, u)
    end
    return
end

"""
    deleterowcol!(A, j)

Delete the `j`th row and column of `A` in-place.
"""
function deleterowcol!(A, j)
    # mask = [1:j - 1..., j + 1:size(A, 1)...]
    # @views A[mask, mask]
    # copy rows
    n = size(A, 1)
    for i in j:n - 1
        A[i, :] = @view A[i + 1, :]
    end
    # copy columns
    for i in j:n - 1
        A[:, i] = @view A[:, i + 1]
    end
    @view A[1:n - 1, 1:n - 1]
end

"""
    entropynaive(X, kernel, s)

Select the `s` most entropic points from `X` greedily.
"""
function entropynaive(X, kernel, s)
    n = length(X)
    s = min(s, n)
    indexes = Array{typeof(s)}(undef, s)
    # unroll first unconditional selection
    var = kernelmatrix_diag(kernel, X)
    k = argmax(var)
    indexes[1] = k
    var[k] = -1
    # pre-allocate covariances
    theta = Array{eltype(var)}(undef, s, s)
    cov = Array{eltype(var)}(undef, n, s)
    @views kernelmatrix!(cov[:, 1:1], kernel, X, X[k:k])
    theta[1, 1] = cov[k, 1]

    for i in 2:s
        # compute conditional variance
        @views cond_var = var - pddot(theta[1:i - 1, 1:i - 1],
                                      cov[:, 1:i - 1]')
        # pick best entry
        k = argmax(cond_var)
        indexes[i] = k
        var[k] = -1
        # store kernel function evaluations
        @views kernelmatrix!(cov[:, i:i], kernel, X, X[k:k])
        @views theta[1:i, i] = cov[indexes[1:i], i]
        @views theta[i, 1:i] = cov[indexes[1:i], i]
    end
    return indexes
end

"""
    entropy(X, kernel, s)

Select the `s` most entropic points from `X` greedily.
"""
function entropy(X, kernel, s)
    n = length(X)
    s = min(s, n)
    # initialization
    indexes = Array{typeof(s)}(undef, s)
    cond_var = kernelmatrix_diag(kernel, X)
    L = Array{eltype(cond_var)}(undef, n, s)

    for i in 1:s
        # pick best entry
        k = argmax(cond_var)
        indexes[i] = k
        # update Cholesky factor by left looking
        @views kernelmatrix!(L[:, i:i], kernel, X, X[k:k])
        # @views L[:, i] .-= L[:, 1:i - 1]*L[k, 1:i - 1]
        @views mul!(L[:, i], L[:, 1:(i - 1)], L[k, 1:(i - 1)], -1.0, 1.0)
        @views @. L[:, i] /= sqrt(L[k, i])
        # update conditional variance
        @views @. cond_var -= L[:, i]^2
        cond_var[k] = -1
    end
    return indexes
end

"""
    miprec(X, kernel, s)

Max mutual information between selected and non-selected points.
"""
function miprec(X, kernel, s)
    n = length(X)
    s = min(s, n)
    # initialization
    indexes = Array{typeof(s)}(undef, s)
    candidates = Array{typeof(s)}(1:n)
    prec = inv(kernelmatrix(kernel, X))
    L = Array{eltype(prec)}(undef, n, s)
    cond_var1 = kernelmatrix_diag(kernel, X)
    cond_var2 = LinearAlgebra.diag(prec)

    for i in 1:s
        # pick best entry
        k = argmax(cond_var1 .* cond_var2)
        indexes[i] = k
        j = only(indexin(k, candidates))
        deleteat!(candidates, j)
        # update Cholesky factor
        @views kernelmatrix!(L[:, i:i], kernel, X, X[k:k])
        @views L[:, i] .-= L[:, 1:i - 1]*L[k, 1:i - 1]
        L[:, i] ./= sqrt(L[k, i])
        # update conditional variance
        @views cond_var1 .-= L[:, i].^2
        cond_var1[k] = -1
        # update precision of candidates
        # marginalization in covariance is conditioning in precision
        prec .-= prec[:, j]*prec[:, j]' ./ prec[j, j]
        mask = [1:j - 1..., j + 1:size(prec, 1)...]
        @views prec = prec[mask, mask]
        # update conditional variance of candidates
        cond_var2[candidates] .= LinearAlgebra.diag(prec)
    end
    return indexes
end

"""
    mi(X, kernel, s)

Max mutual information between selected and non-selected points.
"""
function mi(X, kernel, s)
    n = length(X)
    s = min(s, n)
    # initialization
    indexes = Array{typeof(s)}(undef, s)
    candidates = Array{typeof(s)}(1:n)
    L2 = inv(LinearAlgebra.cholesky(kernelmatrix(kernel, reverse(X))).U)
    L2 = L2[end:-1:begin, end:-1:begin]
    L1 = Array{eltype(L2)}(undef, n, s)
    cond_var1 = kernelmatrix_diag(kernel, X)
    cond_var2 = vec(sum(L2 .* L2; dims=2))

    for i in 1:s
        # pick best entry
        k = argmax(cond_var1 .* cond_var2)
        indexes[i] = k
        j = only(indexin(k, candidates))
        deleteat!(candidates, j)
        # update Cholesky factor
        @views kernelmatrix!(L1[:, i:i], kernel, X, X[k:k])
        @views L1[:, i] .-= L1[:, 1:i - 1]*L1[k, 1:i - 1]
        @views L1[:, i] ./= sqrt(L1[k, i])
        # update conditional variance
        @views cond_var1 .-= L1[:, i].^2
        cond_var1[k] = -1
        # update Cholesky factor of precision of candidates
        # marginalization in covariance is conditioning in precision
        u = L2*L2[j, :]
        u ./= sqrt(u[j])
        choldowndate!(L2, u, j - 1)
        L2 = deleterowcol!(L2, j)
        # update conditional variance of candidates
        # cond_var2[candidates] .= vec(sum(L2 .* L2; dims=2))
        v = zeros(eltype(cond_var2), length(candidates))
        for j in 1:size(L2, 1)
            @views v .+= L2[:, j].^2
        end
        cond_var2[candidates] .= v
    end
    return indexes
end

end
