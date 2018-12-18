function is_pareto_efficient(X, minimize=true, return_mask = true)
    M, N = size(X)
    is_efficient = range(1,stop=N)
    next_point_index = 1  # Next index in the is_efficient array to search for
    _X = copy(X)

    while next_point_index<=size(_X,2)
        if minimize
            nondominated_point_mask = vec(any(_X.<=_X[:,next_point_index],dims=1))
        else
            nondominated_point_mask = vec(any(_X.>=_X[:,next_point_index],dims=1))
        end
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        _X = _X[:,nondominated_point_mask]
        next_point_index = sum(nondominated_point_mask[1:next_point_index])+1
    end
    if return_mask
        is_efficient_mask = zeros(Bool,N)
        is_efficient_mask[is_efficient] .= true
        return is_efficient_mask
    else
        return is_efficient
    end
end
