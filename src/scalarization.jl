linear_scalarization(X, weights=ones(size(X,1))) = X'*weights
masked_max(X, mask=fill(true,size(X,1))) = map(x-> x[1] ? x[2] : x[3], 
	zip(mask, maximum(X, dims=2), ones(size(X,1))))
no_preference(X, z_ideal, norm=2) = vec(mapslices(x->LinearAlgebra.norm(x,norm), X.-z_ideal, dims=1))
function best_index(X, weight_mask=fill(true,size(X,1)), variant="linear")
	mmx = masked_max(X, weight_mask)
	crit = fill(0.0,size(X,2))
	if variant=="linear"
		crit = linear_scalarization(X, 1 ./ mmx)
	elseif variant=="no_preference"
		crit = -no_preference(X, mmx)
	end
	return argmax(crit), crit
end
function pareto_best_index(X, weight_mask=fill(true,size(X,1)), variant="linear")
	pareto_mask = is_pareto_efficient(X)
	bi, _ = best_index(X[:,pareto_mask], weight_mask, variant)
	return findall(x->x, pareto_mask)[bi]
end