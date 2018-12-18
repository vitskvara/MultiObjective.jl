import MultiObjective
import ADMetricEvaluation
ADME = ADMetricEvaluation

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
datasets = readdir(data_path)
dataset = datasets[20]
metrics = [:auc_mean,        
 :auc_weighted_mean,
 :auc_at_5_mean,
 #:auc_at_1_mean,    
 :prec_at_5_mean,   
 #:prec_at_1_mean,   
 :tpr_at_5_mean,    
 #:tpr_at_1_mean,    
 :vol_at_5_mean,    
 #:vol_at_1_mean
 ]

dfs = map(ADME.average_over_folds, ADME.loaddata(dataset, data_path))
df = dfs[4]
X = Array(convert(Array, df[metrics])')
pareto_mask = MultiObjective.is_pareto_efficient(X)
_X = X[:,pareto_mask]
weight_mask = fill(false,size(X,1))
weight_mask[2] = true
mmx = MultiObjective.masked_max(_X,weight_mask)
lin_crit = MultiObjective.linear_scalarization(_X,1 ./mmx)
no_pref_crit = MultiObjective.no_preference(_X,mmx)

best_lin_i = MultiObjective.pareto_best_index(X,weight_mask)
best_lin = X[:,best_lin_i]
println(best_lin_i, " ", best_lin)
best_np_i = MultiObjective.pareto_best_index(X,weight_mask,"no_preference")
best_np = X[:,best_np_i]
println(best_np_i, " ", best_np)
