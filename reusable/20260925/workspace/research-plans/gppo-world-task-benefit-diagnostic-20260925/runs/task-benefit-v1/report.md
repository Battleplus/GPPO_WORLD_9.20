# Independent task-benefit diagnostic (prior cost failure retained)

Decision: `support_preparing_simplification_task_basis_only_cost_not_qualified`.

800 new N episodes; 1600 corresponding historical M/R episodes reused. Stage2 full9600 conclusion unchanged.

Utility contrasts: {"N-M":{"energy_used":{"ci95":[-0.005919659949126123,0.012146957946210875],"mean_difference":0.003560195364926746,"sd":0.13110587529572515},"host_on_time_observed":{"ci95":[-0.002083333333333334,0.00020833333333333343],"mean_difference":-0.0008333333333333337,"sd":0.01665623369746487},"physical_on_time":{"ci95":[0.0,0.0014583333333333332],"mean_difference":0.0006249999999999999,"sd":0.010193425531770776},"utility":{"ci95":[-3.178172165290304e-05,0.0009952632955688395],"mean_difference":0.00041036104355888115,"sd":0.007347312191572461}},"N-R":{"energy_used":{"ci95":[0.9060913955562048,1.0219169221275681],"mean_difference":0.9642853332328581,"sd":0.8387122961425363},"host_on_time_observed":{"ci95":[0.07875000000000001,0.1004166666666667],"mean_difference":0.08958333333333335,"sd":0.15764870759022162},"physical_on_time":{"ci95":[0.08208333333333333,0.10020833333333332],"mean_difference":0.09104166666666666,"sd":0.1329323274646239},"utility":{"ci95":[0.056856860459785184,0.07018504922785092],"mean_difference":0.06344182240275784,"sd":0.09673677945432475}}}

Costs: {"cpu_seconds":86.78125,"decisions":10514,"matched_M_cpu_seconds":98.6875,"mean_cpu_ms":8.253875784668063,"paired_replay_ratio":0.7931818181818182,"per_decision_ratio_to_matched_M":0.8803576593734365,"per_episode_ratio_to_matched_M":0.8793540215326155,"wall_p95_ms":3.8244999988819472}

Prior A cost failure remains; practical acceptance is false regardless of these task outcomes. World-independent contribution is not established by this removal test. No automatic training or follow-up. Full process costs and historical lineage are in status.json.
