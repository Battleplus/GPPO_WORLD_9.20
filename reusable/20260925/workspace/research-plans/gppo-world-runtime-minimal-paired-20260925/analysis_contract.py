"""Three scientific conclusions; clear loss separated from uncertainty."""
import math

def decide(*,utility_ci,rule_ci,physical_ci,host_ci,loss_family_ci,
           n_cpu_ms,n_wall_p95_ms,rss_sum_bytes,replay_ratio,formal_decision_ratio,formal_episode_ratio):
    cis=[utility_ci,rule_ci,physical_ci,host_ci,*loss_family_ci]
    if any(len(c)!=2 or c[0]>c[1] or not all(math.isfinite(x) for x in c) for c in cis):raise ValueError('Invalid CI')
    costs=[n_cpu_ms,n_wall_p95_ms,rss_sum_bytes,replay_ratio,formal_decision_ratio,formal_episode_ratio]
    if not all(math.isfinite(x) and x>=0 for x in costs):raise ValueError('Invalid cost')
    retained=utility_ci[0]>-.01
    practical=rule_ci[0]>.01
    tasks=physical_ci[0]>-.02 and host_ci[0]>-.02
    absolute=n_cpu_ms<=10 and n_wall_p95_ms<=50 and rss_sum_bytes<=4*1024**3
    saving=replay_ratio<=.8 and formal_decision_ratio<=.8 and formal_episode_ratio<=.8
    clear_loss=any(ci[1]<-margin for ci,margin in zip(loss_family_ci,[.01,.02,.02]))
    if retained and practical and tasks and absolute and saving:
        result='support_simplification_of_frozen_inference'
    elif clear_loss:
        result='clear_loss_direct_removal_not_supported'
    else:
        result='uncertain_or_practical_gate_not_met_stop'
    return {'decision':result,'utility_retention_pass':retained,'increment_vs_rule_pass':practical,
        'task_guard_pass':tasks,'absolute_cost_pass':absolute,'cpu_saving_pass':saving,
        'clear_loss_family_pass':clear_loss,'world_training_contribution_proven':False,
        'automatic_training':False,'automatic_more_samples':False}
