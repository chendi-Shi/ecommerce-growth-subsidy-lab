"""Decision diagnostics for the already frozen Hillstrom policies.

This is a retrospective extension of a previously inspected test set. It adds
economic and precision diagnostics, not a new untouched validation experiment.
No model, feature, content choice, contact threshold, or seed is selected here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .download import prepare
from .models import feature_frame, fit_all, predictions, rankers, split
from .stats import dr_scores, estimate, fixed_random, policy_scores, targeting


def planning_n(score_variance, alternative, hurdle=0., alpha=.05, power=.8):
    """TOTAL future three-arm logging users for a fixed-policy OPE contrast.

    A conservative two-sided normal planning approximation. The design alternative
    must exceed the null/business hurdle; the observed effect is not the assumed
    alternative. This is not the per-arm size of a direct policy A/B experiment.
    """
    if score_variance <= 0 or alternative <= hurdle or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError('Positive pilot variance and alternative > hurdle required.')
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z*z*score_variance / (alternative-hurdle)**2))


def hurdle_status(low, high, hurdle):
    if low > high:
        raise ValueError('Unordered interval.')
    if low > hurdle:
        return 'lower_bound_above_hurdle'
    if high < hurdle:
        return 'upper_bound_below_hurdle'
    return 'unresolved_at_this_hurdle'


def content_audience_components(mixed, same_audience_one_content, random_one_content):
    """Exact paired-score decomposition, not a causal mediation analysis."""
    mixed = np.asarray(mixed, dtype=float)
    same = np.asarray(same_audience_one_content, dtype=float)
    random = np.asarray(random_one_content, dtype=float)
    if mixed.shape != same.shape or same.shape != random.shape:
        raise ValueError('Policy scores must cover the same users.')
    return {'audience_selection': same-random,
            'content_selection_given_audience': mixed-same,
            'combined_targeting': mixed-random}


def reconstruct(frame, summary):
    train, validation, test = split(frame, summary['seed'])
    models = fit_all(train, summary['seed'])
    mu = predictions(models['ridge'], test)
    psi = dr_scores(test.spend, test.treatment, mu)
    predicted = rankers(models, test)
    arm = int(summary['best_homogeneous_arm'])
    economics = summary['economics']

    def chosen(row):
        if row['model'] == 'none':
            return fixed_random(len(test), arm, 0)
        return targeting(predicted[row['model']], row['depth'],
                         economics['margin'], economics['contact_cost_usd'])

    targeted = chosen(summary['budget_selected'])
    contacts = 1-targeted[:, 0]
    share = float(contacts.mean())
    homogeneous = np.zeros_like(targeted)
    homogeneous[:, 0] = 1-contacts
    homogeneous[:, arm] = contacts
    probability = models['conversion'].predict_proba(feature_frame(test))[:, 1]
    natural = np.zeros(len(test), dtype=int)
    natural[np.argsort(-probability, kind='stable')[:int(np.floor(share*len(test)))]] = arm
    policies = {
        'no_contact': fixed_random(len(test), arm, 0),
        'all_best_arm': fixed_random(len(test), arm, 1),
        'random_same_contacts': fixed_random(len(test), arm, share),
        'purchase_propensity': np.eye(3)[natural],
        'budget_targeted': targeted,
        'validation_optimized': chosen(summary['validation_selected']),
        'same_targeted_users_one_content': homogeneous,
    }
    values = {name: policy_scores(p, psi) for name, p in policies.items()}
    errors = {row['policy']: abs(float(values[row['policy']].mean())-row['incremental_revenue'])
              for row in summary['policies']}
    if max(errors.values()) > 1e-9:
        raise ValueError(f'Frozen policy reconstruction mismatch: {errors}')
    if len(test) != summary['test_n'] or len(validation) != summary['validation_n']:
        raise ValueError('Frozen split changed.')
    return test, policies, values, errors


def run(data_root, source_report, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source_report/'summary.json').read_text(encoding='utf-8'))
    path, source = prepare(data_root)
    frame = pd.read_csv(path, dtype={'customer_id': 'string'})
    test, policies, values, errors = reconstruct(frame, summary)
    components = content_audience_components(values['budget_targeted'],
        values['same_targeted_users_one_content'], values['random_same_contacts'])
    component_table = pd.DataFrame([dict(component=name, **estimate(score))
                                   for name, score in components.items()])
    component_table.to_csv(output/'audience_content_decomposition.csv', index=False)
    identity_error = float(np.max(np.abs(components['audience_selection']+
        components['content_selection_given_audience']-components['combined_targeting'])))
    if identity_error > 1e-10:
        raise ValueError('Paired-score decomposition did not reconcile.')

    thresholds = []
    for name, p in policies.items():
        q = float((1-p[:, 0]).mean())
        result = estimate(values[name])
        thresholds.append(dict(policy=name, contact_share=q, **result,
            revenue_cost_ratio_break_even=result['estimate']/q if q else None,
            break_even_low=result['ci_low']/q if q else None,
            break_even_high=result['ci_high']/q if q else None))
    threshold_table = pd.DataFrame(thresholds)
    threshold_table.to_csv(output/'operating_thresholds.csv', index=False)

    # c/m is the required incremental revenue per contact, without invented costs.
    # Policies remain frozen. The curve cannot be used as fresh test-set tuning.
    sensitivity = []
    for ratio in np.linspace(0., 1., 21):
        for name, p in policies.items():
            score = values[name]-ratio*(1-p[:, 0])
            sensitivity.append(dict(policy=name, contact_cost_over_margin=ratio,
                                    **estimate(score)))
    pd.DataFrame(sensitivity).to_csv(output/'cost_ratio_sensitivity.csv', index=False)
    combined = estimate(components['combined_targeting'])
    hurdles = [0., .02, .05, .10, .20, .30]
    decisions = [dict(illustrative_revenue_hurdle=h,
                     status=hurdle_status(combined['ci_low'], combined['ci_high'], h),
                     ci_low=combined['ci_low'], ci_high=combined['ci_high']) for h in hurdles]
    pd.DataFrame(decisions).to_csv(output/'business_hurdles.csv', index=False)
    # Forward planning uses historical variance only. All alternatives are explicit
    # scenarios; none is obtained by treating the observed -0.006 as truth.
    pilot_variance = float(np.var(components['combined_targeting'], ddof=1))
    plans = []
    for hurdle, alternative in [(0., .02), (0., .05), (0., .10), (0., .20),
                               (.02, .05), (.05, .10), (.05, .20)]:
        for variance_factor in [.5, 1., 2.]:
            plans.append(dict(null_hurdle=hurdle, design_alternative=alternative,
                distance_to_null=alternative-hurdle, pilot_score_variance=pilot_variance,
                variance_factor=variance_factor, alpha=.05, power=.8,
                future_total_users=planning_n(pilot_variance*variance_factor,
                                              alternative, hurdle)))
    planning = pd.DataFrame(plans)
    planning.to_csv(output/'future_evaluation_size.csv', index=False)
    out = dict(test_n=len(test), test_buyers=int(test.conversion.sum()),
        evidence_status='RETROSPECTIVE diagnostic extension; previously inspected test set',
        source=source, frozen_seed=summary['seed'],
        frozen_policy_max_reconstruction_error=max(errors.values()),
        paired_decomposition_max_error=identity_error,
        full_vs_test='Full 64k average ITT and 12.8k policy OPE have different precision; do not conflate.',
        combined_targeting=combined, pilot_score_variance=pilot_variance,
        illustrative_hurdles=decisions,
        economic_formula='Net advantage = N * (margin * revenue_advantage - cost * contact_share_difference) - extra_implementation_cost',
        equal_contact_hurdle='required revenue advantage = extra_implementation_cost / (N * margin)',
        cost_ratio='c/m, USD revenue required per contact; actual c and m are unobserved',
        planning_design='NEW independent three-arm 1/3 logging experiment, same frozen policies and DR estimator; N is TOTAL users, not per-arm policy-A/B size',
        planning_limits='Normal approximation and stable pilot variance; no peeking/early stopping; direct policy A/B requires different variance/design; no retraining uncertainty included')
    (output/'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ['Audience selection', 'Content given audience', 'Combined targeting']
    means = component_table.estimate.to_numpy()
    axes[0].errorbar(means, np.arange(3), xerr=np.vstack([
        means-component_table.ci_low.to_numpy(), component_table.ci_high.to_numpy()-means]),
        fmt='o', capsize=4, color='#2563eb')
    axes[0].set_yticks(np.arange(3), names)
    axes[0].axvline(0, color='gray', linewidth=1)
    axes[0].set_xlabel('Incremental spend difference / eligible user (USD)')
    axes[0].set_title('Who to contact vs which content?\nFixed policies, pointwise 95% intervals')
    chosen = planning[(planning.null_hurdle == 0) & (planning.variance_factor == 1)]
    axes[1].plot(chosen.design_alternative, chosen.future_total_users, marker='o', color='#0891b2')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Design advantage over null (USD / eligible user)')
    axes[1].set_ylabel('Total NEW three-arm evaluation users (log scale)')
    axes[1].set_title('Precision has a research cost\nIllustrative 80% power, alpha = 0.05')
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'decision_precision.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(combined=combined, decomposition=component_table.to_dict('records'),
        pilot_variance=pilot_variance, reconstruction_error=max(errors.values()), output=str(output)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--source-report', type=Path, default=Path('reports/campaign'))
    parser.add_argument('--output', type=Path, default=Path('reports/campaign_decision'))
    args = parser.parse_args()
    run(args.data_root, args.source_report, args.output)


if __name__ == '__main__':
    main()
