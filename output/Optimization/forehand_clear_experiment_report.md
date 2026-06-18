# Forehand Clear Post-Processing Experiment

## Conclusion

Analyzed 27 forehand-clear clips. The final post-processed signal passed validation on 27/27 clips and was marked usable for training on 27/27 exported reference bundles (B: 27).

## Main Aggregate Metrics For Retargeting/Training

| Metric | Comparator mean | Post-processed mean | Improvement |
| --- | ---: | ---: | ---: |
| Beta variation max abs (raw -> post) | 0.8545 | 0.0000 | fixed to 0 |
| Max foot penetration, m (raw -> post) | 0.8999 | 0.0049 | 99.5% reduction |
| Mean foot penetration, m (raw -> post) | 0.8449 | 0.0003 | 100.0% reduction |
| Mean stance foot speed, m/s (world-grounded -> post) | 0.3793 | 0.2710 | 28.7% reduction |
| Sliding contact samples (world-grounded -> post) | 341.7037 | 65.5185 | 79.5% reduction |
| Root vertical jitter (world-grounded -> post) | 0.000750 | 0.000664 | 9.9% reduction |

## Training Suitability

- Frame count was preserved on 27/27 clips.
- Upper-body pose was preserved on 27/27 clips; this protects the stroke action while allowing lower-body correction.
- Lower-body pose delta stayed within the configured bound on 27/27 clips.
- Final reference bundles use AMASS-style z-up motion/contact data and have matching FPS metadata on 27/27 clips.
- Residual risk: max instantaneous contact speed is not improved on average; the training-oriented evidence is the lower mean stance speed, far fewer sliding samples, and all validation gates passing.

## Output Files

- `output/Optimization/forehand_clear_stage_metrics.csv`
- `output/Optimization/forehand_clear_sequence_summary.csv`
- `output/Optimization/forehand_clear_experiment_summary.json`
