"""EEG 分析模块包"""
from .flow_recovery import (
    load_eeg, load_events, preprocess,
    compute_band_powers, compute_entropy, extract_features,
    compute_recovery_time, compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
    generate_sample_eeg, events_to_df, run_full_pipeline,
    BANDS,
)

__all__ = [
    'load_eeg', 'load_events', 'preprocess',
    'compute_band_powers', 'compute_entropy', 'extract_features',
    'compute_recovery_time', 'compute_all_recovery', 'compute_attenuation',
    'paired_t_test', 'repeated_measures_anova', 'pearson_correlation',
    'generate_sample_eeg', 'events_to_df', 'run_full_pipeline',
    'BANDS',
]
