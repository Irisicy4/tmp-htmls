modal app stop deprivacy-100-bench  
modal deploy scripts_batch/deprivacy_modal_runner.py

python scripts_batch/trigger_deprivacy.py --agent codex --run-tag deprivacy-codex-v4
modal app logs deprivacy-100-bench

rm -rf results/deprivacy/deprivacy-codex-v4                                                                                
modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy-codex-v4
python scripts_batch/analyze_cross_agent.py results/deprivacy/deprivacy-codex-v4/

