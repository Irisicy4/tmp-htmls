modal app stop deprivacy-100-bench
modal deploy scripts_batch/deprivacy_modal_runner.py

python scripts_batch/trigger_deprivacy.py --agent codex --run-tag deprivacy-codex-v4
modal app logs deprivacy-100-bench

rm -rf results/deprivacy/deprivacy-codex-v4
modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy-codex-v4
python scripts_batch/analyze_cross_agent.py results/deprivacy/deprivacy-codex-v4/

#compare

# 1. Stop lingering apps                                                                                                   
modal app stop deprivacy-100-bench
modal app list | grep "hm-" | awk '{print $2}' | xargs -I {} modal app stop {}    
modal app list | grep "dp-" | awk '{print $2}' | xargs -I {} modal app stop {}                                                                                           
                                                                                                                            
# 2. Deploy
modal deploy scripts_batch/deprivacy_modal_runner.py                                                                       
                                                                
# 3. Run (pick one)                                                                                                        
python scripts_batch/trigger_deprivacy.py --agent codex --tasks-dir real_118 --run-tag real118-codex
python scripts_batch/trigger_deprivacy.py --agent claude-code --tasks-dir real_118 --run-tag real118-claude
python scripts_batch/trigger_deprivacy.py --agent codex --tasks-dir synthetic_382 --run-tag synthetic_382-codex
python scripts_batch/trigger_deprivacy.py --agent claude-code --tasks-dir synthetic_382 --run-tag synthetic_382-claude
                                                                                                                            
# 4. Watch logs                                                 
modal app logs deprivacy-100-bench                                                                                         
                                                                
# 5. Check if done
modal volume ls deprivacy-100-results real118-codex/ | grep "_summary.json" && echo "DONE" || echo "Still running. Task results so far:"
modal volume ls deprivacy-100-results real118-codex/ | grep -c "\.json"

modal volume ls deprivacy-100-results real118-claude/ | grep "_summary.json" && echo "DONE" || echo "Still running. Task results so far:"
modal volume ls deprivacy-100-results real118-claude/ | grep -c "\.json"

modal volume ls deprivacy-100-results synthetic_382-codex/ | grep "_summary.json" && echo "DONE" || echo "Still running. Task results so far:"
modal volume ls deprivacy-100-results synthetic_382-codex/ | grep -c "\.json"

modal volume ls deprivacy-100-results synthetic_382-claude/ | grep "_summary.json" && echo "DONE" || echo "Still running. Task results so far:"
modal volume ls deprivacy-100-results synthetic_382-claude/ | grep -c "\.json"
                                                                                                                            
# 6. Collect
rm -rf results/real118-codex                                                                                               
modal run scripts_batch/deprivacy_modal_runner.py --collect real118-codex                                                  
rm -rf results/real118-claude                                                                                               
modal run scripts_batch/deprivacy_modal_runner.py --collect real118-claude
rm  -rf results/synthetic_382-codex                                                                                             
modal run scripts_batch/deprivacy_modal_runner.py --collect synthetic_382-codex
rm  -rf results/synthetic_382-claude                                                                                             
modal run scripts_batch/deprivacy_modal_runner.py --collect synthetic_382-claude


# 7. Analyze                                                                                                               
python scripts_batch/analyze_cross_agent.py results/real118/
python scripts_batch/analyze_cross_agent.py results/synthetic_382/


curl -s "https://api.uniapi.io/claude/v1/messages" -H "Content-Type: application/json" -H "x-api-key: $(grep ANTHROPIC_API_KEY .env | cut -d= -f2)" -H "anthropic-version: 2023-06-01" -d '{"model":"claude-sonnet-4-20250514","max_tokens":50,"messages":[{"role":"user","content":"Say hi"}]}'