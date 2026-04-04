# 1. Activate the env                                                                                                                            
conda activate evolvebench                                                                                                                     
                                                                                                                                                
# 2. Go to repo                                                                                                                                  
cd /Users/icy/Desktop/act/2020604-evolvebench/evolve_bench_harbor                                                                                
                                                                                                                                                
# 3. Setup Modal (one-time — opens browser to login)                                                                                             
modal setup                                                                                                                                      
                                                                                                                                                
# 4. Fill in your .env                                                                                                                           
cp env.template .env                                                                                                                             
# Then edit .env and add your keys:                                                                                                              
#   OPENAI_API_KEY=sk-...                                                                                                                        
#   ANTHROPIC_API_KEY=sk-ant-...                                                                                                                 
#   GEMINI_API_KEY=...                                                                                                                           
                                                                                                                                                
# 5. Deploy the Modal app                                                                                                                        
modal deploy scripts_batch/deprivacy_modal_runner.py      

python scripts_batch/trigger_deprivacy.py --agent codex --task-names task-01-im-looking-for-backpack-under --run-tag test-v4-codex
                                                                                                                                                
# 6. Run the mini test (1 task x 3 agents)                                                                                                       
bash scripts_batch/test_mini.sh

#If you want to see deeper (check if specific agents have finished):                                                                              
   
modal volume ls deprivacy-100-results test-v2/claude-code/                                                                                       
modal volume ls deprivacy-100-results test-v2/codex/                                                                                             
modal app logs deprivacy-100-bench              
                                                                                                                                                
# 7. Wait ~10-20 min, then collect + analyze                                                                                                     
bash scripts_batch/test_mini_collect.sh     

# See if any results have landed on the volume yet                                                                                               
modal run scripts_batch/deprivacy_modal_runner.py --list-runs                                                                                    
                                                                                                                                                
# Check running functions on Modal dashboard                                                                                                     
modal app list                                                                                                                                   
                            