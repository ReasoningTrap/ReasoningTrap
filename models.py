MODELS = {
    # model_id, max_len, temperature, top_p, top_k, repetition_penalty
    "qwen3_8b_think": ("qwen/qwen3-8b", 32768, 0.6, 0.95, -1, 1.0),
    "qwen3_8b_nothink": ("qwen/qwen3-8b", 16384, 0.7, 0.8, -1, 1.0),
    "qwen3_32b_think": ("qwen/qwen3-32b", 32768, 0.6, 0.95, -1, 1.0),
    "qwen3_32b_nothink": ("qwen/qwen3-32b", 16384, 0.7, 0.8, -1, 1.0),
    "qwq": ("Qwen/QwQ-32B", 32768, 0.6, 0.95, -1, 1.0),
    "qwen2.5_1.5b_instruct": ("Qwen/Qwen2.5-1.5B-Instruct", 32768, 0.6, 0.95, -1, 1.0),
    "qwen2.5_7b_instruct": ("Qwen/Qwen2.5-7B-Instruct", 32768, 0.6, 0.95, -1, 1.0),
    "qwen2.5_14b_instruct": ("Qwen/Qwen2.5-14B-Instruct", 32768, 0.6, 0.95, -1, 1.0),
    "qwen2.5_32b_instruct": ("Qwen/Qwen2.5-32B-Instruct", 32768, 0.6, 0.95, -1, 1.0),
    'deepscale': ("agentica-org/DeepScaleR-1.5B-Preview", 16384, 0.5, 0.9, -1, 1.0),
    'still': ("RUC-AIBOX/STILL-3-1.5B-preview", 32768, 0.6, 0.95, -1, 1.0),
    'skyt1': ("NovaSky-AI/Sky-T1-32B-Preview", 32768, 0.7, 0.9, 20, 1.05),
    "thinkprm": ("launch/ThinkPRM-14B", 32768, 0.6, 0.95, -1, 1.0),
    "orz_7b": ("Open-Reasoner-Zero/Open-Reasoner-Zero-7B", 32768, 0.7, 0.95, 50, 1.0),
    "orz_32b": ("Open-Reasoner-Zero/Open-Reasoner-Zero-32B", 32768, 0.7, 0.95, 50, 1.0),
    "azr_7b": ("andrewzh2/Absolute_Zero_Reasoner-Base-7b", 32768, 0.7, 0.8, 20, 1.05),
    "azr_14b": ("andrewzh2/Absolute_Zero_Reasoner-Base-14b", 32768, 0.7, 0.8, 20, 1.05),
    "satori": ("Satori-reasoning/Satori-7B-Round2", 8192, 0.7, 0.95, 50, 1.0),
    "countback": ("yangxw/Llama-3.2-1B-countdown-backtrack", 32768, 0.6, 0.9, -1, 1.0),
    "rmr1": ("gaotang/RM-R1-Qwen2.5-Instruct-7B", 32768, 0.7, 0.95, 50, 1.0),
    "prime_sft": ("PRIME-RL/Eurus-2-7B-SFT", 4096, 0.5, 0.9, -1, 1.0),
    "prime_rl": ("PRIME-RL/Eurus-2-7B-PRIME", 4096, 0.5, 0.9, -1, 1.0),
    "llama_1b": ("meta-llama/Llama-3.2-1B", 32768, 0.6, 0.9, -1, 1.0),
    "openr1_7b": ("open-r1/OpenR1-Qwen-7B", 32768, 0.6, 0.95, -1, 1.0),
    }