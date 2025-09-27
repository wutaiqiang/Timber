from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask



with read_base():
    # from opencompass.configs.summarizers.chat_core import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################

    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets


# repeat 4 times
for k,v in list(locals().items()):
    if k.endswith('_datasets'):
        v[0]["n"] = 4
    
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'weights/Qwen3-0.6B'

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


#! Replace your path here 
paths = [
    "weights/Qwen3-0.6B/Qwen3-0.6B",
]

Instruct_settings = []
for p in paths:
    n = p.split("/")[-1]
    Instruct_settings.append((n,p))

print(Instruct_settings)


models = []

# set tp and num_gpus
for abbr, path in Instruct_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=47104, max_batch_size=4096, tp=1),
            gen_config=dict(do_sample=True, temperature=0.6, top_k=20, top_p=0.95, max_new_tokens=38912, mini_p=0, enable_thinking=True),
            max_seq_len=8192,
            max_out_len=38912,
            batch_size=2048,
            run_cfg=dict(num_gpus=1),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ) 

models = models