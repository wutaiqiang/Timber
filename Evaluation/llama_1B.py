from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask



with read_base():
    # from opencompass.configs.summarizers.chat_core import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import math_datasets as minerva_math_datasets # minerva_math

    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets 

    # from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets


# repeat 4 times
for k,v in list(locals().items()):
    if k.endswith('_datasets'):
        v[0]["n"] = 4
    
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'weights/Llama1B'

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel

#! Replace your path here 
paths = [
    "weights/Llama-3.2-1B/Llama-3.2-1B-B-I-g02",
]

Instruct_settings = []
for p in paths:
    n = p.split("/")[-1]
    Instruct_settings.append((n,p))

print(Instruct_settings)


models = []

for abbr, path in Instruct_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=8192, max_batch_size=4096, tp=8),
            gen_config=dict(do_sample=True, temperature=0.6, top_p=0.9, max_new_tokens=4096, mini_p=0),
            max_seq_len=4096,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=8)
        )
    )    

models = models