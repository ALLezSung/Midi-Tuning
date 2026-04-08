import os
os.environ['MODELSCOPE_DOMAIN'] = 'www.modelscope.cn'
#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('vllm-ascend/Llama-2-7b-hf', cache_dir = '/root/autodl-tmp', revision = 'master')