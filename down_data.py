
#https://huggingface.co/datasets/heyongxian/watermark_images/tree/main
#从这里下载数据集,之后解压到data目录下

from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/heyongxian/mark-pro/",
    repo_id="heyongxian/watermark_images",
    repo_type="dataset",
)