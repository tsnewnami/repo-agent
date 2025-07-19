from huggingface_hub import HfApi
import dotenv
import os

dotenv.load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./data/",
    repo_id="JamesSED/synthetic_QA_code_search_net",
    repo_type="dataset",
)