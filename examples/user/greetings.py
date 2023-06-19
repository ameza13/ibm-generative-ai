import os

from dotenv import load_dotenv

from genai.model import Credentials, Model
from genai.schemas import GenerateParams, ModelType

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>

load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

print("\n------------- Example (Sync Greetings)-------------\n")

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=10,
    min_new_tokens=1,
    stream=False,
    temperature=0.7,
    top_k=50,
    top_p=1,
)

creds = Credentials(api_key, api_endpoint)
model = Model(ModelType.FLAN_UL2, params=params, credentials=creds)

input = "Hello! How are you?"

for result in model.generate([input]):
    if result is not None:
        print(f"\t {result.generated_text}")
    else:
        print("\t <the result was 'None' for this input>")
