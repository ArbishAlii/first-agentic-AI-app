from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
#gemini api key get from ---> aistudio.google.com


# external key defines API key nd fetch key
external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
#important to define base url if using third party api
#base url gets from "google generative base url/ All methods AI(google developers)/Gemini SI Docs/Open AI compatibility"

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(
    model=model,
    model_provider = external_client,
    tracing_disabled = True
    #this tracing step could be done globally too(by importing). This is use to check if APi has used or not
)
agent= Agent(
    name="Frontend Developer",
    instructions= "Expert Front end Developer"

)
result = Runner.run_sync(
    agent,
    input="Hey, How are you",
    run_config = config
)
print(result.final_output)