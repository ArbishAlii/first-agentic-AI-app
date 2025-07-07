from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
# gemini api key get from ---> aistudio.google.com

# external key defines API key nd fetch key
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# important to define base url if using third party api
# base url gets from "google generative base url/ All methods AI(google developers)/Gemini SI Docs/Open AI compatibility"

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
    # this tracing step could be done globally too(by importing). This is use to check if API has used or not
)

agent = Agent(
    name="Frontend Developer",
    instructions="Expert Front end Developer"
)


@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello From Arbish! How can I Assist You Today?").send()

# @cl.on_chat_start ---- this only works when we are on chat, then it will remove. This is working Using "History"


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    # Pass structured history (list of dicts), not just a string
    result = await Runner.run(
        agent,
        # input=message.content,
        # this only shows the chat with the user but doesn't call the last msg b/c it is not storing the memory

        input=history,
        # while this shows the last msg too b/c we have append history (with the user)
        run_config=config
    )

    # Directly stream the final output (not duplicated)
    await msg.stream_token(result.final_output)

    # we have created the agent memory for the user, but it doesn't call its own memory so appending....
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # @cl.on_message ---- this is the decorator
    # send is used for the output, means it will accept the input
    # message.content is the input we have sent
    # we are await to wait for the output, while "(content=result.final_output)" this is the final output
    # .send --> sends the final output (not needed again here)
    