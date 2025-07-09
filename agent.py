import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def run_agent(input: str) -> str:
    print("hello")

if __name__ == "__main__":
    asyncio.run(run_agent("hello"))