from openai import OpenAI
import json
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import Dict, List
import asyncio
from mcp_client import MCPClient


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

DOMAIN = "airline"


class ReActAgent:

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.mcp_client = MCPClient()
        self.tools = []
        self.initialize()

    def initialize(self):
        self.history = []
        asyncio.run(self.mcp_client.initialize())
        self.tools = self.mcp_client.list_OPENAI_tools()
        print(f"Initialized with {len(self.tools)} tools.")
        print("Tools:")
        print(json.dumps(self.tools, indent=2))

    def _get_domain_policy(self) -> str:
        domain_policy_file = f"tau2/domains/{DOMAIN}/policy.md"
        with open(domain_policy_file, "r") as f:
            domain_policy = f.read().strip()
        return domain_policy

    @property
    def system_prompt(self) -> str:
        system_prompt_template = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()
        domain_policy = self._get_domain_policy()
        return system_prompt_template.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=domain_policy,
        )

    def ReAct_loop(self, user_input: str) -> Dict:
        self.history.append({"role": "user", "content": user_input})
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.history

        while True:
            messages = [
                {"role": "system", "content": self.system_prompt},
            ] + self.history
            response: ChatCompletionMessage = self._call_LLM(messages, self.tools)

            if response.tool_calls:
                self.history.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                )
                for tool_call in response.tool_calls:
                    tool_call: ChatCompletionMessageFunctionToolCall
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    tool_response = asyncio.run(
                        self.mcp_client.call_tool(tool_name, tool_args)
                    )
                    print(
                        f"Tool call: {tool_name} with args {tool_args} returned {tool_response}"
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "content": tool_response.text,
                            "tool_call_id": tool_call.id,
                        }
                    )
            else:
                self.history.append({"role": "assistant", "content": response.content})
                return response.content

    def _call_LLM(self, messages: list, tools: List) -> Dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=tools,
        )
        return response.choices[0].message


def main():
    agent = ReActAgent()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.ReAct_loop(user_input)
        print("Agent Response: ", response)


if __name__ == "__main__":
    main()
