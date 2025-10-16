from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-5",
    tools=[
        {
            "type": "mcp",
            "server_label": "local_mcp",
            "server_description": "A local MCP server for math operations",
            "server_url": "http://127.0.0.1:8000/",
            "require_approval": "never",
        },
    ],
    input="Add 2 and 3, then multiply the result by 4.",
)

print(resp.output_text)
