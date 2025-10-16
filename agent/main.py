import asyncio
import os

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.metadata_utils import get_display_name

server_params = StdioServerParameters(
    command="start-mcp-server",
)


def serialize_schema(schema: dict) -> str:
    """Serialize a JSON schema to a human-readable string."""
    if not schema:
        return "No input"
    if schema.get("type") == "object" and "properties" in schema:
        props = schema["properties"]
        required = schema.get("required", [])
        lines = []
        for prop, details in props.items():
            line = f"- {prop} ({details.get('type', 'any')})"
            if prop in required:
                line += " [required]"
            if "description" in details:
                line += f": {details['description']}"
            lines.append(line)
        return "\n".join(lines)
    return str(schema)


async def run():
    """Run the completion client example."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            for tool in tools.tools:
                # get_display_name() returns the title if available, otherwise the name
                display_name = get_display_name(tool)
                print(f"Tool: {display_name}")
                if tool.description:
                    print(f"  {tool.description}")
                # input schema
                if tool.inputSchema:
                    print("  Input:")
                    print(
                        "    " + serialize_schema(tool.inputSchema).replace("\n", "\n    ")
                    )
                if tool.outputSchema:
                    print("  Output:")
                    print(
                        "    " + serialize_schema(tool.outputSchema).replace("\n", "\n    ")
                    )


def main():
    """Entry point for the completion client."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
