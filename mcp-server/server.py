from mcp.server.fastmcp import FastMCP

# Create a new FastMCP instance
# mcp = FastMCP(
#     "local-mcp-server",
#     ssl_keyfile="/Users/yhong3/Documents/Research/Software Security/working_repo/mcp-agent/key.pem",
#     ssl_certfile="/Users/yhong3/Documents/Research/Software Security/working_repo/mcp-agent/cert.pem",
# )


mcp = FastMCP(
    "local-mcp-server",
)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def main():
    mcp.run(transport="streamable-http")
    # mcp.run(transport="sse", mount_path="/mcp")


if __name__ == "__main__":
    main()
