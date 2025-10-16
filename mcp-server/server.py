from fastmcp import FastMCP
from typing import Annotated


mcp = FastMCP(
    "local-mcp-server",
)


@mcp.tool
def process_image(
    image_url: Annotated[str, "URL of the image to process"],
    resize: Annotated[bool, "Whether to resize the image"] = False,
    width: Annotated[int, "Target width in pixels"] = 800,
    format: Annotated[str, "Output image format"] = "jpeg",
) -> dict:
    """Process an image with optional resizing."""
    # Implementation...
    return {}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
