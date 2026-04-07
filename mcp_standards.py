# /// script
# dependencies = [
#   "mcp",
# ]
# ///

from mcp.server.fastmcp import FastMCP
import os

# Initialize FastMCP server
mcp = FastMCP("ProjectMastermind")

# USE YOUR ABSOLUTE PATH HERE
RULES_FILE = "/Users/pulkitpandey/Desktop/WarRoom/.project-standards.md"

@mcp.tool()
def get_project_rules() -> str:
    """Retrieve the current project architecture and coding standards."""
    if not os.path.exists(RULES_FILE):
        return f"No rules file found at {RULES_FILE}."
    with open(RULES_FILE, "r") as f:
        return f.read()

@mcp.resource("standards://current")
def current_standards() -> str:
    """The authoritative project standards resource."""
    return get_project_rules()

if __name__ == "__main__":
    mcp.run()