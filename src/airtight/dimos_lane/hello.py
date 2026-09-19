from __future__ import annotations

from importlib import resources

from dimos.agents.annotation import skill
from dimos.core.module import Module

from airtight.contracts import Site


def load_example_site() -> Site:
    text = resources.files("airtight.contracts.examples").joinpath("site.json").read_text()
    return Site.model_validate_json(text)


class SiteStatusModule(Module):
    """Smallest possible airtight module: proves entry-point discovery and skill exposure over MCP."""

    @skill
    def site_status(self) -> str:
        """Describe the loaded site: its name, entry points and docks."""
        site = load_example_site()
        entries = ", ".join(e.id for e in site.entry_points)
        return f"site {site.name}: {len(site.entry_points)} entry points ({entries}), {len(site.docks)} docks, response time {site.response_time_s:.0f}s"
