from dimos.core.coordination.blueprints import autoconnect

from airtight.dimos_lane.hello import SiteStatusModule

airtight_site = autoconnect(SiteStatusModule.blueprint())
