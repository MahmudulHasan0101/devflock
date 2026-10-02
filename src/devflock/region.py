"""The `.devflock` file that lives at the root of each region.

Format is plain Markdown with two headers DevFlock treats specially:
  ## Public       -- interface, exports, how to run this region's tests.
                     Other regions' workers are shown ONLY this section
                     when they ask about a region they don't own.
  ## Private       -- conventions, gotchas, recent changes, open questions.
                     Only this region's own worker sees it.

Anything outside those two headers (e.g. a title, a date) is kept but not
specially parsed. Workers are instructed to rewrite this file at the end of
every task; DevFlock does not try to diff/merge it, last write wins.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

PUBLIC_RE = re.compile(r"^##\s*Public\s*$", re.MULTILINE)
PRIVATE_RE = re.compile(r"^##\s*Private\s*$", re.MULTILINE)

TEMPLATE = """\
# {region} — .devflock

## Public
_Nothing recorded yet. Describe this region's purpose, its exported
interfaces (functions/endpoints/types other regions may call), and how to
run its tests._

## Private
_Conventions, gotchas, and open questions for whoever (human or worker)
next picks up this region._
"""


@dataclasses.dataclass
class RegionMemory:
    region: str
    public: str
    private: str
    raw: str

    @classmethod
    def load_or_init(cls, region_dir: Path, region_name: str) -> "RegionMemory":
        f = region_dir / ".devflock"
        if not f.exists():
            raw = TEMPLATE.format(region=region_name)
            f.write_text(raw)
        else:
            raw = f.read_text()
        return cls.parse(region_name, raw)

    @classmethod
    def parse(cls, region: str, raw: str) -> "RegionMemory":
        pm = PUBLIC_RE.search(raw)
        prm = PRIVATE_RE.search(raw)
        public = private = ""
        if pm:
            end = prm.start() if prm and prm.start() > pm.start() else len(raw)
            public = raw[pm.end():end].strip()
        if prm:
            public_before = pm and pm.start() < prm.start()
            start = prm.end()
            end = len(raw)
            if not public_before:
                nxt = PUBLIC_RE.search(raw, prm.end())
                if nxt:
                    end = nxt.start()
            private = raw[start:end].strip()
        return cls(region=region, public=public, private=private, raw=raw)

    def render(self) -> str:
        return (f"# {self.region} — .devflock\n\n"
                f"## Public\n{self.public.strip()}\n\n"
                f"## Private\n{self.private.strip()}\n")

    def save(self, region_dir: Path):
        (region_dir / ".devflock").write_text(self.render())


def public_only(region_dir: Path, region_name: str) -> str:
    """What a WORKER FROM ANOTHER REGION is allowed to see about this one."""
    mem = RegionMemory.load_or_init(region_dir, region_name)
    return mem.public
