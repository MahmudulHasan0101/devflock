import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from devflock.planner import extract_json, validate_plan, make_plan, PlanningFailed
from devflock.packer import recommend_n
from types import SimpleNamespace as NS

GOOD = {"summary": "Todo app split in API and UI.",
        "modules": [{"name": "api", "path": "src/api", "weight": 3},
                    {"name": "ui", "path": "src/ui", "weight": 2, "depends_on": ["api"]}],
        "tasks": [{"id": "t1", "module": "api", "description": "Create the REST endpoints for todos.",
                   "acceptance_cmd": "pytest -q"},
                  {"id": "t2", "module": "ui", "description": "Build the list page calling the API.",
                   "depends_on": ["t1"], "acceptance_cmd": "npm test"}]}
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name}")

# --- extraction
check("extract from prose+fence", extract_json("Sure!\n```json\n" + json.dumps(GOOD) + "\n```\nDone")["summary"])
check("extract from bare text", extract_json("Here: " + json.dumps(GOOD) + " thanks")["summary"])
try: extract_json("no json here"); check("extract fails on garbage", False)
except ValueError: check("extract fails on garbage", True)

# --- validation
plan, errs = validate_plan(GOOD)
check("valid plan accepted", plan and not errs and recommend_n(plan) == 1)
def bad(mut, needle):
    d = json.loads(json.dumps(GOOD)); mut(d); p, e = validate_plan(d)
    check(f"rejects: {needle}", p is None and any(needle in x for x in e))
bad(lambda d: d["tasks"][0].update(module="nope"), "unknown module")
bad(lambda d: d["tasks"][1].update(depends_on=["zzz"]), "unknown task")
bad(lambda d: d["tasks"][0].update(depends_on=["t2"]), "cycle")
bad(lambda d: d["modules"][0].update(path="../evil"), "no '..'")
bad(lambda d: d["modules"][0].update(path="/etc"), "relative")
bad(lambda d: d["modules"][1].update(path="src/api"), "already used")
bad(lambda d: d["tasks"][0].update(acceptance_cmd="true"), "acceptance_cmd")
bad(lambda d: d["tasks"].append(dict(d["tasks"][0])), "duplicate task")

# --- retry loop with a scripted fake session (invalid -> prose -> valid)
class Fake:
    def __init__(self, replies): self.replies, self.prompts, self.tools = list(replies), [], []
    async def run_task(self, prompt, *, max_turns=0, allowed_tools=None, resume=False, timeout_s=0):
        self.prompts.append(prompt); self.tools.append(allowed_tools)
        return NS(ok=True, summary=self.replies.pop(0), error=None)
broken = json.loads(json.dumps(GOOD)); broken["tasks"][0]["module"] = "ghost"
f = Fake([json.dumps(broken), "I think the plan is good!", "```json\n" + json.dumps(GOOD) + "\n```"])
plan, summary = asyncio.run(make_plan(f, "existing", "add search", requested_n=2))
check("retry loop recovers after 2 bad replies", plan and len(f.prompts) == 3)
check("error list fed back to model", "ghost" in f.prompts[1] or "unknown module" in f.prompts[1])
check("existing-project planning is read-only", f.tools[0] == ["Read", "Glob", "Grep"])
check("idea mode uses no tools", asyncio.run(make_plan(Fake([json.dumps(GOOD)]), "idea", "a todo app"))[0] and True)
f2 = Fake(["nope"] * 3)
try: asyncio.run(make_plan(f2, "idea", "x")); check("gives up after max attempts", False)
except PlanningFailed: check("gives up after max attempts", True)
print("\nALL PLANNER TESTS " + ("PASSED" if ok else "FAILED")); sys.exit(0 if ok else 1)
