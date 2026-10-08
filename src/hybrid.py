"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""

    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [min, max]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 OPTIONAL: groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   OPTIONAL: ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the Puzzle into the IR format JSON. Do not solve the puzzle.
Puzzle:
{text}

Encoding Rules:
1. Variable Domains:
    - All variable domains MUST be integers [0, N-1] or explicit integer sets;
    - Single integer values are FORBIDDEN;
    - String domains and symbolic values are FORBIDDEN;
    - Variable names must follow snake_case identifiers representing entities or attributes (e.g., meeting_kickoff, shift_alice).

2. Temporal Indexing:
    - Order days, slots, and positions sequentially from 0 to N-1;
    - For sub-day periods: day = slot // K, period = slot % K (where K is periods per day).

3. Formalization specs:
    - Next to:
        Immediately after: B == A + 1;
        Unordered adjacent: abs(A - B) == 1.

    - Between (B is between A and C):
        Fixed order: A < B and B < C;
        Any order: (A < B and B < C) or (C < B and B < A).

    - At most:
        Maximum value or position: A <= K;
        Distance limit: abs(A - B) <= K;
        Count limit: (cond1) + (cond2) <= 1.

    - Fewer:
        Strict less than: count_A < count_B;
        Exactly K fewer: count_A == count_B - K;
        At most K fewer: count_B - count_A <= K and count_A < count_B;
        Total count threshold: (cond1) + (cond2) + (cond3) < K.
        
    - At least:
        Minimum value or position: A >= K;
        Minimum distance: abs(A - B) >= K;
        Minimum count: (cond1) + (cond2) + (cond3) >= K.

5. constraints usage rules:
    - A statement S made by person X MUST be encoded as: X == (S);
    - Declare mutual exclusivity via "all_different" groups rather than multiple != clauses.

{ir_spec}

{feedback}
"""


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def find_symbols(data):
    symbols = set()
    if isinstance(data, dict):
        for k, v in data.items():
            symbols.update(find_symbols(v))
    elif isinstance(data, list):
        for item in data:
            symbols.update(find_symbols(item))
    elif isinstance(data, str):
        symbols.add(data)
    return symbols


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid.

    TODO(team). Things to check:
      * every domain is [lo, hi] with lo <= hi, or {"values": [...]} with integers;
      * every name used in all_different / constraints / objective is a declared variable;
      * every constraint parses and uses only allowed syntax.
    Hint: ir_solver.build(ir) raises ValueError with a readable message for unknown variables and
    unsupported syntax - you can call it inside try/except and turn the exception into an error message.
    """

    errors = []
    variables = []

    # VARIABLES CHECK
    for k, v in ir["variables"].items():
        variables.append(k)

        if isinstance(v, dict):
            vals = v.get("values")
            if not isinstance(vals, list) or not vals:
                errors.append(
                    f"Set '{k}' to a non-empty list of discrete integers, e.g. '{k}': {{\"values\": [0, 1, 2]}}."
                )
            elif not all(isinstance(x, int) for x in vals):
                non_ints = [x for x in vals if not isinstance(x, int)]
                errors.append(f"Convert all elements of '{k}' values {non_ints} into integer indices.")

        elif isinstance(v, list) and all(isinstance(x, int) for x in v):
            if len(v) == 2:
                if v[0] > v[1]:
                    errors.append(f"Swap bounds for '{k}' so lower bound comes first: [{v[1]}, {v[0]}].")
            elif len(v) == 1:
                errors.append(
                    f"Change '{k}' from single-element list {v} to discrete format: '{k}': {{\"values\": {v}}}."
                )
            else:
                errors.append(
                    f"Format '{k}' as continuous range [{min(v)}, {max(v)}] or discrete list '{k}': {{\"values\": {v}}}."
                )

        else:
            errors.append(
                f"Define '{k}' as either an integer range [lo, hi] or discrete values '{k}': {{\"values\": [v1, v2, ...]}}."
            )

    # USAGE
    uncreated_vars = []

    for l in ir.get("all_different", []):
        if len(l) > 0:
            for v in l:
                if v not in variables:
                    uncreated_vars.append(v)

    if uncreated_vars:
        errors.append(f"Add variables to 'variables': {', '.join(uncreated_vars)} ")

    # SANITY CHECK: PIGEONHOLE PRINCIPLE IN ALL_DIFFERENT
    for group in ir.get("all_different", []):
        valid_members = [v for v in group if v in ir.get("variables", {})]
        if len(valid_members) <= 1:
            continue

        all_possible_vals = set()
        for v in valid_members:
            dom = ir["variables"][v]
            if isinstance(dom, list) and len(dom) == 2 and dom[0] <= dom[1]:
                all_possible_vals.update(range(dom[0], dom[1] + 1))
            elif isinstance(dom, dict) and isinstance(dom.get("values"), list):
                all_possible_vals.update(dom["values"])

        if len(valid_members) > len(all_possible_vals):
            errors.append(
                f"Unsatisfiable 'all_different' constraint: {len(valid_members)} variables {valid_members} "
                f"compete for only {len(all_possible_vals)} distinct values ({sorted(list(all_possible_vals))}). "
                f"Remove this group from 'all_different'."
            )

    try:
        a = ir_solver.build(ir)
    except ValueError as e:
        errors.append(str(e))
    except:
        pass

    return errors


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}.

    TODO(team): with limit=1 the solver stops after the first solution, so it reports "unique" even when
    several solutions exist. Think about what limit you need to tell "unique" from "multiple".
    """

    # print(ir)

    return ir_solver.solve(ir, limit=1)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format.

    Provided baseline: the model translates the assignment. This is a weak point - the model can
    introduce errors AFTER the solver. Status and objective are always taken from the solver, never
    from the model. Extension: write deterministic code for problem types where the mapping is obvious.
    """
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(
        text=problem["text"],
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
        answer_format=problem["answer_format"],
    )
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        ir = formalize(problem, feedback="\n".join(errors))
        errors = validate_ir(ir)

        if errors:
            trace["attempts"].append({"ir": ir, "errors": errors})

    try:
        result = run_solver(ir)
    except ValueError as e:  # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {
        "status": result["status"],
        "n_solutions": len(result["solutions"]),
        "objective": result.get("objective"),
        "first_solution": result["solutions"][0] if result["solutions"] else None,
    }

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
