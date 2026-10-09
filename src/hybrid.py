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
import re
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
String and symbolic values are FORBIDDEN: {{values: ["A", "B"]}}, ["A", "B"].
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces, e.g., meeting_kickoff, role_alice."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON. You arent a programmer just translator."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the Puzzle into the IR format JSON. Do not solve the puzzle. Output ONLY valid JSON matching the IR specification.

Puzzle:
{text}

{ir_spec}

Encoding Puzzle Rules:

1. Logic & Arithmetic:
   - Use relational operators (<, <=, >, >=, +, -, //, %) instead of enumerating concrete cases.
   - If a TOTAL VALUE is given, constrain the sum: (A + B + C) == TOTAL_VALUE.

2. Variable Mapping:
   - Map entities to attribute values (Entity -> Slot/Day/Role), never the reverse.

3. Domains & Types:
   - Single value: {{"values": [v1]}}
   - Explicit set: {{"values": [v1, v2, ..., vN]}}
   - Contiguous range: [0, N-1]

4. Roles & Truth Values:
   - Truth/role domain is always binary: [0, 1] (0 = false/knave, 1 = true/knight).
   - Statement S by person X: X == (S).
   - Never add truth/role variables to "all_different".

5. Time & Calendar Indexing:
   - Order days and positions chronologically: 0 to N-1.
   - Flatten sub-day slots: slot = day * K + period (K = periods per day).
     Example (K=2): 0: Mon AM, 1: Mon PM, 2: Tue AM, 3: Tue PM, ...
   - Slot deconstruction:
     * Morning: A % K == 0
     * Afternoon: A % K == 1
     * Same day: A // K == B // K
     * Day after: B // K == (A // K) + 1
     * Earlier day: A // K < B // K

6. Exclusivity ("all_different"):
   - Group mutually exclusive entities in a single list: [["A", "B", "C"]].
   - Groups must contain at least 2 variables; never emit single-variable lists.

{feedback}
"""


def is_binary_domain(domain) -> bool:
    if isinstance(domain, list) and len(domain) == 2 and domain[0] == 0 and domain[1] == 1:
        return True
    if isinstance(domain, dict):
        vals = set(domain.get("values", []))
        return vals.issubset({0, 1}) and len(vals) > 0
    return False


def sanitize_all_different_binary_vars(ir: dict) -> list:
    errors = []
    variables = ir.get("variables", {})
    all_diff = ir.get("all_different", [])

    if not isinstance(all_diff, list):
        return errors

    cleaned_all_different = []

    for idx, group in enumerate(all_diff):
        if not isinstance(group, list):
            continue

        binary_vars = [v for v in group if v in variables and is_binary_domain(variables[v])]

        if len(binary_vars) > 1:
            errors.append(f"Remove 'all_different' from previous IR.")
            filtered_group = [v for v in group if v not in binary_vars]
            if len(filtered_group) > 1:
                cleaned_all_different.append(filtered_group)
        else:
            cleaned_all_different.append(group)

    return errors


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = (
        f"\nPREVIOUS ATTEMPT REJECTED BY VALIDATOR:\n"
        f"{feedback}\n\n"
        f"CRITICAL REPAIR INSTRUCTIONS:\n"
        f"Read each error message above and fix or follow the instruction bellow \n"
        if feedback
        else ""
    )
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


def check_all_different_group_sizes(ir: dict) -> list:
    errors = []
    all_diff = ir.get("all_different", [])

    if not isinstance(all_diff, list):
        errors.append("'all_different' must be a list of lists.")
        return errors

    for idx, group in enumerate(all_diff):
        if not isinstance(group, list):
            errors.append(f"Group at index {idx} in 'all_different' is not a list: {group}")
        elif len(group) == 1:
            errors.append(
                f"Single-element list found in 'all_different' at index {idx}: {group}. "
                f"'all_different' groups must contain at least 2 distinct variables, "
                f"or group all mutually exclusive variables together: [{group[0]}, ...]."
            )
        elif len(group) == 0:
            errors.append(f"Empty list found in 'all_different' at index {idx}. Remove empty groups.")

    return errors


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
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

    def is_valid_variable_name(name: str) -> bool:
        return bool(re.match(pattern, name))

    def sanitize_variable_name(raw_name: str) -> str:
        cleaned = raw_name.strip()
        cleaned = re.sub(r"[-\s.]+", "_", cleaned)
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "", cleaned)
        if not cleaned:
            cleaned = "var"
        elif cleaned[0].isdigit():
            cleaned = "_" + cleaned
        return cleaned

    # VARIABLES CHECK
    for k, v in ir["variables"].items():
        variables.append(k)

        if not is_valid_variable_name(k):
            errors.append(f"Rename variable '{k}' to '{sanitize_variable_name(k)}' to make it a valid identifier.")

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
                    errors.append(f"Swap bounds for '{k}' variable ONLY so lower bound comes first: [{v[1]}, {v[0]}].")
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

    _err = check_all_different_group_sizes(ir)
    if _err:
        errors.append(*_err)

    _err = sanitize_all_different_binary_vars(ir)
    if _err:
        errors.append(*_err)

    # try:
    #     a = ir_solver.build(ir)
    # except ValueError as e:
    #     errors.append(str(e))
    # except:
    #     pass

    print(errors)
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
