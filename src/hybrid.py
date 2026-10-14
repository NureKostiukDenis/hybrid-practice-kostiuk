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
  "variables": {"<name>": [min, max] for CONTINUOS VALUES or {"values": [a, b, c]} for EXPLIT SET or {"values": [a]} for ONLY ONE or [0, 1] }, integer variables and their domains
  "all_different": [["<name>", ...], ...], OPTIONAL: groups whose values must all differ
  "constraints": ["<expression>", ...], every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"} OPTIONAL: ONLY for optimization problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
String and symbolic values are FORBIDDEN.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces, e.g., meeting_kickoff, role_alice."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the Puzzle into the IR format JSON. Do not solve the puzzle.
Puzzle: {text}


{ir_spec}


1. Variable Domains & Mapping:
- Unknown entities: Full candidate range [0, total_slots - 1], calculated strictly from the current puzzle dimensions (never guess or copy from examples).
- Define ALL of entities, items, destinations, and categories as they appear in the puzzle text.
- Map every individual option mentioned in the puzzle to a single key in "variables".


2. True / False Puzzle Logic (Is A Attribute?, Is B Attribute?):
- Use binary variables: 0 = false, 1 = true.
- Dont use "all_different" field. 
- "A and B": A and B.
- "A or B": A or B.
- "Not A": not A.
- "If A, then B": not A or B.
- "A if and only if B": A == B.
- "A and B cannot both be true": A + B <= 1.
- "At least N conditions are true": A + B + ... >= N.
- "At most N conditions are true": A + B + ... <= N.
- "Exactly N conditions are true": A + B + ... == N.
- "Exactly one of A and B is true": A + B == 1.


3. Time,Slot, Week Puzzle (When?):
- Variable domains MUST strictly use the format [0, total_slots - 1].
- K represents periods per day (e.g., K = 2 for Morning and Afternoon).
- NEVER create dummy variables like: slot, total_slots, K, morning, afternoon inside "variables".
- Create variables for every entity/meeting/item listed in the puzzle.
- "First slot of the week": "variable == 0".
- "Last slot of the week": "variable == total_slots - 1".
- "Morning slot": "variable % K == 0".
- "Afternoon slot": "variable % K == 1".
- "Same day": "A // K == B // K".
- "Day after": "B // K == (A // K) + 1".
- "Earlier day": "A // K < B // K".


1. Example Logic Puzzle (What is order?):
Four books stand on a shelf in positions 1 to 4 from left to right: a dictionary, an atlas, a novel and a cookbook.
1. The atlas stands at one of the two ends of the shelf.
2. The novel stands immediately to the right of the dictionary.
3. The cookbook is not next to the atlas.
4. The atlas is somewhere to the left of the novel.
What is the order of the books?
Answer:
{{
"variables": {{
    "dictionary": [0, 4],
    "atlas": [0, 4],
    "novel": [0, 4],
    "cookbook": [0, 4]
}},
"all_different": [["dictionary", "atlas", "novel", "cookbook"]],
    "constraints": [
    "atlas == 0 or atlas == 4",
    "novel == dictionary + 1",
    "abs(cookbook - atlas) != 1",
    "(atlas < novel) == 1"
]
}}


2. Example Slots Puzzle (When? Which slot does fit?):
Six meetings — Planning, Design review, Budget, Hiring, Security review and Retro — take place from Monday to Wednesday. Each day has a morning slot and an afternoon slot, and each meeting takes one slot.
1. Retro is the last meeting of the week.
2. Planning and Design review are on the same day, with Planning first.
3. Budget is held in a morning slot.
4. Security review is held on the day after Budget.
5. Hiring is not on Monday.
6. Budget is not on the same day as Planning.
Which slot does each meeting take?
Answer:
{{
"variables": {{
    ...
}},
"all_different": [["planning","design_review","budget","hiring","security_review","retro"]],
"constraints": [
    "retro == 5",
    "planning // 2 == design_review // 2",
    "planning < design_review",
    "budget % 2 == 0",
    "security_review // 2 == (budget // 2) + 1",
    "hiring // 2 > 0",
    "budget // 2 != planning // 2"
]
}}


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

        if len(binary_vars) > 1 and ir.get("objective", None) is None:
            # errors.append('Remove "all_different": [...] field from previous IR JSON.')
            filtered_group = [v for v in group if v not in binary_vars]
            if len(filtered_group) > 1:
                cleaned_all_different.append(filtered_group)
        else:
            cleaned_all_different.append(group)

    return errors


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    fb = (
        f"\nPrevious attempt was rejected though errors.\nInstructions:\n"
        f"{feedback}\n\n"
        f"Read each instruction/error message above and fix/follow instruction\n"
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
    all_diff = ir.get("all_different", [])

    if not isinstance(all_diff, list):
        return ["Set 'all_different' to a list of lists."]

    invalid_groups = [g for g in all_diff if not isinstance(g, list) or len(g) <= 1]
    if not invalid_groups:
        return []

    collected_vars = [g[0] for g in all_diff if isinstance(g, list) and len(g) == 1]

    if len(collected_vars) > 1:
        merged_example = f'"all_different": [{collected_vars}]'
        return [f"Combine all mutually exclusive entities into one shared group: {merged_example}."]

    return [
        "Each list in 'all_different' must contain at least 2 distinct variables. "  # noqa: ISC004
        'Combine mutually exclusive variables into a single group like ["var1", "var2", ...] and remove empty or single-element groups.'
    ]


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
            errors.append(
                f'Replace \'{k}\': {{"values": {v}}} with \'{sanitize_variable_name(k)}\': {{"values": {v}}} in "variables".'
            )

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
                f"Define '{k}' as either an integer range [min, max] or discrete values '{k}': {{\"values\": [v1, v2, vN]}}."
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
        errors.extend(_err)

    _err = sanitize_all_different_binary_vars(ir)
    if _err:
        errors.extend(_err)

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
