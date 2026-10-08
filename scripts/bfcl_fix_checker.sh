#!/usr/bin/env bash
set -eu
PY=${PY:-python3}
: "${BFCL_SRC:?set BFCL_SRC to the berkeley-function-call-leaderboard dir}"
F="$BFCL_SRC/bfcl_eval/eval_checker/ast_eval/ast_checker.py"
MARK="$BFCL_SRC/bfcl_eval/eval_checker/ast_eval/.rapt_checker_fix"
VERSION=rapt-checker-fix-1
[ -s "$F" ] || { echo "FATAL: no ast_checker.py at $F" >&2; exit 1; }
if grep -q "_RAPT_CHECKER_FIX" "$F"; then
  printf '%s\n' "$VERSION" > "$MARK"; echo "already fixed: $F"; exit 0
fi
if grep -q "BFCL_SIMPLE_GOLD\|BFCL_DUP_PICK" "$F"; then
  echo "FATAL: $F carries the retired env-gated patches. Restore it (git checkout, or the .orig beside" >&2
  echo "       it) and re-run: the two fixes are now unconditional and must not stack." >&2
  exit 1
fi
cp -n "$F" "$F.orig" 2>/dev/null || true
"$PY" - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()

old_find = '''def find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    else:
        # it is a dict, there is only one function
        return func_descriptions'''
new_find = '''_RAPT_CHECKER_FIX = 1


def find_description(func_descriptions, name):
    """EVERY declaration of `name`, in declaration order -- not the first.

    A perturbation may declare one name twice, and the model is shown both declarations, so a call the
    interface sanctions under either one is a call the interface sanctions. Returning only the first made
    which declaration was validated against a fact about declaration order. simple_function_checker takes
    the list and accepts any declaration that validates; with one declaration the list has one element and
    the behaviour is unchanged.
    """
    if type(func_descriptions) == list:
        matches = [f for f in func_descriptions if f["name"] == name]
        return matches or None
    else:
        # it is a dict, there is only one function
        return func_descriptions'''
assert s.count(old_find) == 1, "find_description is not in the expected form; upstream changed"
s = s.replace(old_find, new_find)

old_head = '''def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
    language: Language,
    model_name: str,
):
    possible_answer = list(possible_answer.values())[0]'''
new_head = '''def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
    language: Language,
    model_name: str,
):
    if type(func_description) == list:
        # Any declaration of the called name that sanctions this call makes the call correct. When none
        # does, the FIRST declaration's failure is what gets reported, so the error message is the one
        # upstream would have produced.
        reported = None
        for _doc in func_description:
            _r = simple_function_checker(_doc, model_output, possible_answer, language, model_name)
            if _r["valid"]:
                return _r
            if reported is None:
                reported = _r
        return reported or {
            "valid": False,
            "error": ["No declaration of the called function."],
            "error_type": "simple_function_checker:wrong_func_name",
        }
    possible_answer = list(possible_answer.values())[0]'''
assert s.count(old_head) == 1, "simple_function_checker's head is not in the expected form"
s = s.replace(old_head, new_head)

old_simple = '''        return simple_function_checker(
            func_description[0], model_output[0], possible_answer[0], language, model_name
        )'''
new_simple = '''        # Resolve the expected declaration by the GOLD KEY, not by declaration order. Taking
        # func_description[0] is only correct when the menu holds exactly one tool; once a perturbation
        # adds one, a decoy declared first is demanded in the gold's place -- correct calls rejected,
        # calls to the decoy accepted. A row whose gold names nothing declared falls back to upstream's
        # choice so its error message is unchanged.
        doc = func_description[0]
        if type(func_description) == list and isinstance(possible_answer[0], dict):
            match = find_description(func_description, next(iter(possible_answer[0]), None))
            if match is not None:
                doc = match
        return simple_function_checker(
            doc, model_output[0], possible_answer[0], language, model_name
        )'''
assert s.count(old_simple) == 1, "the single-tool branch is not in the expected form"
s = s.replace(old_simple, new_simple)

open(p, "w").write(s)
print("fixed", p)
PY
"$PY" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$F"
printf '%s\n' "$VERSION" > "$MARK"
echo "marker  $MARK ($VERSION)"
