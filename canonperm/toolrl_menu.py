import json

ANCHOR = "In your response, you can use the following tools:\n"
D_MARK = "\nDescription: "
P_MARK = "\nParameters: "

_DEC = json.JSONDecoder()


class MenuFormatError(Exception):
    pass


def parse_menu(instruction):
    start = instruction.find(ANCHOR)
    if start < 0:
        raise MenuFormatError("menu anchor not found")
    pos = start + len(ANCHOR)
    preamble = instruction[:pos]

    tools = []
    while True:
        head = "%d. Name: " % (len(tools) + 1)
        if not instruction.startswith(head, pos):
            break
        p = pos + len(head)
        nl = instruction.find("\n", p)
        if nl < 0:
            raise MenuFormatError("unterminated Name line")
        name = instruction[p:nl]
        if not instruction.startswith(D_MARK, nl):
            raise MenuFormatError("Description does not follow Name for %r" % name)
        d0 = nl + len(D_MARK)
        d1 = instruction.find(P_MARK, d0)
        if d1 < 0:
            raise MenuFormatError("no Parameters for %r" % name)
        description = instruction[d0:d1]
        p0 = d1 + len(P_MARK)
        try:
            params, end = _DEC.raw_decode(instruction, p0)
        except ValueError as e:
            raise MenuFormatError("bad Parameters JSON for %r: %s" % (name, e))
        if not isinstance(params, dict):
            raise MenuFormatError("Parameters for %r is not an object" % name)
        tools.append({"name": name, "description": description, "parameters": params})
        pos = end + 1 if instruction.startswith("\n", end) else end

    if not tools:
        raise MenuFormatError("no tools after the anchor")
    return preamble, tools, instruction[pos:]


def render_menu(preamble, tools, epilogue, labels=None):
    out = [preamble]
    for i, t in enumerate(tools):
        name = t["name"] if labels is None else labels[i]
        out.append("%d. Name: %s\nDescription: %s\nParameters: %s\n"
                   % (i + 1, name, t["description"],
                      json.dumps(t["parameters"])))
    out.append(epilogue)
    return "".join(out)


