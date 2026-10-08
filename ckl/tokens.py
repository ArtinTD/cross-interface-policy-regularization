
DIGITS = 10
GATE = "<tool_call>"
CLOSE = "</tool_call>"
THINK_CLOSE = "</think>"
PROBE = "\n\n<tool_call>\n<function=function_"


class LabelTable:

    def __init__(self, tokenizer, space=100, fc=True):
        self.width, self.space, self.pool, self.label_of = 4, int(space), None, {}
        digits = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(DIGITS)]
        if None in digits or digits != list(range(digits[0], digits[0] + DIGITS)):
            raise RuntimeError("the ten digits are not consecutive single token ids: %s -- the digit "
                               "gathers address them as digit_lo + d and would read wrong columns" % digits)
        self.digit_lo, self.digit_hi = digits[0], digits[0] + DIGITS - 1
        self.und = tokenizer.convert_tokens_to_ids("_")
        if tokenizer.convert_ids_to_tokens(self.und) != "_":
            raise RuntimeError("'_' is not a single token in this tokenizer")
        xml_f, xml_a = ("<function=%s_07>", "<parameter=%s_07>") if fc else (None, None)
        self.func = self._prefixes(tokenizer, "function", xml_f)
        self.arg = self._prefixes(tokenizer, "arg", xml_a)

    def _prefixes(self, tokenizer, name, xml_tpl=None):
        out = []
        for tpl in ('"%s_07"', " %s_07", xml_tpl):
            if tpl is None:
                continue
            ids = tokenizer(tpl % name, add_special_tokens=False)["input_ids"]
            hits = [i for i in range(len(ids) - 2)
                    if ids[i + 1] == self.und and ids[i + 2] == self.digit_lo
                    and i + 3 < len(ids) and ids[i + 3] == self.digit_lo + 7]
            if len(hits) == 1:
                out.append(ids[hits[0]])
            elif tpl is not xml_tpl:
                raise RuntimeError("%r does not tokenize to a [prefix, '_', tens, units] unit: %s"
                                   % (tpl % name, ids))
        if not out:
            raise RuntimeError("no prefix token found for %s_NN" % name)
        return tuple(dict.fromkeys(out))


class SpanTokens:

    def __init__(self, tokenizer):
        def one(text, what):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError("%s (%r) is %d tokens, not one: c(h) is read at ONE position"
                                   % (what, text, len(ids)))
            return ids[0]

        self.gate_id = one(GATE, "the call gate")
        self.close_id = one(CLOSE, "the call terminator")
        self.think_ids = tuple(tokenizer(THINK_CLOSE, add_special_tokens=False)["input_ids"])
        self.probe_ids = tuple(tokenizer(PROBE, add_special_tokens=False)["input_ids"])
        if self.gate_id not in self.probe_ids:
            raise RuntimeError("the probe %r does not contain the gate token: rollout_span locates the "
                               "read position by finding the gate inside it" % PROBE)


def build(tokenizer, space=100, fc=True):
    return LabelTable(tokenizer, space=space, fc=fc), SpanTokens(tokenizer)
