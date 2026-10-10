# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Regenerates the expected outputs of the native Jinja test
(tests/agentic_test.cpp) with Python's jinja2, configured the way Hugging
Face renders chat templates. Run after adding a template here:

    python regen.py

Every <name>.jinja is rendered with every context of contexts.json into
expected/<name>.<context>.txt, or expected/<name>.<context>.error when
jinja2 itself refuses (raise_exception in the template): the native
engine must then fail too, so the caller falls back to the built-in format.
"""

import datetime
import glob
import json
import os

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_NOW = 1767268800  # 2026-01-01 12:00 UTC: the same calendar day in every time zone


class GenerationTag(Extension):
    """HF's {% generation %}...{% endgeneration %}: renders its body as is."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols", GenerationTag])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.fromtimestamp(FIXED_NOW).strftime(fmt)
    return env


def main():
    env = environment()
    contexts = json.load(open(os.path.join(HERE, "contexts.json"), encoding="utf-8"))
    out_dir = os.path.join(HERE, "expected")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*")):
        os.remove(old)
    for path in sorted(glob.glob(os.path.join(HERE, "*.jinja"))):
        name = os.path.basename(path)[: -len(".jinja")]
        template = env.from_string(open(path, encoding="utf-8", newline="").read())
        for cname, ctx in contexts.items():
            base = os.path.join(out_dir, f"{name}.{cname}")
            try:
                text = template.render(**ctx)
            except Exception as exc:  # the native side must refuse as well
                open(base + ".error", "w", encoding="utf-8").write(f"{type(exc).__name__}: {exc}\n")
                continue
            open(base + ".txt", "w", encoding="utf-8", newline="").write(text)
    print("expected outputs regenerated in", out_dir)


if __name__ == "__main__":
    main()
