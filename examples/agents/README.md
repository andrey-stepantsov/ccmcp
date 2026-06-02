# Example agent skills + rules for ccmcp

The ccmcp MCP server already ships built-in usage guidance via its MCP
`instructions` field and tool descriptions — every connected agent (Claude
Code, Cursor, anything that speaks MCP) receives that guidance automatically.

The files in this directory are **optional** per-client skills/rules that you
can install into your individual agent for tighter workflow integration. They
duplicate and extend the built-in guidance — useful when you want the agent to
prioritise ccmcp at the workflow level, not just at the tool level.

## Claude Code

```bash
mkdir -p ~/.claude/skills/ccmcp-search
cp .claude/skills/ccmcp-search/SKILL.md ~/.claude/skills/ccmcp-search/
```

Or copy into a per-project `.claude/skills/` directory.

## Cursor

```bash
mkdir -p .cursor/rules
cp .cursor/rules/ccmcp-search.mdc .cursor/rules/
```

`alwaysApply: false` means the rule is offered as a hint when relevant rather
than injected on every turn — flip to `true` if you want it always present.
