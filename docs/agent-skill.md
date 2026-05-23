# Agent Skill

Install the [agentskills.io](https://agentskills.io) skill so coding agents
(Cursor, Claude Code, etc.) can guide you through Emap2lig setup, detection,
building, and result inspection.

## Install

```bash
npx skills add kiharalab/Emap2lig --skill emap2lig
```

Skill package: [`skills/emap2lig/`](../skills/emap2lig/).

## Usage

After installation, prompt your agent, for example:

> Run the Emap2lig pipeline on EMD-30556

The skill documents CLI flags, input YAML, output layout, and Web GUI behavior.
**Inference still runs on your machine** (or cluster) with a CUDA GPU — the skill
does not replace local execution or the [KiharaLab web server](web-server.md).

## Related docs

- [CLI reference](cli.md)
- [Web GUI](web-gui.md)
- [Input formats](input-format.md)
- [Output structure](output.md)
