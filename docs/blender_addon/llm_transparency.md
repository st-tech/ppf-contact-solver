# 🔍 LLM Transparency

This page describes how LLMs were (and continue to be) used on this
project, so that readers can weigh the documentation and the add-on's
code in context. For the README, docstrings, notebooks, and paper, see the
repository-wide
[LLM Transparency](https://github.com/st-tech/ppf-contact-solver/blob/main/articles/llm_transparency.md)
article.

## Documentation, Screenshots, and Diagrams

All text, screenshots, and diagrams in this documentation, and the test
simulations behind them, were produced by an LLM under iterative human
direction and then reviewed. The LLM drives Blender and the solver through
the add-on's bundled [MCP server](integrations/mcp.md).

Planning, steering, and checking claims against the running add-on still
take human effort. The LLM is used as an authoring tool, **not a fully
autonomous author**.

The documentation is written through a semi-automatic pipeline rather than
generated in one pass. The pipeline needs human attention at every
iteration, and its output is reviewed carefully.

The two MCP clips in the [Gallery](gallery.md#from-prompt-to-simulation-via-mcp)
show an LLM agent driving Blender through the same server, and the prompts
that produced them are published there. The other Gallery scenes and the
[Video Tutorials](tutorial.md) are not covered by this page.

## Add-on Code

The add-on was started with GitHub Copilot. Nearly all later coding was
done by Claude Code and Codex under the author's direction. The author
reviews the code before it is made public, but UI logic and glue code are not
reviewed with the same depth as the solver's algorithms.

The add-on's internal algorithms have not been scrutinized to the same
depth as the academic papers associated with the underlying solver engine.
Keep that in mind if you rely on the add-on for research or production
work. The solver engine itself is backed by the peer-reviewed publications
it ships with.

## Code Quality and Testing

An automated test suite that runs in CI verifies the add-on's day-to-day
behavior. It exercises the add-on's UI flows; see Blender CI under
[GitHub Actions](https://github.com/st-tech/ppf-contact-solver#-github-actions)
in the README. The coding agents also help write and maintain these tests.
Hunting down every edge-case bug is not a hard requirement of the project,
so occasional bugs remain.

If you run into a bug,
[open an issue on GitHub](https://github.com/st-tech/ppf-contact-solver/issues);
[Reaching the Author](reaching_the_author.md#reporting-a-bug) lists what to
attach. Reports with a reproduction path are the most useful.
