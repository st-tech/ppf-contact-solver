# 🔍 LLM Transparency

This page describes how LLMs were (and continue to be) used on this
project, so that readers can weigh the documentation and the add-on's
code with the right context in mind.

## Documentation and Media

All text, screenshots, diagrams, and the test simulations behind them
were produced by an LLM under iterative human direction and then
reviewed. The LLM drives Blender and the solver through the add-on's
bundled [MCP server](integrations/mcp.md).

We kindly note that planning, steering, and verifying claims against
the running add-on still take real human effort. The LLM is used as an
authoring tool, **not a fully autonomous author**.

A significant effort has been poured into setting up a semi-automatic
pipeline for LLM-assisted documentation authoring rather than
generating the docs in one shot. That pipeline still depends on
noticeable human attention and involvement at every iteration, and it
is not meant to imply that the output is written without careful
review.

## Add-on Code

The Blender add-on itself was first developed with GitHub Copilot in
its early stages. Later, essentially all direct coding has been
carried out by Claude Code and Codex under the author's direction.

The add-on's internal algorithms have not been scrutinized to the
same depth as the academic papers associated with the underlying
solver engine. Readers relying on the add-on for research or
production work should treat the add-on's code with that context in
mind; the solver engine itself remains backed by the peer-reviewed
publications it ships with. Day-to-day behavior of the add-on is
verified by a semi-automated test suite that exercises it end to end.

## Code Quality and Testing

Code quality is kept in check through an automated test suite, and
the coding agents themselves are a significant help in writing and
maintaining those tests. That said, exhaustively hunting down every
edge-case bug (which would be possible with more effort) is not a
hard requirement of the project, so occasional rough edges should be
expected.

If you run into a bug, please feel free to
[open an issue on GitHub](https://github.com/st-tech/ppf-contact-solver/issues).
Reports with a reproduction path are especially helpful and are the
main way the add-on's rough edges get smoothed out over time.
