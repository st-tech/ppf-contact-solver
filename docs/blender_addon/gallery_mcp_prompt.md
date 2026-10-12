---
orphan: true
---

# 🤖 MCP Gallery Clip: Source Prompt

This is the natural-language prompt that drove the bowl-and-spheres clip on
the [Gallery](gallery.md#from-prompt-to-simulation-via-mcp) page. The agent
in the clip is Codex. The text below is the prompt as it appears in the
clip, including the original spelling ("hemispher"). The agent planned and
executed every step through the [add-on's MCP server](integrations/mcp.md).
The clip starts with the connection already open; after that, nothing in
Blender was clicked by hand.

The prompt is free-form text. It is not one of the add-on's four
[prompt templates](integrations/mcp.md#instructions-and-prompts).

To try something similar, start the MCP server and add it to your MCP client
(see [Starting the Server](integrations/mcp.md#starting-the-server) and
[Adding the Server to an MCP Client](integrations/mcp.md#adding-the-server-to-an-mcp-client)),
then send the prompt from the client. If no connection is open, the agent
needs the host details to open one with a `connect_*` tool, so either connect
the add-on first (see [Connections](connections/index.md)) or put the details
in your prompt.

## Prompt

```text
Use ZOZO's Contact Solver MCP:

1. Make a hemisphere bowl at the center.
2. Apply some thickness with Solidify modifier.
3. Make the hemispher 3x larger.
4. Reduce the thickness one third.
5. Make this as a static object.
6. Apply the modifier and bake to the mesh.
7. Make 8 spheres at the center above the bowl with some clearance each other.
8. Add these eight spheres as dynamic solids.
9. Now simulate this configuration.

Detailed Implementation Steps:
1. Create hemisphere bowl:
   - Create UV sphere at center (0,0,0) with radius 1
   - Remove upper hemisphere (vertices with Z >= 0) to create bowl shape
   - Name object "HemisphereBowl"

2. Add thickness:
   - Apply Solidify modifier with 0.1 thickness
   - Set offset to 0 for even distribution
   - Enable even thickness option

3. Scale up:
   - Scale object by 3x in all dimensions

4. Reduce thickness:
   - Reduce Solidify thickness to 1/3 of original (~0.033)

5. Configure as static:
   - Create dynamics group
   - Set group type to "STATIC"
   - Add HemisphereBowl to the group

6. Bake modifier:
   - Apply Solidify modifier to mesh permanently

7. Create spheres:
   - Create 8 icospheres (not UV spheres) with 0.5 radius
   - Two layers of 4 spheres each
   - First layer: 2x2 grid pattern above bowl at height 1.0
   - Second layer: 2x2 grid pattern above bowl at height 3.0
   - Use at least 0.2 clearance between spheres
   - Name them Sphere_1 through Sphere_8

8. Configure spheres as dynamic:
   - Create second dynamics group
   - Set group type to "SOLID"
   - Add all 8 spheres to the group

9. Run simulation:
   - Connect to server (use existing SSH configuration)
   - Transfer scene data to solver
   - Start simulation
   - Monitor progress until completion
   - Fetch animation results

Result: Physics simulation of 8 spheres falling into a static hemisphere bowl under gravity.
```

The prompt leaves details such as mesh resolution, exact sphere placement,
and solver parameters open. The agent fills them in as it calls MCP tools
in sequence.
