"""
Shared system-prompt fragments used across agent nodes.
"""

# ─────────────────────────────────────────────────────────────────────────────
# C# / RhinoCommon scripting context
# Inject this into the system message whenever the agent may call
# run_csharp_script so the LLM writes correct, idiomatic RhinoCommon code.
# ─────────────────────────────────────────────────────────────────────────────

CSHARP_SCRIPT_CONTEXT = """
== C# / RhinoCommon Scripting Context ==

When calling `run_csharp_script` write standard C# 10 (Roslyn scripting).

GLOBALS (pre-injected, ready to use):
  RhinoDoc Doc  — the active Rhino document (equivalent to RhinoDoc.ActiveDoc)

AUTO-IMPORTED NAMESPACES (no `using` needed):
  Rhino, Rhino.Geometry, Rhino.DocObjects, Rhino.Display
  System, System.Linq, System.Collections.Generic, System.Text.Json

RETURN VALUE:
  The last expression in the script is JSON-serialised and returned as
  `return_value`. Do NOT use `return` — just end with the expression.
  Examples:
    id                              → returns the Guid as a string
    new { id = id.ToString() }      → returns an object
    Doc.Objects.Count               → returns an integer

COMMON PATTERNS:

  // Add a sphere
  var sphere = new Sphere(new Point3d(0, 0, 0), 5.0);
  var id = Doc.Objects.AddSphere(sphere);
  Doc.Views.Redraw();
  id

  // Add a box (Brep)
  var box = new Box(new BoundingBox(new Point3d(0,0,0), new Point3d(10,8,5)));
  var id = Doc.Objects.AddBrep(box.ToBrep());
  Doc.Views.Redraw();
  id

  // Add a cylinder (Brep)
  var axis   = new Line(new Point3d(0,0,0), new Point3d(0,0,40));
  var circle = new Circle(new Plane(axis.From, axis.Direction), 20.0);
  var cyl    = new Cylinder(circle, axis.Length);
  var id     = Doc.Objects.AddBrep(cyl.ToBrep(true, true));
  Doc.Views.Redraw();
  id

  // Move an object by GUID
  var obj = Doc.Objects.FindId(new Guid("PASTE-GUID-HERE"));
  Doc.Objects.Transform(obj.Id, Transform.Translation(new Vector3d(5, 0, 0)), true);
  Doc.Views.Redraw();
  true

  // Delete an object by GUID
  Doc.Objects.Delete(new Guid("PASTE-GUID-HERE"), true);
  Doc.Views.Redraw();
  true

  // List all object GUIDs and types
  Doc.Objects.Select(o => new { id = o.Id.ToString(), type = o.ObjectType.ToString() })
             .ToArray()

RULES:
  - Always call Doc.Views.Redraw() after adding or modifying geometry.
  - Use exact constructor signatures — call list_rhinocommon_types /
    get_type_members first if you are unsure about a type.
  - Prefer strongly-typed geometry constructors over string-based helpers.
  - Do not declare `using` statements — namespaces are already imported.
  - AddSphere / AddBrep / AddCurve return a Guid — capture it when the
    caller will need to reference the object later.
"""


def build_csharp_system_prompt(base: str, tool_list: str) -> str:
    """
    Combine a base role description, the C# context block, and the tool list
    into a complete system prompt for nodes that may call run_csharp_script.
    """
    return f"{base}\n{CSHARP_SCRIPT_CONTEXT}\nAvailable tools:\n{tool_list}"
