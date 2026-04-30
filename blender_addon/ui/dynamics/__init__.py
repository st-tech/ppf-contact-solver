# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import (
    bake_ops,
    collision_window_ops,
    dyn_param_ops,
    group_ops,
    invisible_collider_ops,
    overlay,
    panels,
    pin_ops,
    profile_ops,
    static_ops,
    ui_lists,
    velocity_keyframe_ops,
)

from .group_ops import (
    OBJECT_OT_AddObjectsToGroup,
    OBJECT_OT_CreateGroup,
    OBJECT_OT_DeleteAllGroups,
    OBJECT_OT_DeleteGroup,
    OBJECT_OT_RemoveObjectFromGroup,
)
from .invisible_collider_ops import (
    SCENE_OT_AddColliderKeyframe,
    SCENE_OT_AddInvisibleCollider,
    SCENE_OT_RemoveColliderKeyframe,
    SCENE_OT_RemoveInvisibleCollider,
)
from .overlay import apply_object_overlays, draw_overlay_callback
from .panels import (
    DYNAMICS_PT_Groups,
    MAIN_PT_SceneConfiguration,
    SNAPMERGE_PT_SnapAndMerge,
    VISUALIZATION_PT_Visualization,
)
from .pin_ops import (
    OBJECT_OT_AddPinOperation,
    OBJECT_OT_AddPinVertexGroup,
    OBJECT_OT_DeletePinKeyframes,
    OBJECT_OT_DeselectPinVertices,
    OBJECT_OT_MakePinKeyframe,
    OBJECT_OT_MovePinOperation,
    OBJECT_OT_RemovePinOperation,
    OBJECT_OT_RemovePinVertexGroup,
    OBJECT_OT_SelectPinVertices,
)
from .dyn_param_ops import (
    SCENE_OT_AddDynParam,
    SCENE_OT_AddDynParamKeyframe,
    SCENE_OT_RemoveDynParam,
    SCENE_OT_RemoveDynParamKeyframe,
)
from .ui_lists import OBJECT_UL_AssignedObjectsList, OBJECT_UL_PinOperationsList, OBJECT_UL_PinVertexGroupsList

__all__ = [
    # Group operators
    "OBJECT_OT_CreateGroup",
    "OBJECT_OT_DeleteGroup",
    "OBJECT_OT_DeleteAllGroups",
    "OBJECT_OT_AddObjectsToGroup",
    "OBJECT_OT_RemoveObjectFromGroup",
    # Pin operators
    "OBJECT_OT_AddPinVertexGroup",
    "OBJECT_OT_RemovePinVertexGroup",
    "OBJECT_OT_MakePinKeyframe",
    "OBJECT_OT_DeletePinKeyframes",
    "OBJECT_OT_SelectPinVertices",
    "OBJECT_OT_DeselectPinVertices",
    "OBJECT_OT_AddPinOperation",
    "OBJECT_OT_RemovePinOperation",
    "OBJECT_OT_MovePinOperation",
    # UI Lists
    "OBJECT_UL_AssignedObjectsList",
    "OBJECT_UL_PinVertexGroupsList",
    "OBJECT_UL_PinOperationsList",
    # Dynamic parameter operators
    "SCENE_OT_AddDynParam",
    "SCENE_OT_RemoveDynParam",
    "SCENE_OT_AddDynParamKeyframe",
    "SCENE_OT_RemoveDynParamKeyframe",
    # Invisible collider operators
    "SCENE_OT_AddInvisibleCollider",
    "SCENE_OT_RemoveInvisibleCollider",
    "SCENE_OT_AddColliderKeyframe",
    "SCENE_OT_RemoveColliderKeyframe",
    # Panels
    "MAIN_PT_SceneConfiguration",
    "DYNAMICS_PT_Groups",
    "VISUALIZATION_PT_Visualization",
    "SNAPMERGE_PT_SnapAndMerge",
    # Overlay functions
    "draw_overlay_callback",
    "apply_object_overlays",
]


def register():
    """Register all classes and handlers"""
    group_ops.register()
    bake_ops.register()
    dyn_param_ops.register()
    invisible_collider_ops.register()
    pin_ops.register()
    static_ops.register()
    velocity_keyframe_ops.register()
    collision_window_ops.register()
    profile_ops.register()
    ui_lists.register()
    panels.register()
    overlay.register()


def unregister():
    """Unregister all classes and handlers"""
    overlay.unregister()
    panels.unregister()
    ui_lists.unregister()
    profile_ops.unregister()
    collision_window_ops.unregister()
    velocity_keyframe_ops.unregister()
    static_ops.unregister()
    pin_ops.unregister()
    dyn_param_ops.unregister()
    bake_ops.unregister()
    invisible_collider_ops.unregister()
    group_ops.unregister()
