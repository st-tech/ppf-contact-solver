# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import (
    bake_ops,
    collision_window_ops,
    dyn_param_ops,
    export_ops,
    group_ops,
    invisible_collider_ops,
    overlay,
    panels,
    pin_capture_ops,
    pin_ops,
    profile_ops,
    sand_ops,
    save_checkpoint_ops,
    static_deform_ops,
    static_ops,
    ui_lists,
    velocity_keyframe_ops,
)

from .group_ops import (
    OBJECT_OT_AddObjectsToGroup,
    OBJECT_OT_CreateGroup,
    OBJECT_OT_DeleteAllGroups,
    OBJECT_OT_DeleteGroup,
    OBJECT_OT_DuplicateGroup,
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
    UTILITY_PT_UtilityTools,
    VISUALIZATION_PT_Visualization,
)
from .pin_ops import (
    OBJECT_OT_AddPinOperation,
    OBJECT_OT_AddPinVertexGroup,
    OBJECT_OT_CreatePinVertexGroup,
    OBJECT_OT_DeletePinKeyframes,
    OBJECT_OT_DeselectPinVertices,
    OBJECT_OT_MakePinKeyframe,
    OBJECT_OT_MovePinOperation,
    OBJECT_OT_MovePinVertexGroup,
    OBJECT_OT_PickCenterFromSelected,
    OBJECT_OT_PickVertexCenter,
    OBJECT_OT_RemovePinOperation,
    OBJECT_OT_RemovePinVertexGroup,
    OBJECT_OT_RenamePinVertexGroup,
    OBJECT_OT_SelectPinVertices,
)
from .dyn_param_ops import (
    SCENE_OT_AddDynParam,
    SCENE_OT_AddDynParamKeyframe,
    SCENE_OT_RemoveDynParam,
    SCENE_OT_RemoveDynParamKeyframe,
)
from .ui_lists import OBJECT_UL_AssignedObjectsList, OBJECT_UL_PinOperationsList, OBJECT_UL_PinVertexGroupsList
from .static_ops import (
    OBJECT_OT_AddStaticOp,
    OBJECT_OT_MoveStaticOp,
    OBJECT_OT_RemoveStaticOp,
)
from .sand_ops import (
    OBJECT_OT_ConvertToParticleMesh,
)
from .bake_ops import (
    OBJECT_OT_BakeAnimation,
    OBJECT_OT_BakeSingleFrame,
    SOLVER_OT_BakeAbort,
    SOLVER_OT_BakeAllAnimation,
    SOLVER_OT_BakeAllSingleFrame,
)
from .export_ops import (
    SOLVER_OT_ExportAlembic,
    SOLVER_OT_ExportUSD,
)
from .velocity_keyframe_ops import (
    OBJECT_OT_AddVelocityKeyframe,
    OBJECT_OT_CopyVelocityKeyframes,
    OBJECT_OT_PasteVelocityKeyframes,
    OBJECT_OT_RemoveVelocityKeyframe,
)
from .pin_capture_ops import (
    OBJECT_OT_CapturePinDeformation,
    OBJECT_OT_ClearPinDeformation,
    OBJECT_OT_RefreshFullPinState,
    SOLVER_OT_PinCaptureAbort,
)
from .static_deform_ops import (
    OBJECT_OT_CaptureStaticDeformation,
    OBJECT_OT_ClearStaticDeformation,
    SOLVER_OT_CaptureAbort,
)
from .collision_window_ops import (
    OBJECT_OT_AddCollisionWindow,
    OBJECT_OT_RemoveCollisionWindow,
)
from .save_checkpoint_ops import (
    SCENE_OT_AddSaveCheckpoint,
    SCENE_OT_RemoveSaveCheckpoint,
)
from .profile_ops import (
    OBJECT_OT_ClearMaterialProfile,
    OBJECT_OT_ClearPinProfile,
    OBJECT_OT_CopyMaterialParams,
    OBJECT_OT_CopyPinOps,
    OBJECT_OT_OpenMaterialProfile,
    OBJECT_OT_OpenPinProfile,
    OBJECT_OT_PasteMaterialParams,
    OBJECT_OT_PastePinOps,
    OBJECT_OT_ReloadMaterialProfile,
    OBJECT_OT_ReloadPinProfile,
    OBJECT_OT_SaveMaterialProfile,
    OBJECT_OT_SavePinProfile,
    SCENE_OT_ClearSceneProfile,
    SCENE_OT_OpenSceneProfile,
    SCENE_OT_ReloadSceneProfile,
    SCENE_OT_SaveSceneProfile,
)

__all__ = [
    # Group operators
    "OBJECT_OT_CreateGroup",
    "OBJECT_OT_DeleteGroup",
    "OBJECT_OT_DuplicateGroup",
    "OBJECT_OT_DeleteAllGroups",
    "OBJECT_OT_AddObjectsToGroup",
    "OBJECT_OT_RemoveObjectFromGroup",
    # Pin operators
    "OBJECT_OT_CreatePinVertexGroup",
    "OBJECT_OT_AddPinVertexGroup",
    "OBJECT_OT_RemovePinVertexGroup",
    "OBJECT_OT_RenamePinVertexGroup",
    "OBJECT_OT_MakePinKeyframe",
    "OBJECT_OT_DeletePinKeyframes",
    "OBJECT_OT_SelectPinVertices",
    "OBJECT_OT_DeselectPinVertices",
    "OBJECT_OT_AddPinOperation",
    "OBJECT_OT_RemovePinOperation",
    "OBJECT_OT_MovePinOperation",
    "OBJECT_OT_MovePinVertexGroup",
    "OBJECT_OT_PickCenterFromSelected",
    "OBJECT_OT_PickVertexCenter",
    # Static collider operators
    "OBJECT_OT_AddStaticOp",
    "OBJECT_OT_RemoveStaticOp",
    "OBJECT_OT_MoveStaticOp",
    # Sand particle-mesh operators
    "OBJECT_OT_ConvertToParticleMesh",
    # Collision window operators
    "OBJECT_OT_AddCollisionWindow",
    "OBJECT_OT_RemoveCollisionWindow",
    # Save checkpoint operators
    "SCENE_OT_AddSaveCheckpoint",
    "SCENE_OT_RemoveSaveCheckpoint",
    # Velocity keyframe operators
    "OBJECT_OT_AddVelocityKeyframe",
    "OBJECT_OT_RemoveVelocityKeyframe",
    "OBJECT_OT_CopyVelocityKeyframes",
    "OBJECT_OT_PasteVelocityKeyframes",
    # Pin capture operators
    "OBJECT_OT_CapturePinDeformation",
    "OBJECT_OT_ClearPinDeformation",
    "OBJECT_OT_RefreshFullPinState",
    "SOLVER_OT_PinCaptureAbort",
    # Static deformation capture operators
    "OBJECT_OT_CaptureStaticDeformation",
    "OBJECT_OT_ClearStaticDeformation",
    "SOLVER_OT_CaptureAbort",
    # Bake operators
    "OBJECT_OT_BakeAnimation",
    "OBJECT_OT_BakeSingleFrame",
    "SOLVER_OT_BakeAllAnimation",
    "SOLVER_OT_BakeAllSingleFrame",
    "SOLVER_OT_BakeAbort",
    # Export operators
    "SOLVER_OT_ExportUSD",
    "SOLVER_OT_ExportAlembic",
    # Profile operators
    "SCENE_OT_OpenSceneProfile",
    "SCENE_OT_ClearSceneProfile",
    "SCENE_OT_ReloadSceneProfile",
    "SCENE_OT_SaveSceneProfile",
    "OBJECT_OT_OpenMaterialProfile",
    "OBJECT_OT_ClearMaterialProfile",
    "OBJECT_OT_ReloadMaterialProfile",
    "OBJECT_OT_SaveMaterialProfile",
    "OBJECT_OT_CopyMaterialParams",
    "OBJECT_OT_PasteMaterialParams",
    "OBJECT_OT_OpenPinProfile",
    "OBJECT_OT_ClearPinProfile",
    "OBJECT_OT_ReloadPinProfile",
    "OBJECT_OT_SavePinProfile",
    "OBJECT_OT_CopyPinOps",
    "OBJECT_OT_PastePinOps",
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
    "UTILITY_PT_UtilityTools",
    # Overlay functions
    "draw_overlay_callback",
    "apply_object_overlays",
]


def register():
    """Register all classes and handlers"""
    group_ops.register()
    bake_ops.register()
    export_ops.register()
    dyn_param_ops.register()
    invisible_collider_ops.register()
    pin_ops.register()
    pin_capture_ops.register()
    static_deform_ops.register()
    static_ops.register()
    sand_ops.register()
    velocity_keyframe_ops.register()
    collision_window_ops.register()
    save_checkpoint_ops.register()
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
    save_checkpoint_ops.unregister()
    collision_window_ops.unregister()
    velocity_keyframe_ops.unregister()
    sand_ops.unregister()
    static_ops.unregister()
    static_deform_ops.unregister()
    pin_capture_ops.unregister()
    pin_ops.unregister()
    dyn_param_ops.unregister()
    export_ops.unregister()
    bake_ops.unregister()
    invisible_collider_ops.unregister()
    group_ops.unregister()
