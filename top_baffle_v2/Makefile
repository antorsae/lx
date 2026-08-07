# Generates every flag-dependent artifact for BOTH stand-foot states:
#
#   build/floor_stand/      LX_STAND_FOOT=1  fused foot + NL8 panel; Obi-Wan carries
#                     its integral floor mount in the LM carrier
#   build/no_floor_stand/   LX_STAND_FOOT=0  flat piece_bottom, bridge
#                     pass-throughs; Obi-Wan LM owns a front-flush solid web
#
# Each folder gets a complete print set (stl/), all review STEP models, and
# five documentation PNGs (three overlays + two routing profiles).
# Proud attachments.step is flag-independent and stays at the top level.
# Obi-Wan LM core and its attachment manifest differ by stand state; the floor
# mount is integral LM material, while the remaining attachments are optional.
#
# Self-contained: STLs/PNGs use build123d/shapely/matplotlib/numpy and
# STEPs use build123d's native exporter via scripts/export_steps.py -- no
# external CAD-skill dependency. OCC booleans retain native memory, so local
# jobs stay serial while osado uses only the dispatcher's bounded slots.
#
# Usage:
#   make                     remote build on osado.lan (default)
#   make floor_stand         remote build of every floor-state variant
#   make floor_obiwan          focused remote Obi-Wan floor release + QA
#   make obiwan_release        both Obi-Wan states + Ac/Ae, concurrent on osado
#   LX_CAD_EXECUTION=local make PYTHON=/path/venv/bin/python
#   make check               proud regression + final Obi-Wan R6F suites
#   make candidate           checks + regenerated candidate artifacts + QA
#   make release             candidate QA + fail-closed physical-release gate
#   make check_bambu_3mf_audit local synthetic 3MF transform/mesh regressions
#   make bambu_profiles      local Bambu preset resolution; no CAD/printer I/O
#   make bambu_dry_run       local per-STL command/provenance preview
#   make bambu_slice_obiwan  local filtered ready-project diagnostic
#   make bambu_slice_release local authoritative ready-project slice/audit
#   make obiwan_no_floor_combo_plate local no-floor 01a+02+03+04 plate
#   make obiwan_floor_combo_plate local floor-stand 01b+02+03+04 plate
#   make obiwan_ac_wing_plate local 05b+06b+08b+09b single-plate slice/audit
#   make obiwan_ae_wing_plate local 11b+12b+14b+15b single-plate slice/audit
#   make to_print            local complete P2S shelf: audit + 51 friendly files
#   make to_print_validate   validate an existing P2S shelf without slicing
#   make artifacts           local artifacts/ product facade: relink + rehash
#   make iso_matrix          local standardized README ISO render set
#
# Compatible with the macOS-provided GNU make 3.81.

# Package geometry and importable CLIs have one canonical absolute search
# path.  It remains valid inside recipes that render from a build subdirectory.
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/scripts

# First-class local packaging artifacts are declared before the execution-mode
# router so invoking either a public alias or a concrete output path can never
# fall through to the remote CAD dispatcher.
CAPTIVE_MAGNET_CATALOG ?= review/captive_magnet_release_catalog.json
BAMBU_AUDIT_OUTPUT ?= review/captive_magnet_slice_audit
PETG_GF_PROFILE := captive_magnet_slicing_profile_petg_gf.json
PETG_GF_AUDIT_OUTPUT ?= review/captive_magnet_slice_audit_petg_gf_01a
PETG_GF_01B_AUDIT_OUTPUT ?= review/captive_magnet_slice_audit_petg_gf_01b
TO_PRINT_SHELF ?= to_print
OBIWAN_01A_ARTIFACT_ID := no_floor_stand:Obi-Wan-split:obiwan_optional_lm_keyed_1_of_2_bottom
OBIWAN_01A_SLUG := no_floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_1_of_2_bottom
OBIWAN_01A_RELEASED_STL := $(CURDIR)/build/no_floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl
OBIWAN_01A_RELEASED_SIDECAR := $(OBIWAN_01A_RELEASED_STL:.stl=.print.json)
OBIWAN_01A_MODIFIER_DIR := build/no_floor_stand/process_modifiers
OBIWAN_01A_MODIFIER_STL := $(OBIWAN_01A_MODIFIER_DIR)/obiwan_optional_lm_keyed_1_of_2_bottom.bridge_root_100pct.modifier.stl
OBIWAN_01A_MODIFIER_CONTRACT := $(OBIWAN_01A_MODIFIER_STL:.stl=.json)
OBIWAN_01A_MODIFIER_STAMP := $(OBIWAN_01A_MODIFIER_DIR)/.stamp_bridge_root_100pct
OBIWAN_01A_PETG_GF_SLICE_DIR := $(PETG_GF_AUDIT_OUTPUT)/slices/$(OBIWAN_01A_SLUG)
OBIWAN_01A_PETG_GF_READY_DIR := $(OBIWAN_01A_PETG_GF_SLICE_DIR)/ready
OBIWAN_01A_PETG_GF_PROJECT := $(OBIWAN_01A_PETG_GF_READY_DIR)/ready_to_print.gcode.3mf
OBIWAN_01A_PETG_GF_GCODE := $(OBIWAN_01A_PETG_GF_READY_DIR)/plate_1.gcode
OBIWAN_01A_PETG_GF_AUDIT := $(OBIWAN_01A_PETG_GF_SLICE_DIR)/captive_magnet_slice_audit.json
OBIWAN_01A_PETG_GF_STAMP := $(OBIWAN_01A_PETG_GF_READY_DIR)/.stamp_petg_gf_validated
OBIWAN_01A_FRIENDLY_NAME := obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand
OBIWAN_01A_PETG_GF_PROMOTION_STAMP := $(OBIWAN_01A_PETG_GF_READY_DIR)/.stamp_to_print_promoted
OBIWAN_01A_PETG_GF_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_01A_FRIENDLY_NAME).stl
OBIWAN_01A_PETG_GF_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_01A_FRIENDLY_NAME).gcode.3mf
OBIWAN_01A_PETG_GF_SHELF_OUTPUTS := $(OBIWAN_01A_PETG_GF_SHELF_STL) \
                                   $(OBIWAN_01A_PETG_GF_SHELF_PROJECT)
OBIWAN_01B_ARTIFACT_ID := floor_stand:Obi-Wan-split:obiwan_optional_lm_keyed_1_of_2_bottom
OBIWAN_01B_SLUG := floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_1_of_2_bottom
OBIWAN_01B_RELEASED_STL := $(CURDIR)/build/floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl
OBIWAN_01B_RELEASED_SIDECAR := $(OBIWAN_01B_RELEASED_STL:.stl=.print.json)
OBIWAN_01B_PETG_GF_SLICE_DIR := $(PETG_GF_01B_AUDIT_OUTPUT)/slices/$(OBIWAN_01B_SLUG)
OBIWAN_01B_PETG_GF_READY_DIR := $(OBIWAN_01B_PETG_GF_SLICE_DIR)/ready
OBIWAN_01B_PETG_GF_PROJECT := $(OBIWAN_01B_PETG_GF_READY_DIR)/ready_to_print.gcode.3mf
OBIWAN_01B_PETG_GF_GCODE := $(OBIWAN_01B_PETG_GF_READY_DIR)/plate_1.gcode
OBIWAN_01B_PETG_GF_AUDIT := $(OBIWAN_01B_PETG_GF_SLICE_DIR)/captive_magnet_slice_audit.json
OBIWAN_01B_PETG_GF_STAMP := $(OBIWAN_01B_PETG_GF_READY_DIR)/.stamp_petg_gf_validated
OBIWAN_01B_FRIENDLY_NAME := obiwan_01_LM_bottom_keyed_1_of_2_floor_stand
OBIWAN_01B_PETG_GF_PROMOTION_STAMP := $(OBIWAN_01B_PETG_GF_READY_DIR)/.stamp_to_print_promoted
OBIWAN_01B_PETG_GF_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_01B_FRIENDLY_NAME).stl
OBIWAN_01B_PETG_GF_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_01B_FRIENDLY_NAME).gcode.3mf
OBIWAN_01B_PETG_GF_SHELF_OUTPUTS := $(OBIWAN_01B_PETG_GF_SHELF_STL) \
                                   $(OBIWAN_01B_PETG_GF_SHELF_PROJECT)
OBIWAN_COMBO_SOURCE_DIR := build/print_plates/obiwan
OBIWAN_NO_FLOOR_COMBO_NAME := obiwan_01_02_03_04_LM_UM_combo_no_floor_stand
OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP := $(OBIWAN_COMBO_SOURCE_DIR)/.stamp_$(OBIWAN_NO_FLOOR_COMBO_NAME)
OBIWAN_NO_FLOOR_COMBO_STL := $(OBIWAN_COMBO_SOURCE_DIR)/$(OBIWAN_NO_FLOOR_COMBO_NAME).stl
OBIWAN_NO_FLOOR_COMBO_MANIFEST := $(OBIWAN_COMBO_SOURCE_DIR)/$(OBIWAN_NO_FLOOR_COMBO_NAME).plate.json
OBIWAN_NO_FLOOR_COMBO_WORKSPACE ?= review/to_print_slice_workspace/composite/$(OBIWAN_NO_FLOOR_COMBO_NAME)
OBIWAN_NO_FLOOR_COMBO_READY_DIR := $(OBIWAN_NO_FLOOR_COMBO_WORKSPACE)/ready
OBIWAN_NO_FLOOR_COMBO_READY_STAMP := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/.stamp_validated
OBIWAN_NO_FLOOR_COMBO_PROJECT := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/$(OBIWAN_NO_FLOOR_COMBO_NAME).gcode.3mf
OBIWAN_NO_FLOOR_COMBO_GCODE := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/plate_1.gcode
OBIWAN_NO_FLOOR_COMBO_RESULT := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/result.json
OBIWAN_NO_FLOOR_COMBO_AUDIT := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/plate_audit.json
OBIWAN_NO_FLOOR_COMBO_READY_OUTPUTS := $(OBIWAN_NO_FLOOR_COMBO_PROJECT) \
                                      $(OBIWAN_NO_FLOOR_COMBO_GCODE) \
                                      $(OBIWAN_NO_FLOOR_COMBO_RESULT) \
                                      $(OBIWAN_NO_FLOOR_COMBO_AUDIT)
OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP := $(OBIWAN_NO_FLOOR_COMBO_READY_DIR)/.stamp_to_print_promoted
OBIWAN_NO_FLOOR_COMBO_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_NO_FLOOR_COMBO_NAME).stl
OBIWAN_NO_FLOOR_COMBO_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_NO_FLOOR_COMBO_NAME).gcode.3mf
OBIWAN_NO_FLOOR_COMBO_SHELF_OUTPUTS := $(OBIWAN_NO_FLOOR_COMBO_SHELF_STL) \
                                      $(OBIWAN_NO_FLOOR_COMBO_SHELF_PROJECT)
OBIWAN_FLOOR_COMBO_NAME := obiwan_01_02_03_04_LM_UM_combo_floor_stand
OBIWAN_FLOOR_COMBO_SOURCE_STAMP := $(OBIWAN_COMBO_SOURCE_DIR)/.stamp_$(OBIWAN_FLOOR_COMBO_NAME)
OBIWAN_FLOOR_COMBO_STL := $(OBIWAN_COMBO_SOURCE_DIR)/$(OBIWAN_FLOOR_COMBO_NAME).stl
OBIWAN_FLOOR_COMBO_MANIFEST := $(OBIWAN_COMBO_SOURCE_DIR)/$(OBIWAN_FLOOR_COMBO_NAME).plate.json
OBIWAN_FLOOR_COMBO_WORKSPACE ?= review/to_print_slice_workspace/composite/$(OBIWAN_FLOOR_COMBO_NAME)
OBIWAN_FLOOR_COMBO_READY_DIR := $(OBIWAN_FLOOR_COMBO_WORKSPACE)/ready
OBIWAN_FLOOR_COMBO_READY_STAMP := $(OBIWAN_FLOOR_COMBO_READY_DIR)/.stamp_validated
OBIWAN_FLOOR_COMBO_PROJECT := $(OBIWAN_FLOOR_COMBO_READY_DIR)/$(OBIWAN_FLOOR_COMBO_NAME).gcode.3mf
OBIWAN_FLOOR_COMBO_GCODE := $(OBIWAN_FLOOR_COMBO_READY_DIR)/plate_1.gcode
OBIWAN_FLOOR_COMBO_RESULT := $(OBIWAN_FLOOR_COMBO_READY_DIR)/result.json
OBIWAN_FLOOR_COMBO_AUDIT := $(OBIWAN_FLOOR_COMBO_READY_DIR)/plate_audit.json
OBIWAN_FLOOR_COMBO_READY_OUTPUTS := $(OBIWAN_FLOOR_COMBO_PROJECT) \
                                   $(OBIWAN_FLOOR_COMBO_GCODE) \
                                   $(OBIWAN_FLOOR_COMBO_RESULT) \
                                   $(OBIWAN_FLOOR_COMBO_AUDIT)
OBIWAN_FLOOR_COMBO_PROMOTION_STAMP := $(OBIWAN_FLOOR_COMBO_READY_DIR)/.stamp_to_print_promoted
OBIWAN_FLOOR_COMBO_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_FLOOR_COMBO_NAME).stl
OBIWAN_FLOOR_COMBO_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_FLOOR_COMBO_NAME).gcode.3mf
OBIWAN_FLOOR_COMBO_SHELF_OUTPUTS := $(OBIWAN_FLOOR_COMBO_SHELF_STL) \
                                   $(OBIWAN_FLOOR_COMBO_SHELF_PROJECT)
OBIWAN_AC_WING_PLATE_NAME := obiwan_ac_wings_split2_combo
OBIWAN_AC_WING_PLATE_SOURCE_DIR := build/print_plates/obiwan
OBIWAN_AC_WING_PLATE_SOURCE_STAMP := $(OBIWAN_AC_WING_PLATE_SOURCE_DIR)/.stamp_$(OBIWAN_AC_WING_PLATE_NAME)
OBIWAN_AC_WING_PLATE_STL := $(OBIWAN_AC_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AC_WING_PLATE_NAME).stl
OBIWAN_AC_WING_PLATE_MANIFEST := $(OBIWAN_AC_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AC_WING_PLATE_NAME).plate.json
OBIWAN_AC_WING_PLATE_PREVIEW := $(OBIWAN_AC_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AC_WING_PLATE_NAME).layout.png
OBIWAN_AC_WING_PLATE_WORKSPACE ?= review/to_print_slice_workspace/composite/$(OBIWAN_AC_WING_PLATE_NAME)
OBIWAN_AC_WING_PLATE_READY_DIR := $(OBIWAN_AC_WING_PLATE_WORKSPACE)/ready
OBIWAN_AC_WING_PLATE_READY_STAMP := $(OBIWAN_AC_WING_PLATE_READY_DIR)/.stamp_validated
OBIWAN_AC_WING_PLATE_PROJECT := $(OBIWAN_AC_WING_PLATE_READY_DIR)/$(OBIWAN_AC_WING_PLATE_NAME).gcode.3mf
OBIWAN_AC_WING_PLATE_GCODE := $(OBIWAN_AC_WING_PLATE_READY_DIR)/plate_1.gcode
OBIWAN_AC_WING_PLATE_RESULT := $(OBIWAN_AC_WING_PLATE_READY_DIR)/result.json
OBIWAN_AC_WING_PLATE_AUDIT := $(OBIWAN_AC_WING_PLATE_READY_DIR)/plate_audit.json
OBIWAN_AC_WING_PLATE_READY_OUTPUTS := $(OBIWAN_AC_WING_PLATE_PROJECT) \
                                     $(OBIWAN_AC_WING_PLATE_GCODE) \
                                     $(OBIWAN_AC_WING_PLATE_RESULT) \
                                     $(OBIWAN_AC_WING_PLATE_AUDIT)
OBIWAN_AC_WING_PLATE_PROMOTION_STAMP := $(OBIWAN_AC_WING_PLATE_READY_DIR)/.stamp_to_print_promoted
OBIWAN_AC_WING_PLATE_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_AC_WING_PLATE_NAME).stl
OBIWAN_AC_WING_PLATE_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_AC_WING_PLATE_NAME).gcode.3mf
OBIWAN_AC_WING_PLATE_SHELF_OUTPUTS := $(OBIWAN_AC_WING_PLATE_SHELF_STL) \
                                     $(OBIWAN_AC_WING_PLATE_SHELF_PROJECT)
OBIWAN_AE_WING_PLATE_NAME := obiwan_ae_wings_split2_combo
OBIWAN_AE_WING_PLATE_SOURCE_DIR := build/print_plates/obiwan
OBIWAN_AE_WING_PLATE_SOURCE_STAMP := $(OBIWAN_AE_WING_PLATE_SOURCE_DIR)/.stamp_$(OBIWAN_AE_WING_PLATE_NAME)
OBIWAN_AE_WING_PLATE_STL := $(OBIWAN_AE_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AE_WING_PLATE_NAME).stl
OBIWAN_AE_WING_PLATE_MANIFEST := $(OBIWAN_AE_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AE_WING_PLATE_NAME).plate.json
OBIWAN_AE_WING_PLATE_PREVIEW := $(OBIWAN_AE_WING_PLATE_SOURCE_DIR)/$(OBIWAN_AE_WING_PLATE_NAME).layout.png
OBIWAN_AE_WING_PLATE_WORKSPACE ?= review/to_print_slice_workspace/composite/$(OBIWAN_AE_WING_PLATE_NAME)
OBIWAN_AE_WING_PLATE_READY_DIR := $(OBIWAN_AE_WING_PLATE_WORKSPACE)/ready
OBIWAN_AE_WING_PLATE_READY_STAMP := $(OBIWAN_AE_WING_PLATE_READY_DIR)/.stamp_validated
OBIWAN_AE_WING_PLATE_PROJECT := $(OBIWAN_AE_WING_PLATE_READY_DIR)/$(OBIWAN_AE_WING_PLATE_NAME).gcode.3mf
OBIWAN_AE_WING_PLATE_GCODE := $(OBIWAN_AE_WING_PLATE_READY_DIR)/plate_1.gcode
OBIWAN_AE_WING_PLATE_RESULT := $(OBIWAN_AE_WING_PLATE_READY_DIR)/result.json
OBIWAN_AE_WING_PLATE_AUDIT := $(OBIWAN_AE_WING_PLATE_READY_DIR)/plate_audit.json
OBIWAN_AE_WING_PLATE_READY_OUTPUTS := $(OBIWAN_AE_WING_PLATE_PROJECT) \
                                     $(OBIWAN_AE_WING_PLATE_GCODE) \
                                     $(OBIWAN_AE_WING_PLATE_RESULT) \
                                     $(OBIWAN_AE_WING_PLATE_AUDIT)
OBIWAN_AE_WING_PLATE_PROMOTION_STAMP := $(OBIWAN_AE_WING_PLATE_READY_DIR)/.stamp_to_print_promoted
OBIWAN_AE_WING_PLATE_SHELF_STL := $(TO_PRINT_SHELF)/obiwan/stl/$(OBIWAN_AE_WING_PLATE_NAME).stl
OBIWAN_AE_WING_PLATE_SHELF_PROJECT := $(TO_PRINT_SHELF)/obiwan/3mf/$(OBIWAN_AE_WING_PLATE_NAME).gcode.3mf
OBIWAN_AE_WING_PLATE_SHELF_OUTPUTS := $(OBIWAN_AE_WING_PLATE_SHELF_STL) \
                                     $(OBIWAN_AE_WING_PLATE_SHELF_PROJECT)

# Optional Stock/Slim vase family carrying two opposed TEBM35C10-4 BMR
# drivers. CAD is one complete remotely promotable root with independently
# hash-bound profile children; Bambu slicing remains local-only and consumes
# those already-promoted roots without rebuilding CAD.
VASE_TEBM_NAME := vase_TEBM35C10-4
VASE_TEBM_ROOT := build/$(VASE_TEBM_NAME)
VASE_TEBM_STOCK_ROOT := $(VASE_TEBM_ROOT)/stock
VASE_TEBM_SLIM_ROOT := $(VASE_TEBM_ROOT)/slim
VASE_TEBM_STOCK_BREP := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).brep
VASE_TEBM_STOCK_STEP := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).step
VASE_TEBM_STOCK_STL := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).stl
VASE_TEBM_STOCK_SIDECAR := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).print.json
VASE_TEBM_STOCK_FACTS := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).facts.json
VASE_TEBM_STOCK_CATALOG := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).catalog.json
VASE_TEBM_STOCK_PROFILE := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).slicing_profile.json
VASE_TEBM_STOCK_MANIFEST := $(VASE_TEBM_STOCK_ROOT)/cad_manifest.json
VASE_TEBM_STOCK_EXPORT_STAMP := $(VASE_TEBM_STOCK_ROOT)/.stamp_cad_validated
VASE_TEBM_SLIM_BREP := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).brep
VASE_TEBM_SLIM_STEP := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).step
VASE_TEBM_SLIM_STL := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).stl
VASE_TEBM_SLIM_SIDECAR := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).print.json
VASE_TEBM_SLIM_FACTS := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).facts.json
VASE_TEBM_SLIM_CATALOG := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).catalog.json
VASE_TEBM_SLIM_PROFILE := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).slicing_profile.json
VASE_TEBM_SLIM_MANIFEST := $(VASE_TEBM_SLIM_ROOT)/cad_manifest.json
VASE_TEBM_SLIM_EXPORT_STAMP := $(VASE_TEBM_SLIM_ROOT)/.stamp_cad_validated
VASE_TEBM_CAD_STAMP := $(VASE_TEBM_ROOT)/.stamp_cad_ready
VASE_TEBM_STOCK_SLICE_OUTPUT ?= $(VASE_TEBM_STOCK_ROOT)/slice_audit
VASE_TEBM_SLIM_SLICE_OUTPUT ?= $(VASE_TEBM_SLIM_ROOT)/slice_audit
VASE_TEBM_STOCK_SLUG := shared_Stock-TEBM35C10-4_$(VASE_TEBM_NAME)
VASE_TEBM_SLIM_SLUG := shared_Slim-TEBM35C10-4_$(VASE_TEBM_NAME)
VASE_TEBM_STOCK_SLICE_DIR := $(VASE_TEBM_STOCK_SLICE_OUTPUT)/slices/$(VASE_TEBM_STOCK_SLUG)
VASE_TEBM_SLIM_SLICE_DIR := $(VASE_TEBM_SLIM_SLICE_OUTPUT)/slices/$(VASE_TEBM_SLIM_SLUG)
VASE_TEBM_STOCK_READY_DIR := $(VASE_TEBM_STOCK_SLICE_DIR)/ready
VASE_TEBM_SLIM_READY_DIR := $(VASE_TEBM_SLIM_SLICE_DIR)/ready
VASE_TEBM_STOCK_READY_PROJECT := $(VASE_TEBM_STOCK_READY_DIR)/ready_to_print.gcode.3mf
VASE_TEBM_SLIM_READY_PROJECT := $(VASE_TEBM_SLIM_READY_DIR)/ready_to_print.gcode.3mf
VASE_TEBM_STOCK_3MF := $(VASE_TEBM_STOCK_ROOT)/$(VASE_TEBM_NAME).gcode.3mf
VASE_TEBM_SLIM_3MF := $(VASE_TEBM_SLIM_ROOT)/$(VASE_TEBM_NAME).gcode.3mf
VASE_TEBM_STOCK_GCODE := $(VASE_TEBM_STOCK_READY_DIR)/plate_1.gcode
VASE_TEBM_SLIM_GCODE := $(VASE_TEBM_SLIM_READY_DIR)/plate_1.gcode
VASE_TEBM_STOCK_SLICE_AUDIT := $(VASE_TEBM_STOCK_SLICE_DIR)/captive_magnet_slice_audit.json
VASE_TEBM_SLIM_SLICE_AUDIT := $(VASE_TEBM_SLIM_SLICE_DIR)/captive_magnet_slice_audit.json
VASE_TEBM_STOCK_SLICE_STAMP := $(VASE_TEBM_STOCK_READY_DIR)/.stamp_vase_tebm_validated
VASE_TEBM_SLIM_SLICE_STAMP := $(VASE_TEBM_SLIM_READY_DIR)/.stamp_vase_tebm_validated
VASE_TEBM_STOCK_CAD_OUTPUTS := $(VASE_TEBM_STOCK_BREP) $(VASE_TEBM_STOCK_STEP) \
                               $(VASE_TEBM_STOCK_STL) $(VASE_TEBM_STOCK_SIDECAR) \
                               $(VASE_TEBM_STOCK_FACTS) $(VASE_TEBM_STOCK_CATALOG) \
                               $(VASE_TEBM_STOCK_PROFILE) $(VASE_TEBM_STOCK_MANIFEST)
VASE_TEBM_SLIM_CAD_OUTPUTS := $(VASE_TEBM_SLIM_BREP) $(VASE_TEBM_SLIM_STEP) \
                              $(VASE_TEBM_SLIM_STL) $(VASE_TEBM_SLIM_SIDECAR) \
                              $(VASE_TEBM_SLIM_FACTS) $(VASE_TEBM_SLIM_CATALOG) \
                              $(VASE_TEBM_SLIM_PROFILE) $(VASE_TEBM_SLIM_MANIFEST)
VASE_TEBM_CAD_OUTPUTS := $(VASE_TEBM_STOCK_CAD_OUTPUTS) \
                         $(VASE_TEBM_SLIM_CAD_OUTPUTS)

# Bambu slicing is workstation-only.  Detect these public goals before the
# remote-first dispatcher captures MAKECMDGOALS, force even an explicitly
# supplied remote/remote-worker mode back to local, and reject ambiguous mixed
# CAD+slicer invocations.  Multiple Bambu goals in one invocation are allowed.
LOCAL_BAMBU_GOALS := check_bambu_3mf_audit \
                     bambu_profiles bambu_dry_run \
                     bambu_slice_obiwan bambu_slice_release \
                     vase_tebm35c10_4_3mf \
                     vase_tebm35c10_4_3mf_validate \
                     vase_tebm35c10_4_stock_3mf \
                     vase_tebm35c10_4_slim_3mf \
                     vase_tebm35c10_4_stock_3mf_validate \
                     vase_tebm35c10_4_slim_3mf_validate \
                     obiwan_no_floor_petg_gf_modifier \
                     obiwan_no_floor_petg_gf_01a \
                     obiwan_no_floor_petg_gf_01a_validate \
                     obiwan_no_floor_petg_gf_01a_to_print \
                     obiwan_floor_petg_gf_01b \
                     obiwan_floor_petg_gf_01b_validate \
                     obiwan_floor_petg_gf_01b_to_print \
                     obiwan_combo_plate_source obiwan_combo_plate \
                     obiwan_combo_plate_validate obiwan_combo_plate_to_print \
                     obiwan_no_floor_combo_plate_source \
                     obiwan_no_floor_combo_plate \
                     obiwan_no_floor_combo_plate_validate \
                     obiwan_no_floor_combo_plate_to_print \
                     obiwan_floor_combo_plate_source \
                     obiwan_floor_combo_plate \
                     obiwan_floor_combo_plate_validate \
                     obiwan_floor_combo_plate_to_print \
                     obiwan_ac_wing_plate_source \
                     obiwan_ac_wing_plate \
                     obiwan_ac_wing_plate_validate \
                     obiwan_ac_wing_plate_to_print \
                     obiwan_ae_wing_plate_source \
                     obiwan_ae_wing_plate \
                     obiwan_ae_wing_plate_validate \
                     obiwan_ae_wing_plate_to_print \
                     check_to_print_shelf to_print to_print_validate \
                     $(OBIWAN_01A_MODIFIER_STL) \
                     $(OBIWAN_01A_MODIFIER_CONTRACT) \
                     $(OBIWAN_01A_MODIFIER_STAMP) \
                     $(OBIWAN_01A_PETG_GF_PROJECT) \
                     $(OBIWAN_01A_PETG_GF_GCODE) \
                     $(OBIWAN_01A_PETG_GF_AUDIT) \
                     $(OBIWAN_01A_PETG_GF_STAMP) \
                     $(OBIWAN_01A_PETG_GF_PROMOTION_STAMP) \
                     $(OBIWAN_01A_PETG_GF_SHELF_OUTPUTS) \
                     $(OBIWAN_01B_PETG_GF_PROJECT) \
                     $(OBIWAN_01B_PETG_GF_GCODE) \
                     $(OBIWAN_01B_PETG_GF_AUDIT) \
                     $(OBIWAN_01B_PETG_GF_STAMP) \
                     $(OBIWAN_01B_PETG_GF_PROMOTION_STAMP) \
                     $(OBIWAN_01B_PETG_GF_SHELF_OUTPUTS) \
                     $(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP) \
                     $(OBIWAN_NO_FLOOR_COMBO_STL) \
                     $(OBIWAN_NO_FLOOR_COMBO_MANIFEST) \
                     $(OBIWAN_NO_FLOOR_COMBO_READY_STAMP) \
                     $(OBIWAN_NO_FLOOR_COMBO_READY_OUTPUTS) \
                     $(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP) \
                     $(OBIWAN_NO_FLOOR_COMBO_SHELF_OUTPUTS) \
                     $(OBIWAN_FLOOR_COMBO_SOURCE_STAMP) \
                     $(OBIWAN_FLOOR_COMBO_STL) \
                     $(OBIWAN_FLOOR_COMBO_MANIFEST) \
                     $(OBIWAN_FLOOR_COMBO_READY_STAMP) \
                     $(OBIWAN_FLOOR_COMBO_READY_OUTPUTS) \
                     $(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP) \
                     $(OBIWAN_FLOOR_COMBO_SHELF_OUTPUTS) \
                     $(OBIWAN_AC_WING_PLATE_SOURCE_STAMP) \
                     $(OBIWAN_AC_WING_PLATE_STL) \
                     $(OBIWAN_AC_WING_PLATE_MANIFEST) \
                     $(OBIWAN_AC_WING_PLATE_PREVIEW) \
                     $(OBIWAN_AC_WING_PLATE_READY_STAMP) \
                     $(OBIWAN_AC_WING_PLATE_READY_OUTPUTS) \
                     $(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP) \
                     $(OBIWAN_AC_WING_PLATE_SHELF_OUTPUTS) \
                     $(OBIWAN_AE_WING_PLATE_SOURCE_STAMP) \
                     $(OBIWAN_AE_WING_PLATE_STL) \
                     $(OBIWAN_AE_WING_PLATE_MANIFEST) \
                     $(OBIWAN_AE_WING_PLATE_PREVIEW) \
                     $(OBIWAN_AE_WING_PLATE_READY_STAMP) \
                     $(OBIWAN_AE_WING_PLATE_READY_OUTPUTS) \
                     $(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP) \
                     $(OBIWAN_AE_WING_PLATE_SHELF_OUTPUTS) \
                     $(VASE_TEBM_STOCK_3MF) $(VASE_TEBM_STOCK_GCODE) \
                     $(VASE_TEBM_STOCK_SLICE_AUDIT) \
                     $(VASE_TEBM_STOCK_SLICE_STAMP) \
                     $(VASE_TEBM_SLIM_3MF) $(VASE_TEBM_SLIM_GCODE) \
                     $(VASE_TEBM_SLIM_SLICE_AUDIT) \
                     $(VASE_TEBM_SLIM_SLICE_STAMP)
# The artifacts/ facade is a pure local packaging step: it reads already
# promoted build outputs and rewrites relative symlinks plus SHA-256
# manifests.  Dispatching it to osado would only ship a snapshot round trip
# and could publish links resolved against a worker-local tree, so detect the
# goal before the remote-first router and pin it to local execution.  It is
# deliberately absent from the candidate/all DAG.
# `iso_matrix` joins it for the same reason plus a stronger one: its README
# render set lands in images/generated/iso/, which is not one of the executor's
# promoted output roots, so a remote run would draw the PNGs on osado and
# never return them.
LOCAL_PACKAGING_GOALS := artifacts iso_matrix
PACKAGING_GOALS_REQUESTED := $(filter $(LOCAL_PACKAGING_GOALS),$(MAKECMDGOALS))
ifneq ($(strip $(PACKAGING_GOALS_REQUESTED)),)
ifneq ($(strip $(filter-out $(LOCAL_PACKAGING_GOALS),$(MAKECMDGOALS))),)
$(error local-only packaging goals cannot be mixed with CAD or other Make goals)
endif
ifeq ($(LX_CAD_EXECUTION),remote-worker)
$(error the local packaging goals are workstation-only and cannot run on a remote CAD worker)
endif
override LX_CAD_EXECUTION := local
endif

BAMBU_GOALS_REQUESTED := $(filter $(LOCAL_BAMBU_GOALS),$(MAKECMDGOALS))
ifneq ($(strip $(BAMBU_GOALS_REQUESTED)),)
ifneq ($(strip $(filter-out $(LOCAL_BAMBU_GOALS),$(MAKECMDGOALS))),)
$(error local-only Bambu goals cannot be mixed with CAD or other Make goals)
endif
ifneq ($(shell uname -s),Darwin)
ifneq ($(LX_BAMBU_GRAPH_ONLY),1)
$(error local-only Bambu goals require a Darwin/macOS host and cannot run on osado)
endif
ifeq ($(findstring n,$(firstword $(MAKEFLAGS))),)
$(error LX_BAMBU_GRAPH_ONLY=1 is valid only with make -n/--just-print)
endif
ifeq ($(filter /bin/true /usr/bin/true,$(SHELL)),)
$(error LX_BAMBU_GRAPH_ONLY=1 requires SHELL=/usr/bin/true)
endif
endif
override LX_CAD_EXECUTION := local
endif

LX_CAD_EXECUTION ?= remote
ifeq ($(filter $(LX_CAD_EXECUTION),remote local local-manifold remote-worker),)
$(error LX_CAD_EXECUTION must be remote, local, local-manifold, or remote-worker)
endif

# The ordinary interface is intentionally remote-first. The remote executor
# snapshots the exact working tree, calls this same Makefile with
# LX_CAD_EXECUTION=remote-worker on osado, and promotes only verified
# artifacts. remote-worker is internal; ordinary local builds are pinned to
# the workstation profile below.
ifeq ($(LX_CAD_EXECUTION),remote)
REMOTE_CAD_PYTHON ?= python3
REMOTE_CAD_GOALS := $(if $(strip $(MAKECMDGOALS)),$(MAKECMDGOALS),all)
.DEFAULT_GOAL := __remote_cad_dispatch
.PHONY: __remote_cad_dispatch $(MAKECMDGOALS)
__remote_cad_dispatch:
	$(REMOTE_CAD_PYTHON) scripts/remote_cad.py run $(REMOTE_CAD_GOALS)
ifneq ($(strip $(MAKECMDGOALS)),)
$(MAKECMDGOALS): __remote_cad_dispatch
	@:
endif

else

ifeq ($(LX_CAD_EXECUTION),local)
override LX_CAD_MEMORY_PROFILE := local-macos
override LX_CAD_GUARD_SLOTS := 1
override LX_CAD_ALLOW_PARALLEL := 0
else ifeq ($(LX_CAD_EXECUTION),local-manifold)
# Private promotion-only mode.  It can fan pure STL diagnostics under the
# already-authenticated 8-GiB aggregate promotion guard, but cannot be used
# to make any CAD-producing public target parallel on the workstation.
ifneq ($(strip $(MAKECMDGOALS)),_manifold_parallel)
$(error local-manifold accepts only the private _manifold_parallel target)
endif
override LX_CAD_MEMORY_PROFILE := local-macos
override LX_CAD_GUARD_SLOTS := 1
override LX_CAD_ALLOW_PARALLEL := 1
else
override LX_CAD_MEMORY_PROFILE := osado-512g
override LX_CAD_ALLOW_PARALLEL := 1
endif
export LX_CAD_MEMORY_PROFILE LX_CAD_GUARD_SLOTS
export LX_CAD_ALLOW_PARALLEL

.DELETE_ON_ERROR:

# Prefer the repository environment that carries build123d/Pillow/Shapely;
# callers can still override PYTHON explicitly. Fall back to PATH only when
# the workspace venv is absent.
LOCAL_PYTHON := $(wildcard ../.venv/bin/python)
PYTHON ?= $(if $(LOCAL_PYTHON),$(LOCAL_PYTHON),python3)
# A relative interpreter containing '/' breaks after the PNG recipes cd into
# a state directory. Bare command names remain PATH-resolved as usual.
override PYTHON_RUN := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
MU10_REFS := ../linkwitz/H1658-04_MU10RB-SL_driver.stl \
             ../linkwitz/H1658-04_MU10RB-SL_driver_STL_notes.md \
             ../linkwitz/H1658-04_MU10RB-SL_Datasheet.pdf
W22_REFS := ../E0022_W22EX001.stp
CAD_SRCS_ALL := $(wildcard src/lx521_baffle/*.py) \
        $(wildcard src/lx521_baffle/proud/*.py) \
        $(wildcard src/lx521_baffle/obiwan/*.py)
# These modules are consumed only by the Obi-Wan release family.  Keep the
# legacy/common dependency set conservative (it still contains the shared
# Obi-Wan route/floor stack reached through src/lx521_baffle/cables.py), while
# preventing a carrier/web/split-only edit from invalidating every B/V1/V1L
# STEP and STL.  GNU Make remains the dependency authority.
OBIWAN_EXCLUSIVE_CAD_SRCS := \
        src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/obiwan/bumps.py \
        src/lx521_baffle/obiwan/closure_webs.py \
        src/lx521_baffle/obiwan/joints.py \
        src/lx521_baffle/obiwan/magnets.py \
        src/lx521_baffle/obiwan/rear_entry.py \
        src/lx521_baffle/obiwan/assembled.py \
        src/lx521_baffle/obiwan/attachments.py \
        src/lx521_baffle/obiwan/bridge.py \
        src/lx521_baffle/obiwan/lm_split.py \
        src/lx521_baffle/obiwan/split.py \
        src/lx521_baffle/obiwan/wings.py
SHARED_CAD_SRCS := src/lx521_baffle/assembly.py src/lx521_baffle/magnet_contract.py \
        src/lx521_baffle/geom.py src/lx521_baffle/io.py
# GEOM_SRCS is the geometry authority every non-Obi-Wan CAD recipe imports:
# the proud/common modules plus the reference solids they measure against.
# Its content reaches STEP/STL/PNG bytes, so it stays a prerequisite of them.
GEOM_SRCS := $(filter-out $(OBIWAN_EXCLUSIVE_CAD_SRCS),$(CAD_SRCS_ALL)) \
        src/lx521_baffle/magnets.py $(SHARED_CAD_SRCS) \
        $(MU10_REFS) $(W22_REFS)
OBIWAN_SRCS := $(GEOM_SRCS) $(OBIWAN_EXCLUSIVE_CAD_SRCS)
EXECUTION_SRCS := scripts/run_memory_guarded.py
# This Makefile is deliberately NOT a prerequisite of ordinary CAD artifacts.
# Make cannot detect a recipe-text edit in any case, and listing it here put
# 258 targets (the whole ~35 min remote CAD set) behind every edit to this
# file -- including a new comment or an unrelated new rule.  It remains a
# prerequisite of exactly the nodes whose *content* embeds Makefile bytes or
# Makefile-defined names:
#   * the two obiwan_release_manifest.json ledgers, because
#     write_obiwan_release_manifest.py::generation_source_paths() hashes this
#     file into source_hashes();
#   * the combo-plate and wing-plate source stamps, whose plate NAME
#     variables are baked into the emitted plate manifests;
#   * check_remote_transport and check_release_metadata, which parse this
#     file's text directly.
# A structural edit that changes a recipe's meaning without changing any of
# those (for example flipping a $(call RULES,...) stand-foot argument) still
# requires an explicit `make clean`, exactly as before this split.
BUILD_POLICY_SRCS := Makefile
PRINT_CONTRACT_SRCS := src/lx521_baffle/print_contract.py src/lx521_baffle/io.py
SLICER_SRCS := scripts/slice_captive_magnets.py \
        scripts/release_validation.py scripts/gcode_analysis.py \
        scripts/artifact_emit.py
# obiwan_release_manifest.json records a sha256 of every file returned by
# write_obiwan_release_manifest.py::generation_source_paths(), so this group
# must mirror that function exactly -- each listed file's *bytes* appear in
# the manifest, which is why the broad wildcards are correct here and only
# here.  Keep the two in step: src/lx521_baffle rglob, scripts/*.py,
# tests/test_*.py, the lockfile, this Makefile, the qualification record and
# the reference solids.  CAD_SRCS_ALL already spans the whole src package
# (it has no subpackages below proud/ and obiwan/).
OBIWAN_RELEASE_SRCS := $(CAD_SRCS_ALL) \
        $(wildcard scripts/*.py) \
        $(wildcard tests/test_*.py) \
        $(BUILD_POLICY_SRCS) cad-remote-requirements.lock \
        docs/obiwan_physical_qualification.md \
        $(MU10_REFS) $(W22_REFS)
# export_piece_stls imports the strict checker after meshing, and that checker
# imports the release-manifest helpers at module load.  Keep every STL stamp
# source-complete so a checker/contract change cannot reuse a stale mesh stamp.
STL_EXPORT_SRCS := scripts/export_piece_stls.py scripts/check_manifold.py \
        scripts/obiwan_support_blocker.py \
        src/lx521_baffle/stl_export.py $(PRINT_CONTRACT_SRCS) \
        scripts/write_obiwan_release_manifest.py
# Command-line Make variables must not turn a candidate/release recipe into
# an unguarded OCC process. PYTHON remains selectable, but every invocation
# is forced through this workspace's authenticated process-tree guard.
override GUARD := $(abspath scripts/run_memory_guarded.py)
override RUN := $(PYTHON_RUN) $(GUARD) -- $(PYTHON_RUN)
RUN_STATE = LX_STAND_FOOT=$(1) LX_ROUTING_PROFILE=$(2) $(RUN)
# The remote guard runs two slot pools: a small number of heavy slots sized
# for the 10-16 GiB OCC recipes, and many more light slots at a 6 GiB cap.
# Recipes are heavy unless they declare otherwise, so this prefix is opt-in
# and only for classes whose measured peak RSS in
# .remote-cad/jobs/*/profile.json leaves real headroom under that cap:
#
#   check_manifold.py stamps       0.6 GiB (116 of them)
#   test_clearances.py checks      0.5-4.0 GiB (37 of them)
#   junction-closure dense shards  1.2 GiB (16 of them)
#   routing/overlay/design-map PNG 0.5-0.6 GiB
#   release manifest / catalog     0.02-0.5 GiB
#
# Locally the light pool is disabled outright, so this declaration is inert
# on a workstation and the build stays strictly serial.
LIGHT_GUARD := LX_CAD_GUARD_WEIGHT=light
RUN_LIGHT := $(LIGHT_GUARD) $(RUN)
# State flags stay leftmost so a recipe's rendered prefix is unchanged apart
# from the added weight declaration.
RUN_STATE_LIGHT = LX_STAND_FOOT=$(1) LX_ROUTING_PROFILE=$(2) $(RUN_LIGHT)

# Local builds remain strictly serial. The remote executor may opt into
# several guarded slots only under the large-host profile; the enclosing
# systemd cgroup still limits their aggregate memory to 512 GiB.
ifeq ($(LX_CAD_EXECUTION),local)
.NOTPARALLEL:
endif

GENS := b2_split b2 \
        b1 a_comp \
        b1_assembled a_comp_assembled \
        v1 v1_split \
        v1l_split obiwan_split \
        obiwan_lm_split \
        obiwan_attachments \
        obiwan_assembled \
        um_fit \
        v1_attachments

OVERLAY_PNGS = baffle_b1_drivers.png baffle_b2_drivers.png \
               baffle_variants_drivers.png
FLOOR_STRENGTH_REPORTS = build/floor_stand/obiwan_integrated_floor_strength.json \
                         build/floor_stand/obiwan_integrated_floor_strength.md
OBIWAN_WING_DESIGN_MAP = build/common/obiwan_wing_design_map.png
OBIWAN_WING_SLUGS = ac ae
OBIWAN_WING_STAMPS = $(addprefix build/wings/.stamp_,$(OBIWAN_WING_SLUGS))
OBIWAN_WING_INPUTS = cad-remote-requirements.lock \
        docs/obiwan_acoustic_wings_spec.md src/lx521_baffle/magnets.py \
        src/lx521_baffle/floor_bend.py \
        src/lx521_baffle/assembly.py src/lx521_baffle/magnet_contract.py src/lx521_baffle/geom.py \
        src/lx521_baffle/print_contract.py src/lx521_baffle/io.py src/lx521_baffle/stl_export.py \
        scripts/export_steps.py scripts/gen_driver_overlay.py scripts/export_obiwan_staged.py \
        src/lx521_baffle/base.py src/lx521_baffle/proud/b.py \
        src/lx521_baffle/proud/b1.py src/lx521_baffle/proud/b2.py \
        src/lx521_baffle/cables.py src/lx521_baffle/flush.py \
        src/lx521_baffle/proud/v1.py src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/obiwan/attachments.py \
        src/lx521_baffle/obiwan/route.py \
        src/lx521_baffle/obiwan/bumps.py \
        src/lx521_baffle/obiwan/rear_entry.py \
        src/lx521_baffle/proud/a_comp.py \
        src/lx521_baffle/obiwan/bridge.py \
        src/lx521_baffle/obiwan/floor.py \
        src/lx521_baffle/obiwan/floor_strength.py \
        src/lx521_baffle/obiwan/lm_split.py \
        build/floor_stand/.obiwan_stage/manifest.json \
        build/no_floor_stand/.obiwan_stage/manifest.json

# Strict STL topology is embarrassingly parallel, but the former checker ran
# every mesh in one Python process.  A recursive submake is used deliberately:
# it reparses only after the producer targets have populated the requested
# roots, inherits GNU Make's jobserver, and exposes one ordinary stamp target
# per STL.  There is no second scheduler or opaque cache.  The final phony
# node retains the manifest/review/cross-state gate after all mesh stamps pass.
MANIFOLD_ROOTS ?= build/floor_stand/stl build/no_floor_stand/stl
MANIFOLD_OBIWAN_ONLY ?= 0
MANIFOLD_STAMP_DIR ?= .manifold-stamps
CHECK_STAMP_DIR ?= .check-stamps
ifeq ($(MANIFOLD_OBIWAN_ONLY),1)
# A warm focused Obi-Wan job inherits verified legacy meshes from the Make cache.
# Expand only filenames that actually belong to Obi-Wan; the metadata-only join
# still enforces the Obi-Wan inventory.  GNU Make remains the only scheduler.
MANIFOLD_STLS := $(foreach root,$(MANIFOLD_ROOTS),\
        $(wildcard $(root)/*obiwan*.stl))
else
MANIFOLD_STLS := $(foreach root,$(MANIFOLD_ROOTS),$(wildcard $(root)/*.stl))
endif
manifold_stamp = $(MANIFOLD_STAMP_DIR)/$(1).ok
MANIFOLD_STAMPS := $(foreach stl,$(MANIFOLD_STLS),$(call manifold_stamp,$(stl)))

# These names mirror the authoritative Python registries.  Exposing every
# selector as an ordinary Make node lets the GNU Make jobserver own total
# concurrency across regression checks and artifact writers.  The Python
# suite-level ThreadPool runners remain useful for direct CLI use, but are not
# invoked by `make check`/`make candidate`.
CLEARANCE_CHECK_NAMES := \
        test_foot_lane_webs \
        test_stock_slim_floor_bend_lateral_envelope \
        test_variant_outlines_splice \
        test_v1l_service_envelope \
        test_emboss_driver_keepouts \
        test_attachment_flushness \
        test_magnet_top_site_walls \
        test_magnet_cavities_vs_t_ducts \
        test_ts_captive_keepout_nudge \
        test_stock_captive_magnet_contract \
        test_stock_local_magnet_backing \
        test_individual_attachment_magnet_burial \
        test_duct_vs_w22_pilots \
        test_duct_duct_separation \
        test_duct_vs_um_pilots \
        test_seam_keys_vs_ducts \
        test_shared_lm_clock_and_hidden_seam_b_m3 \
        test_v1_field \
        test_route_smoothness \
        test_um_tail_has_no_curvature_reversal \
        test_lm_exit_curvature_and_fishing \
        test_v1l_um_terminal_axis_handoff \
        test_um_eroded_outline_containment \
        test_ts_eroded_outline_containment \
        test_stock_slim_ts_full_section_containment \
        test_stock_slim_curve_optimization_contracts \
        test_stock_slim_no_floor_full_duct_containment \
        test_stock_slim_no_floor_bottom_topology \
        test_bridge_inserts \
        test_margin_dashboard \
        test_cutter_health \
        test_v1l_mid_right_terminal_duct_topology \
        test_v1l_upper_dovetail_depth_mid_left \
        test_v1l_upper_dovetail_depth_mid_right

R6F_CASE_IDS := \
        route_contract \
        w22_reference_step_geometry \
        insert_bump_clearance \
        floor_insert_bump_clearance \
        no_floor_route_smoothness \
        floor_route_smoothness \
        bump_brep_clearance \
        floor_bump_brep_clearance \
        bump_backfill_contract \
        floor_bump_backfill_contract \
        lm_burial_web_contract \
        floor_lm_burial_web_contract \
        um_burial_web_contract \
        floor_um_burial_web_contract \
        feed_and_flush_mouth_contract \
        floor_feed_and_flush_mouth_contract \
        crossover_brep \
        floor_crossover_brep \
        bridge_contract \
        bridge_geometry \
        joint_load_contract \
        um_driver_spoke_is_separate_from_lm_um_insert_ear \
        floor_lm_core \
        no_floor_lm_core \
        floor_lm_keyed_split \
        no_floor_lm_keyed_split \
        floor_um_shell \
        floor_t_shell \
        no_floor_um_shell \
        no_floor_t_shell \
        lm_cable_clearance \
        um_cable_clearance \
        floor_lm_cable_clearance \
        floor_um_cable_clearance \
        floor_integrated_mount \
        tweeter_and_service \
        floor_tweeter_and_service

R6F_SERVICE_CASE_IDS := tweeter_and_service floor_tweeter_and_service
R6F_FLOOR_CASE_IDS := $(filter floor_%,$(R6F_CASE_IDS))
R6F_NO_FLOOR_CASE_IDS := $(filter-out $(R6F_FLOOR_CASE_IDS),$(R6F_CASE_IDS))
r6f_case_stage_manifest = $(if $(filter $(1),$(R6F_FLOOR_CASE_IDS)),\
        build/floor_stand/.obiwan_stage/manifest.json,\
        build/no_floor_stand/.obiwan_stage/manifest.json)
R6F_PRIVATE_ENV_NAMES := \
        LX_R6F_EXPORT_CARRIER LX_R6F_EXPORT_PATH LX_R6F_EXPORT_SHELL \
        LX_R6F_EXPORT_SHELL_SEGMENT LX_R6F_EXPORT_SHELL_SEGMENT_COUNT \
        LX_R6F_EXPORT_SHELL_WALL LX_R6F_EXPORT_TWEETER \
        LX_R6F_LM_INPUT_PATH LX_R6F_SERVICE_COMPONENT_CHUNK_COUNT \
        LX_R6F_SERVICE_COMPONENT_COLLISION LX_R6F_SERVICE_COMPONENT_EXPORT \
        LX_R6F_SERVICE_COMPONENT_PATH LX_R6F_SERVICE_COMPONENT_SELECTOR \
        LX_R6F_SERVICE_PHASE LX_R6F_SERVICE_PRINTED_PATH \
        LX_R6F_SERVICE_START_PHASE LX_R6F_SERVICE_STOP_AFTER_PHASE \
        LX_R6F_CASE_ID LX_R6F_SUBTRACT_INPUT LX_R6F_SUBTRACT_OUTPUT \
        LX_R6F_SUBTRACT_PART LX_R6F_VALIDATE_CABLE \
        LX_R6F_VALIDATE_CABLE_MODE LX_R6F_VALIDATE_CABLE_SEGMENT \
        LX_R6F_VALIDATE_CABLE_SEGMENT_COUNT LX_R6F_VALIDATE_CARRIER_OWNER \
        LX_R6F_VALIDATE_CARRIER_PATH LX_R6F_VALIDATE_PART_PATHS \
        LX_R6F_VALIDATE_SHELL LX_R6F_VALIDATE_SHELL_PATH \
        LX_R6F_VALIDATE_SHELL_PATHS
R6F_CLEAN_ENV = env $(foreach name,$(R6F_PRIVATE_ENV_NAMES),-u $(name))
CLEARANCE_CHECK_TARGETS := $(addprefix _check_clearance_,$(CLEARANCE_CHECK_NAMES))
R6F_CASE_TARGETS := $(addprefix _check_r6f_,$(R6F_CASE_IDS))
clearance_check_stamp = $(CHECK_STAMP_DIR)/clearance/$(1).ok
r6f_check_stamp = $(CHECK_STAMP_DIR)/r6f/$(1).ok
CLEARANCE_CHECK_STAMPS := $(foreach name,$(CLEARANCE_CHECK_NAMES),\
        $(call clearance_check_stamp,$(name)))
R6F_CASE_STAMPS := $(foreach name,$(R6F_CASE_IDS),\
        $(call r6f_check_stamp,$(name)))
JUNCTION_CLOSURE_STAMP := $(CHECK_STAMP_DIR)/obiwan_junction_closures.ok
OBIWAN_WING_ARTIFACT_CHECK_STAMP := \
        $(CHECK_STAMP_DIR)/obiwan_wings/artifacts.ok
OBIWAN_WING_AC_LIVE_CHECK_STAMP := \
        $(CHECK_STAMP_DIR)/obiwan_wings/live_ac.ok
OBIWAN_WING_AE_LIVE_CHECK_STAMP := \
        $(CHECK_STAMP_DIR)/obiwan_wings/live_ae.ok
OBIWAN_WING_CHECK_STAMPS := \
        $(OBIWAN_WING_ARTIFACT_CHECK_STAMP) \
        $(OBIWAN_WING_AC_LIVE_CHECK_STAMP) \
        $(OBIWAN_WING_AE_LIVE_CHECK_STAMP)
JUNCTION_CLOSURE_BASE_STAMP := \
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/base.ok
junction_closure_base_state_stamp = \
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/base_$(1).ok
JUNCTION_CLOSURE_BASE_STATE_STAMPS := \
        $(call junction_closure_base_state_stamp,no_floor) \
        $(call junction_closure_base_state_stamp,floor)
JUNCTION_CLOSURE_DENSE_STATES := no_floor floor
JUNCTION_CLOSURE_DENSE_SHARDS := 0 1 2 3 4 5 6 7
junction_closure_dense_stamp = \
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/$(1)_$(2).ok
JUNCTION_CLOSURE_DENSE_STATE_STAMPS := \
        $(foreach state,$(JUNCTION_CLOSURE_DENSE_STATES),\
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/state_$(state).ok)
JUNCTION_CLOSURE_DENSE_STAMPS := \
        $(foreach state,$(JUNCTION_CLOSURE_DENSE_STATES),\
        $(foreach shard,$(JUNCTION_CLOSURE_DENSE_SHARDS),\
        $(call junction_closure_dense_stamp,$(state),$(shard))))

# A remote cache is keyed by the attested Python environment, verifies the
# entire seeded tree, and overlays changed sources newer than all targets.
# Therefore its check stamps are safe ordinary Make targets.  Local trees can
# outlive an environment change outside Make's prerequisite graph, so retain
# the conservative always-revalidate behavior there.
ifeq ($(LX_CAD_EXECUTION),remote-worker)
CHECK_REVALIDATE :=
else
.PHONY: _force_check_revalidation
CHECK_REVALIDATE := _force_check_revalidation
endif

# The stamp body is empty and the check is a pure function of the STL plus the
# checker sources, so this Makefile supplies only the MANIFOLD_STLS target
# list -- and Make already reacts to that by target identity.  Listing it as a
# prerequisite here re-ran all 116 mesh checks after any edit to this file.
define MANIFOLD_STL_RULE
$(call manifold_stamp,$(1)): $(1) scripts/check_manifold.py \
        $(PRINT_CONTRACT_SRCS) scripts/write_obiwan_release_manifest.py
	@mkdir -p $$(@D)
	$(RUN_LIGHT) scripts/check_manifold.py --stl-only $(1)
	@touch $$@
endef

$(foreach stl,$(MANIFOLD_STLS),$(eval $(call MANIFOLD_STL_RULE,$(stl))))

define CLEARANCE_CHECK_RULE
_check_clearance_$(1): $(call clearance_check_stamp,$(1))
$(call clearance_check_stamp,$(1)): tests/test_clearances.py $(OBIWAN_SRCS) \
        $(EXECUTION_SRCS) $(CHECK_REVALIDATE)
	@mkdir -p $$(@D)
	LX_CLEARANCE_SINGLE_CHECK=$(1) $(RUN_LIGHT) tests/test_clearances.py
	@touch $$@
endef

define R6F_CASE_RULE
_check_r6f_$(1): $(call r6f_check_stamp,$(1))
$(call r6f_check_stamp,$(1)): tests/test_obiwan_r6f.py tests/test_harness.py $(OBIWAN_SRCS) \
		$(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		$(call r6f_case_stage_manifest,$(1))
	@mkdir -p $$(@D)
	$(R6F_CLEAN_ENV) LX_R6F_CASE_ID=$(1) $(RUN) tests/test_obiwan_r6f.py
	@touch $$@
endef

# On a workstation these two checks deliberately remain lightweight parent
# orchestrators so every service substep gets a fresh guarded lifecycle.  On
# osado the selected process self-wraps once and performs the full large-host
# matrix under that one Make token.
define R6F_SERVICE_CASE_RULE
_check_r6f_$(1): $(call r6f_check_stamp,$(1))
$(call r6f_check_stamp,$(1)): tests/test_obiwan_r6f.py tests/test_harness.py $(OBIWAN_SRCS) \
		$(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		$(call r6f_case_stage_manifest,$(1))
	@mkdir -p $$(@D)
	$(R6F_CLEAN_ENV) LX_R6F_CASE_ID=$(1) \
	    $(PYTHON_RUN) tests/test_obiwan_r6f.py
	@touch $$@
endef

$(foreach name,$(CLEARANCE_CHECK_NAMES),\
    $(eval $(call CLEARANCE_CHECK_RULE,$(name))))
$(foreach name,$(filter-out $(R6F_SERVICE_CASE_IDS),$(R6F_CASE_IDS)),\
    $(eval $(call R6F_CASE_RULE,$(name))))
$(foreach name,$(R6F_SERVICE_CASE_IDS),\
    $(eval $(call R6F_SERVICE_CASE_RULE,$(name))))

# The keyed-split acceptance cases open the exported core STLs through the
# print-sidecar authority, so on a truly cold tree they must order behind
# the state's Obi-Wan STL export, not merely share a public aggregate with
# it.  Prerequisite-only lines compose with the generated case rules.
.check-stamps/r6f/no_floor_lm_keyed_split.ok: build/no_floor_stand/stl/.stamp_obiwan
.check-stamps/r6f/floor_lm_keyed_split.ok: build/floor_stand/stl/.stamp_obiwan

VARIANT_TARGETS = $(1)/stl/.stamp_coupon $(1)/stl/.stamp $(1)/stl/.stamp_v1 $(1)/stl/.stamp_v1l $(1)/stl/.stamp_obiwan \
                  $(addprefix $(1)/,$(addsuffix .step,$(GENS))) \
                  $(1)/baffle_cable_routing_proud.png \
                  $(1)/baffle_cable_routing_obiwan.png \
                  $(addprefix $(1)/,$(OVERLAY_PNGS)) \
                  $(1)/obiwan_release_manifest.json

# The long public phony surface is intentional. Besides release joins, it owns
# remotely whitelisted iteration entry points for mouth/burial/backfill/shell,
# LM-profile/split, bump-BREP, focused mesh, catalog, and Bambu diagnostics.
# Generated `_check_*` aliases are also phony so a named iteration request
# always reaches its ordinary content-addressed stamp prerequisite.
.PHONY: all candidate release validate_obiwan_stages \
        validate_floor_obiwan_stage validate_no_floor_obiwan_stage \
        floor_stand floor_stand_artifacts floor_obiwan \
        no_floor_stand no_floor_stand_artifacts no_floor_obiwan \
        obiwan_state_releases \
        no_floor_obiwan_01a obiwan_release \
        obiwan_wings obiwan_wing_exports obiwan_wing_artifacts \
        check_obiwan_wings \
        common artifacts iso_matrix \
        check check_static check_shared_contracts check_remote_transport \
        check_release_metadata check_slice_metadata \
        check_clearance_suite check_r6f_suite \
        check_captive_magnets check_obiwan \
        check_obiwan_shells check_obiwan_t_shells \
        check_obiwan_service check_obiwan_closure_focus \
        check_obiwan_um_pilot_spoke \
        check_obiwan_mouths check_obiwan_burial check_obiwan_backfills \
        check_obiwan_um_burial \
        check_obiwan_route_boundaries \
        check_floor_um_shell check_floor_t_shell \
        check_no_floor_um_shell check_no_floor_t_shell \
        check_floor_obiwan_mouths check_no_floor_obiwan_mouths \
        check_no_floor_obiwan_mouths_focused \
        check_floor_obiwan_burial check_no_floor_obiwan_burial \
        check_floor_obiwan_um_burial check_no_floor_obiwan_um_burial \
        check_floor_obiwan_backfills check_no_floor_obiwan_backfills \
        check_route_contract check_bump_brep check_floor_integrated_mount \
        check_no_floor_lm_mesh check_obiwan_lm_split \
        check_obiwan_lm_split_two_pin_static \
        check_obiwan_junction_closure_plans \
        check_obiwan_junction_closure_base \
        check_obiwan_junction_closures \
        check_floor_obiwan_lm_split check_no_floor_obiwan_lm_split \
        check_obiwan_lm_profile check_obiwan_lm_profile_plan \
        check_obiwan_lm_profile_brep manifold _manifold_parallel clean \
        check_bambu_3mf_audit bambu_profiles bambu_dry_run \
        bambu_slice_obiwan bambu_slice_release \
        vase_tebm35c10_4_cad vase_tebm35c10_4_3mf \
        vase_tebm35c10_4_3mf_validate \
        vase_tebm35c10_4_stock_cad vase_tebm35c10_4_slim_cad \
        vase_tebm35c10_4_stock_3mf vase_tebm35c10_4_slim_3mf \
        vase_tebm35c10_4_stock_3mf_validate \
        vase_tebm35c10_4_slim_3mf_validate \
        obiwan_no_floor_petg_gf_modifier \
        obiwan_no_floor_petg_gf_01a \
        obiwan_no_floor_petg_gf_01a_validate \
        obiwan_no_floor_petg_gf_01a_to_print \
        obiwan_floor_petg_gf_01b \
        obiwan_floor_petg_gf_01b_validate \
        obiwan_floor_petg_gf_01b_to_print \
        obiwan_combo_plate_source obiwan_combo_plate \
        obiwan_combo_plate_validate obiwan_combo_plate_to_print \
        obiwan_no_floor_combo_plate_source obiwan_no_floor_combo_plate \
        obiwan_no_floor_combo_plate_validate \
        obiwan_no_floor_combo_plate_to_print \
        obiwan_floor_combo_plate_source obiwan_floor_combo_plate \
        obiwan_floor_combo_plate_validate \
        obiwan_floor_combo_plate_to_print \
        obiwan_ac_wing_plate_source obiwan_ac_wing_plate \
        obiwan_ac_wing_plate_validate obiwan_ac_wing_plate_to_print \
        obiwan_ae_wing_plate_source obiwan_ae_wing_plate \
        obiwan_ae_wing_plate_validate obiwan_ae_wing_plate_to_print \
        refresh_captive_magnet_catalog_existing \
        check_to_print_shelf to_print to_print_validate \
        $(CLEARANCE_CHECK_TARGETS) $(R6F_CASE_TARGETS)
all: validate_obiwan_stages floor_stand_artifacts no_floor_stand_artifacts common \
        obiwan_wings $(CAPTIVE_MAGNET_CATALOG)
	# Each state already completed its one-node-per-STL sweep while the other
	# state was still eligible to use the shared jobserver.  Keep only the
	# inexpensive cross-state manifest/review consistency gate on the join.
	$(RUN_LIGHT) scripts/check_manifold.py --metadata-only \
	    build/floor_stand/stl build/no_floor_stand/stl

# Optional opposed-BMR Stock/Slim vase family.  This target is intentionally
# independent of the protected floor/no-floor release roots and is therefore
# cheap to dispatch, cache, and atomically promote as one complete remote
# root.  Both children use the same generator and differ only in their
# released rear envelope; both retain the regular seam-B female dovetails.
VASE_TEBM_CAD_SRCS := \
        src/lx521_baffle/proud/vase_tebm35c10_4.py \
        src/lx521_baffle/proud/b.py \
        src/lx521_baffle/proud/b2.py \
        src/lx521_baffle/proud/b2_split.py \
        src/lx521_baffle/base.py src/lx521_baffle/cables.py \
        src/lx521_baffle/magnets.py src/lx521_baffle/magnet_contract.py \
        src/lx521_baffle/print_contract.py src/lx521_baffle/stl_export.py \
        src/lx521_baffle/geom.py src/lx521_baffle/io.py \
        scripts/export_vase_tebm35c10_4.py scripts/export_piece_stls.py \
        scripts/export_steps.py scripts/check_manifold.py \
        scripts/gen_vase_tebm35c10_4_slicing_profile.py \
        scripts/release_validation.py scripts/json_schema_subset.py \
        captive_magnet_release_catalog.schema.json \
        captive_magnet_slicing_profile.json \
        tests/test_vase_tebm35c10_4.py $(EXECUTION_SRCS)

$(VASE_TEBM_CAD_STAMP): $(VASE_TEBM_CAD_SRCS)
	@rm -f "$@"
	# Remove the superseded Stock-only flat-root delivery.  Stock and Slim now
	# have independent canonical children; retaining the old flat files would
	# create a third, stale authority for the same artifact name.
	@rm -f "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).brep" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).step" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).stl" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).print.json" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).facts.json" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).catalog.json" \
	    "$(VASE_TEBM_ROOT)/$(VASE_TEBM_NAME).slicing_profile.json" \
	    "$(VASE_TEBM_ROOT)/cad_manifest.json" \
	    "$(VASE_TEBM_ROOT)/.stamp_cad_validated" \
	    "$(VASE_TEBM_ROOT)/.$(VASE_TEBM_NAME).step.glb"
	@rm -rf "$(VASE_TEBM_ROOT)/slice_audit"
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=proud $(RUN) \
	    tests/test_vase_tebm35c10_4.py
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=proud $(RUN) \
	    scripts/export_vase_tebm35c10_4.py \
	        --profile stock --output-root "$(VASE_TEBM_STOCK_ROOT)"
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=proud $(RUN) \
	    scripts/export_vase_tebm35c10_4.py \
	        --profile slim --output-root "$(VASE_TEBM_SLIM_ROOT)"
	$(PYTHON_RUN) scripts/gen_vase_tebm35c10_4_slicing_profile.py \
	    --base captive_magnet_slicing_profile.json \
	    --profile stock --output "$(VASE_TEBM_STOCK_PROFILE)"
	$(PYTHON_RUN) scripts/gen_vase_tebm35c10_4_slicing_profile.py \
	    --base captive_magnet_slicing_profile.json \
	    --profile slim --output "$(VASE_TEBM_SLIM_PROFILE)"
	@for path in $(VASE_TEBM_CAD_OUTPUTS) \
	        $(VASE_TEBM_STOCK_EXPORT_STAMP) \
	        $(VASE_TEBM_SLIM_EXPORT_STAMP); do \
		test -s "$$path" || exit 1; \
	done
	@touch "$@"

$(VASE_TEBM_CAD_OUTPUTS): | $(VASE_TEBM_CAD_STAMP)
	@test -s "$@" || { \
		rm -f "$(VASE_TEBM_CAD_STAMP)"; \
		$(MAKE) --no-print-directory "$(VASE_TEBM_CAD_STAMP)"; \
		test -s "$@"; \
	}

vase_tebm35c10_4_cad: $(VASE_TEBM_CAD_OUTPUTS)
	@echo "Stock BREP: $(VASE_TEBM_STOCK_BREP)"
	@echo "Stock STEP: $(VASE_TEBM_STOCK_STEP)"
	@echo "Stock STL:  $(VASE_TEBM_STOCK_STL)"
	@echo "Slim BREP:  $(VASE_TEBM_SLIM_BREP)"
	@echo "Slim STEP:  $(VASE_TEBM_SLIM_STEP)"
	@echo "Slim STL:   $(VASE_TEBM_SLIM_STL)"
	@echo "both catalogs/profiles ready for local slicing"

vase_tebm35c10_4_stock_cad: $(VASE_TEBM_STOCK_CAD_OUTPUTS)
	@echo "Stock BMR vase: $(VASE_TEBM_STOCK_STEP)"

vase_tebm35c10_4_slim_cad: $(VASE_TEBM_SLIM_CAD_OUTPUTS)
	@echo "Slim BMR vase: $(VASE_TEBM_SLIM_STEP)"

# This catalog is generated only after all 58 released magnet-bearing meshes
# and their exact front-down print transforms exist.  It is the CAD-side
# authority consumed by the local, no-printer-contact Bambu slicing audit.
$(CAPTIVE_MAGNET_CATALOG): scripts/generate_captive_magnet_catalog.py \
        captive_magnet_release_catalog.schema.json $(SLICER_SRCS) \
        scripts/json_schema_subset.py src/lx521_baffle/magnets.py $(SHARED_CAD_SRCS) \
        src/lx521_baffle/stl_export.py \
        build/floor_stand/stl/.stamp build/floor_stand/stl/.stamp_v1 \
        build/floor_stand/stl/.stamp_v1l build/floor_stand/stl/.stamp_obiwan \
        build/floor_stand/stl/.stamp_coupon \
        build/no_floor_stand/stl/.stamp build/no_floor_stand/stl/.stamp_v1 \
        build/no_floor_stand/stl/.stamp_v1l build/no_floor_stand/stl/.stamp_obiwan \
        build/no_floor_stand/stl/.stamp_coupon $(OBIWAN_WING_STAMPS)
	@mkdir -p review
	$(call RUN_STATE_LIGHT,0,obiwan,.) \
	    scripts/generate_captive_magnet_catalog.py --output $@
	@test -s $@

# Iteration-only catalog refresh over the already exported, remotely verified
# release inventory.  This deliberately has no mesh prerequisites: it updates
# hash/provenance bindings after a focused CAD export without rebuilding the
# unaffected floor, wing, and legacy artifacts.  The generator still applies
# the complete 58-artifact/94-station fail-closed catalog contract.
refresh_captive_magnet_catalog_existing: scripts/generate_captive_magnet_catalog.py \
        captive_magnet_release_catalog.schema.json $(SLICER_SRCS) \
        scripts/json_schema_subset.py src/lx521_baffle/magnets.py $(SHARED_CAD_SRCS) \
        src/lx521_baffle/stl_export.py
	@mkdir -p review
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/generate_captive_magnet_catalog.py --output $(CAPTIVE_MAGNET_CATALOG)
	@test -s $(CAPTIVE_MAGNET_CATALOG)

# Runtime package versions are part of the native-stage fingerprint but not
# filesystem prerequisites. Validate first on every build; the stage command
# is a no-write fast path when current and atomically rebuilds only on drift.
validate_obiwan_stages: validate_floor_obiwan_stage validate_no_floor_obiwan_stage

# $(1) = public state prefix, $(2) = output root, $(3) = LX_STAND_FOOT.
# The manifest is an alias for the always-validated transaction so parallel
# consumers share one stage node instead of racing a second file recipe.
define OBIWAN_STAGE_RULE
validate_$(1)_obiwan_stage: scripts/export_obiwan_staged.py scripts/export_steps.py \
		$$(OBIWAN_SRCS)
	@mkdir -p $(2)/.obiwan_stage
	$$(call RUN_STATE,$(3),obiwan,.) scripts/export_obiwan_staged.py stage --manifest $(2)/.obiwan_stage/manifest.json

$(2)/.obiwan_stage/manifest.json: validate_$(1)_obiwan_stage
	@test -f $$@
endef

$(eval $(call OBIWAN_STAGE_RULE,floor,build/floor_stand,1))
$(eval $(call OBIWAN_STAGE_RULE,no_floor,build/no_floor_stand,0))

candidate: check all

release: candidate
	# Candidate already completed the strict mesh sweep.  Release adds only the
	# fail-closed physical-authorization and metadata gate.
	$(RUN_LIGHT) scripts/check_manifold.py --metadata-only --require-release-authorized

manifold: scripts/check_manifold.py
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/floor_stand/stl build/no_floor_stand/stl'

_manifold_parallel: $(MANIFOLD_STAMPS)
	@test -n "$(strip $(MANIFOLD_STLS))" || { \
	    echo "no STLs found under: $(MANIFOLD_ROOTS)" >&2; exit 1; }
	$(RUN_LIGHT) scripts/check_manifold.py --metadata-only \
	    $(if $(filter 1,$(MANIFOLD_OBIWAN_ONLY)),--obiwan-only) \
	    $(MANIFOLD_ROOTS)
	@echo "$(words $(MANIFOLD_STLS))/$(words $(MANIFOLD_STLS)) STLs strict-manifold (Make-parallel)"

# polar-measurement turntable (mates under the floor-stand foot only)
floor_stand_artifacts: validate_floor_obiwan_stage build/floor_stand/stl/.stamp_polar
build/floor_stand/stl/.stamp_polar: scripts/polar_index_base.py
	@rm -f $@
	$(RUN) scripts/polar_index_base.py
	@touch $@
floor_stand_artifacts: $(call VARIANT_TARGETS,build/floor_stand) \
        $(FLOOR_STRENGTH_REPORTS)
	# Start this state's topology fan-out as soon as its final artifact lands;
	# the recursive Make inherits the enclosing GNU Make jobserver.
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/floor_stand/stl'
floor_stand: floor_stand_artifacts

# Focused atomic state releases emit the complete hash-backed Obi-Wan family
# without rebuilding unrelated proud/V1/V1L print sets.
# $(1) = public state prefix, $(2) = output root.
define FOCUSED_OBIWAN_RELEASE_RULE
$(1)_obiwan: validate_$(1)_obiwan_stage \
		$(2)/obiwan_release_manifest.json
	$$(MAKE) --no-print-directory _manifold_parallel MANIFOLD_ROOTS='$(2)/stl' MANIFOLD_OBIWAN_ONLY=1
endef

$(eval $(call FOCUSED_OBIWAN_RELEASE_RULE,floor,build/floor_stand))
$(eval $(call FOCUSED_OBIWAN_RELEASE_RULE,no_floor,build/no_floor_stand))

# One remote-safe join lets both focused state releases share the bounded
# Make jobserver without expanding into the unrelated proud/V1/V1L DAG.
obiwan_state_releases: floor_obiwan no_floor_obiwan

# Focused iteration/export contract for the only no-floor part whose cable
# routing owns the D20 entry cluster and LM-recess crossing.  The native stage
# remains the STEP-first authority; only 01a is tessellated, while the keyed
# pair STEP and uncluttered routing sheet are refreshed for review.
no_floor_obiwan_01a: validate_no_floor_obiwan_stage
	$(call RUN_STATE,0,obiwan) scripts/export_piece_stls.py \
	    --variant obiwan \
	    --obiwan-key optional_lm_keyed_1_of_2_bottom \
	    --obiwan-stage-manifest build/no_floor_stand/.obiwan_stage/manifest.json \
	    --outdir build/no_floor_stand/stl
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/export_obiwan_staged.py step \
	    --manifest build/no_floor_stand/.obiwan_stage/manifest.json \
	    --kind lm_split \
	    --output build/no_floor_stand/obiwan_lm_split.step
	cd build/no_floor_stand && LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan \
	    $(RUN) ../../scripts/gen_cable_routing.py
	@test -s build/no_floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl
	@test -s build/no_floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.print.json
	@test -s build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_1_of_2_bottom.support_blocker.stl
	@test -s build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_1_of_2_bottom.support_blocker.json
	@test -s build/no_floor_stand/obiwan_lm_split.step
	@test -s build/no_floor_stand/baffle_cable_routing_obiwan.png

# One public remote target owns the complete two-state Obi-Wan release, Ac/Ae
# magnetic attachments and the focused final-BREP interface checks.  A single
# Make DAG lets osado share both native stages and fan independent state,
# wing and validation recipes across its bounded jobserver slots.
obiwan_release: floor_stand_artifacts no_floor_stand_artifacts \
        obiwan_wings \
        $(CAPTIVE_MAGNET_CATALOG) \
        $(call clearance_check_stamp,test_stock_slim_floor_bend_lateral_envelope) \
        $(call clearance_check_stamp,test_lm_exit_curvature_and_fishing) \
        check_route_contract check_obiwan_lm_split check_obiwan_shells \
        check_obiwan_route_boundaries check_floor_integrated_mount \
        check_obiwan_um_pilot_spoke \
        check_obiwan_lm_split_two_pin_static \
        check_obiwan_junction_closures
	# Ac/Ae already pass their own final gate through obiwan_wings.
	# State-local sweeps already ran exactly once through the artifact nodes;
	# retain only the inexpensive cross-state metadata gate on this join.
	$(RUN_LIGHT) scripts/check_manifold.py --metadata-only \
	    build/floor_stand/stl build/no_floor_stand/stl

# Source-linked, dimensioned Ac/Ae attachment layout for the Obi-Wan carrier.
# Ac and Ae are the complete STEP-first wing inventory. Their
# independent stamps let osado build both slug trees concurrently; each tree is
# one atomic, hash-backed transaction containing STEP, STL, facts and review
# images.  The PNG remains the shared analytic contract and is regenerated
# before either solid model.
obiwan_wings: $(OBIWAN_WING_DESIGN_MAP) obiwan_wing_artifacts

# A successful export-only checkpoint publishes the expensive Ac/Ae
# transactions into the verified remote Make cache before live-BREP QA.  A
# test-only failure can then be fixed and rerun without rebuilding either
# fine-mesh wing family.
obiwan_wing_exports: $(OBIWAN_WING_DESIGN_MAP) $(OBIWAN_WING_STAMPS)

obiwan_wing_artifacts: obiwan_wing_exports check_obiwan_wings

build/wings/.stamp_%: scripts/export_obiwan_wings.py src/lx521_baffle/obiwan/wings.py \
        scripts/gen_obiwan_wing_design_map.py $(OBIWAN_WING_DESIGN_MAP) \
        $(OBIWAN_WING_INPUTS)
	@rm -f $@
	@mkdir -p build/wings
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/export_obiwan_wings.py --slug $* --output-root build/wings
	@test -s build/wings/$*/obiwan_wing_$*.step
	@test -s build/wings/$*/obiwan_wing_$*_assembled.step
	@test -s build/wings/$*/obiwan_wing_$*_assembled_split2.step
	@touch $@

$(OBIWAN_WING_ARTIFACT_CHECK_STAMP): tests/test_obiwan_wings.py \
        $(OBIWAN_WING_STAMPS) $(OBIWAN_SRCS) $(EXECUTION_SRCS) \
        $(CHECK_REVALIDATE)
	@mkdir -p $(@D)
	LX_OBIWAN_WING_SINGLE_CHECK=test_exported_artifact_contract \
	    $(call RUN_STATE,0,obiwan,.) \
	    tests/test_obiwan_wings.py --artifact-root build/wings
	@touch $@

$(OBIWAN_WING_AC_LIVE_CHECK_STAMP): tests/test_obiwan_wings.py \
        $(OBIWAN_WING_STAMPS) $(OBIWAN_SRCS) $(EXECUTION_SRCS) \
        $(CHECK_REVALIDATE)
	@mkdir -p $(@D)
	LX_OBIWAN_WING_SINGLE_CHECK=test_live_brep_geometry_contract_ac \
	    $(call RUN_STATE,0,obiwan,.) \
	    tests/test_obiwan_wings.py --artifact-root build/wings
	@touch $@

$(OBIWAN_WING_AE_LIVE_CHECK_STAMP): tests/test_obiwan_wings.py \
        $(OBIWAN_WING_STAMPS) $(OBIWAN_SRCS) $(EXECUTION_SRCS) \
        $(CHECK_REVALIDATE)
	@mkdir -p $(@D)
	LX_OBIWAN_WING_SINGLE_CHECK=test_live_brep_geometry_contract_ae \
	    $(call RUN_STATE,0,obiwan,.) \
	    tests/test_obiwan_wings.py --artifact-root build/wings
	@touch $@

check_obiwan_wings: $(OBIWAN_WING_CHECK_STAMPS)
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/wings/ac/stl build/wings/ae/stl'

$(OBIWAN_WING_DESIGN_MAP): scripts/gen_obiwan_wing_design_map.py \
        scripts/gen_driver_overlay.py src/lx521_baffle/proud/a_comp.py \
        src/lx521_baffle/proud/b.py src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/floor_bend.py \
        src/lx521_baffle/obiwan/bridge.py \
        src/lx521_baffle/obiwan/floor.py
	@rm -f $@
	$(call RUN_STATE_LIGHT,0,obiwan,.) \
	    scripts/gen_obiwan_wing_design_map.py --output $@
	@test -s $@

no_floor_stand_artifacts: validate_no_floor_obiwan_stage \
        $(call VARIANT_TARGETS,build/no_floor_stand)
	# Symmetric with the floor state: expose this sweep while independent
	# floor-state artifact work can continue under the same Make jobserver.
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/no_floor_stand/stl'
no_floor_stand: no_floor_stand_artifacts
common: build/common/attachments.step

# Rebuild the product-facing artifacts/ facade from whatever is already
# promoted under build/, docs/ and images/generated/.  It writes only
# relative symlinks and hash manifests and never invokes CAD.
artifacts: scripts/build_artifact_catalog.py
	$(PYTHON_RUN) scripts/build_artifact_catalog.py

# Standardized README comparison renders: one shared camera and one declared
# frame per scale group, drawn from the promoted STEP files.  Like the
# artifacts facade it deliberately does not list those STEPs as prerequisites,
# so a missing input can never launch heavyweight CAD on this workstation; the
# generator fails closed and names the target that produces it.
ISO_MATRIX_OUTDIR ?= images/generated/iso
iso_matrix: scripts/gen_product_iso_matrix.py
	$(RUN) scripts/gen_product_iso_matrix.py --outdir "$(ISO_MATRIX_OUTDIR)"

check: validate_obiwan_stages check_static check_clearance_suite check_r6f_suite \
        check_obiwan_junction_closures

check_static: check_shared_contracts check_remote_transport check_release_metadata \
        check_slice_metadata check_to_print_shelf check_captive_magnets \
        check_obiwan_lm_split_two_pin_static

check_shared_contracts: tests/test_shared_contracts.py src/lx521_baffle/io.py \
        src/lx521_baffle/magnet_contract.py src/lx521_baffle/print_contract.py src/lx521_baffle/stl_export.py \
        src/lx521_baffle/assembly.py src/lx521_baffle/geom.py scripts/run_memory_guarded.py
	$(PYTHON_RUN) tests/test_shared_contracts.py

# test_remote_cad.py reads this file and asserts on the rendered DAG, so
# the Makefile text is a genuine input to this check.
check_remote_transport: tests/test_remote_cad.py scripts/remote_cad.py \
        $(BUILD_POLICY_SRCS)
	$(PYTHON_RUN) tests/test_remote_cad.py

check_release_metadata: $(CAPTIVE_MAGNET_CATALOG) \
        tests/test_release_metadata.py scripts/check_manifold.py \
        $(PRINT_CONTRACT_SRCS) src/lx521_baffle/stl_export.py src/lx521_baffle/geom.py \
        scripts/write_obiwan_release_manifest.py $(BUILD_POLICY_SRCS)
	$(PYTHON_RUN) tests/test_release_metadata.py

check_slice_metadata: tests/test_slice_captive_magnets.py \
        $(SLICER_SRCS) scripts/json_schema_subset.py \
        captive_magnet_slicing_profile.json src/lx521_baffle/magnet_contract.py \
        $(PRINT_CONTRACT_SRCS)
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py

# The shelf gate must not race the release catalog during a remote candidate
# build.  Local Bambu entry points already require an existing catalog before
# reaching this node, while the explicit prerequisite orders the remote DAG.
TO_PRINT_CATALOG_PREREQ := $(if $(filter remote-worker,$(LX_CAD_EXECUTION)),\
        $(CAPTIVE_MAGNET_CATALOG),$(abspath $(CAPTIVE_MAGNET_CATALOG)))
check_to_print_shelf: tests/test_to_print_shelf.py scripts/build_to_print_shelf.py \
        scripts/build_obiwan_combo_plate.py tests/test_obiwan_combo_plates.py \
        scripts/build_obiwan_wing_plate.py tests/test_obiwan_wing_plates.py \
        scripts/composite_bambu_3mf_audit.py to_print/catalog.json \
        $(SLICER_SRCS) scripts/bambu_3mf_audit.py $(PRINT_CONTRACT_SRCS) \
        $(TO_PRINT_CATALOG_PREREQ)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_to_print_shelf.py

# These targets consume an existing, remotely generated catalog but never list
# it as a Make prerequisite: a missing catalog must not launch heavyweight CAD
# on this workstation.  They run Bambu Studio only as a local CLI slicer and do
# not contain any printer transport/upload/start operation.
BAMBU_JOBS ?= 1

define REQUIRE_BAMBU_CATALOG
	@test -s "$(CAPTIVE_MAGNET_CATALOG)" || { \
		echo "missing $(CAPTIVE_MAGNET_CATALOG)" >&2; \
		echo "run 'make obiwan_release' first to generate/promote it remotely, then rerun this local Bambu target" >&2; \
		exit 1; \
	}
endef

# The only pytest-style suite in the tree.  It exercises the 3MF transform,
# mesh-equivalence, result-bbox and bed-fit helpers on pure synthetic fixtures,
# so it belongs to the workstation Bambu family rather than the remote DAG:
# pytest is deliberately absent from cad-remote-requirements.lock, and adding
# it to check_static would make the osado graph depend on a package the pinned
# executor environment does not carry.
check_bambu_3mf_audit: tests/test_bambu_3mf_audit.py scripts/bambu_3mf_audit.py
	$(PYTHON_RUN) -m pytest -q tests/test_bambu_3mf_audit.py

bambu_profiles: check_slice_metadata check_bambu_3mf_audit
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--prepare-profiles-only

bambu_dry_run: check_slice_metadata check_bambu_3mf_audit
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--dry-run

bambu_slice_obiwan: check_slice_metadata check_bambu_3mf_audit
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--only '*Obi-Wan*' --emit-ready-projects

bambu_slice_release: check_slice_metadata check_bambu_3mf_audit
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--emit-ready-projects

# The optional TEBM vase family is deliberately outside the protected
# 58-artifact release inventory.  Each profile catalog still uses every
# geometry, G-code, pause, STL/3MF-equivalence, and ready-project gate; only
# the expected inventory count is scoped to its isolated one-artifact child.
define REQUIRE_VASE_TEBM_CAD
	@for path in $(VASE_TEBM_CAD_OUTPUTS); do \
		test -s "$$path" || { \
			echo "missing promoted TEBM vase CAD artifact: $$path" >&2; \
			echo "run 'make vase_tebm35c10_4_cad' first, then rerun this local Bambu target" >&2; \
			exit 1; \
		}; \
	done
	@test -e "$(VASE_TEBM_CAD_STAMP)" || { \
		echo "missing promoted TEBM vase CAD stamp: $(VASE_TEBM_CAD_STAMP)" >&2; \
		echo "run 'make vase_tebm35c10_4_cad' first, then rerun this local Bambu target" >&2; \
		exit 1; \
	}
endef

$(VASE_TEBM_STOCK_SLICE_STAMP): $(VASE_TEBM_STOCK_CAD_OUTPUTS) \
        $(VASE_TEBM_CAD_STAMP) \
        scripts/slice_captive_magnets.py scripts/release_validation.py \
        scripts/gcode_analysis.py scripts/artifact_emit.py \
        scripts/bambu_3mf_audit.py \
        scripts/validate_vase_tebm35c10_4_delivery.py \
        src/lx521_baffle/print_contract.py src/lx521_baffle/io.py
	@rm -f "$@"
	$(REQUIRE_VASE_TEBM_CAD)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--auxiliary-catalog \
		--catalog "$(VASE_TEBM_STOCK_CATALOG)" \
		--profile "$(VASE_TEBM_STOCK_PROFILE)" \
		--output "$(VASE_TEBM_STOCK_SLICE_OUTPUT)" \
		--jobs 1 \
		--emit-ready-projects
	@test -s "$(VASE_TEBM_STOCK_READY_PROJECT)"
	@test -s "$(VASE_TEBM_STOCK_GCODE)"
	@test -s "$(VASE_TEBM_STOCK_SLICE_AUDIT)"
	@rm -f "$(VASE_TEBM_STOCK_3MF)"
	@temporary="$(VASE_TEBM_STOCK_3MF).tmp"; \
	    cp "$(VASE_TEBM_STOCK_READY_PROJECT)" "$$temporary"; \
	    mv "$$temporary" "$(VASE_TEBM_STOCK_3MF)"
	$(PYTHON_RUN) scripts/validate_vase_tebm35c10_4_delivery.py \
		--profile stock \
		--catalog "$(VASE_TEBM_STOCK_CATALOG)" \
		--audit "$(VASE_TEBM_STOCK_SLICE_AUDIT)" \
		--project "$(VASE_TEBM_STOCK_3MF)" \
		--gcode "$(VASE_TEBM_STOCK_GCODE)"
	@touch "$@"

$(VASE_TEBM_SLIM_SLICE_STAMP): $(VASE_TEBM_SLIM_CAD_OUTPUTS) \
        $(VASE_TEBM_CAD_STAMP) \
        scripts/slice_captive_magnets.py scripts/release_validation.py \
        scripts/gcode_analysis.py scripts/artifact_emit.py \
        scripts/bambu_3mf_audit.py \
        scripts/validate_vase_tebm35c10_4_delivery.py \
        src/lx521_baffle/print_contract.py src/lx521_baffle/io.py
	@rm -f "$@"
	$(REQUIRE_VASE_TEBM_CAD)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--auxiliary-catalog \
		--catalog "$(VASE_TEBM_SLIM_CATALOG)" \
		--profile "$(VASE_TEBM_SLIM_PROFILE)" \
		--output "$(VASE_TEBM_SLIM_SLICE_OUTPUT)" \
		--jobs 1 \
		--emit-ready-projects
	@test -s "$(VASE_TEBM_SLIM_READY_PROJECT)"
	@test -s "$(VASE_TEBM_SLIM_GCODE)"
	@test -s "$(VASE_TEBM_SLIM_SLICE_AUDIT)"
	@rm -f "$(VASE_TEBM_SLIM_3MF)"
	@temporary="$(VASE_TEBM_SLIM_3MF).tmp"; \
	    cp "$(VASE_TEBM_SLIM_READY_PROJECT)" "$$temporary"; \
	    mv "$$temporary" "$(VASE_TEBM_SLIM_3MF)"
	$(PYTHON_RUN) scripts/validate_vase_tebm35c10_4_delivery.py \
		--profile slim \
		--catalog "$(VASE_TEBM_SLIM_CATALOG)" \
		--audit "$(VASE_TEBM_SLIM_SLICE_AUDIT)" \
		--project "$(VASE_TEBM_SLIM_3MF)" \
		--gcode "$(VASE_TEBM_SLIM_GCODE)"
	@touch "$@"

$(VASE_TEBM_STOCK_3MF) $(VASE_TEBM_STOCK_GCODE) \
        $(VASE_TEBM_STOCK_SLICE_AUDIT): | $(VASE_TEBM_STOCK_SLICE_STAMP)
	@test -s "$@" || { \
		rm -f "$(VASE_TEBM_STOCK_SLICE_STAMP)"; \
		$(MAKE) --no-print-directory "$(VASE_TEBM_STOCK_SLICE_STAMP)"; \
		test -s "$@"; \
	}

$(VASE_TEBM_SLIM_3MF) $(VASE_TEBM_SLIM_GCODE) \
        $(VASE_TEBM_SLIM_SLICE_AUDIT): | $(VASE_TEBM_SLIM_SLICE_STAMP)
	@test -s "$@" || { \
		rm -f "$(VASE_TEBM_SLIM_SLICE_STAMP)"; \
		$(MAKE) --no-print-directory "$(VASE_TEBM_SLIM_SLICE_STAMP)"; \
		test -s "$@"; \
	}

vase_tebm35c10_4_stock_3mf: \
        $(VASE_TEBM_STOCK_3MF) $(VASE_TEBM_STOCK_SLICE_AUDIT)
	@echo "Stock ready 3MF: $(VASE_TEBM_STOCK_3MF)"

vase_tebm35c10_4_slim_3mf: \
        $(VASE_TEBM_SLIM_3MF) $(VASE_TEBM_SLIM_SLICE_AUDIT)
	@echo "Slim ready 3MF: $(VASE_TEBM_SLIM_3MF)"

vase_tebm35c10_4_3mf: \
        vase_tebm35c10_4_stock_3mf vase_tebm35c10_4_slim_3mf
	@echo "Stock/Slim BMR vase projects validated"

vase_tebm35c10_4_stock_3mf_validate:
	$(REQUIRE_VASE_TEBM_CAD)
	@test -s "$(VASE_TEBM_STOCK_3MF)"
	@test -s "$(VASE_TEBM_STOCK_GCODE)"
	@test -s "$(VASE_TEBM_STOCK_SLICE_AUDIT)"
	$(PYTHON_RUN) scripts/validate_vase_tebm35c10_4_delivery.py \
		--profile stock \
		--catalog "$(VASE_TEBM_STOCK_CATALOG)" \
		--audit "$(VASE_TEBM_STOCK_SLICE_AUDIT)" \
		--project "$(VASE_TEBM_STOCK_3MF)" \
		--gcode "$(VASE_TEBM_STOCK_GCODE)"

vase_tebm35c10_4_slim_3mf_validate:
	$(REQUIRE_VASE_TEBM_CAD)
	@test -s "$(VASE_TEBM_SLIM_3MF)"
	@test -s "$(VASE_TEBM_SLIM_GCODE)"
	@test -s "$(VASE_TEBM_SLIM_SLICE_AUDIT)"
	$(PYTHON_RUN) scripts/validate_vase_tebm35c10_4_delivery.py \
		--profile slim \
		--catalog "$(VASE_TEBM_SLIM_CATALOG)" \
		--audit "$(VASE_TEBM_SLIM_SLICE_AUDIT)" \
		--project "$(VASE_TEBM_SLIM_3MF)" \
		--gcode "$(VASE_TEBM_SLIM_GCODE)"

vase_tebm35c10_4_3mf_validate: \
        vase_tebm35c10_4_stock_3mf_validate \
        vase_tebm35c10_4_slim_3mf_validate

# Local-only structural packaging for the no-floor Obi-Wan LM bottom.  The
# modifier is slicer geometry, not CAD: it changes infill only where it
# intersects the released bridge/root while preserving every duct and bore.
$(OBIWAN_01A_MODIFIER_STAMP): \
        scripts/build_obiwan_bridge_root_modifier.py \
        tests/test_obiwan_bridge_root_modifier.py \
        $(OBIWAN_01A_RELEASED_STL) $(OBIWAN_01A_RELEASED_SIDECAR)
	@rm -f "$@"
	$(PYTHON_RUN) tests/test_obiwan_bridge_root_modifier.py
	$(PYTHON_RUN) scripts/build_obiwan_bridge_root_modifier.py
	@test -s "$(OBIWAN_01A_MODIFIER_STL)"
	@test -s "$(OBIWAN_01A_MODIFIER_CONTRACT)"
	@touch "$@"

$(OBIWAN_01A_MODIFIER_STL) $(OBIWAN_01A_MODIFIER_CONTRACT): | \
        $(OBIWAN_01A_MODIFIER_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_01A_MODIFIER_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_01A_MODIFIER_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_petg_gf_modifier: $(OBIWAN_01A_MODIFIER_STL) \
        $(OBIWAN_01A_MODIFIER_CONTRACT)

$(OBIWAN_01A_PETG_GF_STAMP): \
        $(OBIWAN_01A_MODIFIER_STL) $(OBIWAN_01A_MODIFIER_CONTRACT) \
        $(PETG_GF_PROFILE) $(SLICER_SRCS) \
        tests/test_slice_captive_magnets.py \
        $(abspath $(CAPTIVE_MAGNET_CATALOG))
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01A_ARTIFACT_ID)" \
		--dry-run
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01A_ARTIFACT_ID)" \
		--emit-ready-projects
	@test -s "$(OBIWAN_01A_PETG_GF_PROJECT)"
	@test -s "$(OBIWAN_01A_PETG_GF_GCODE)"
	@test -s "$(OBIWAN_01A_PETG_GF_AUDIT)"
	@touch "$@"

$(OBIWAN_01A_PETG_GF_PROJECT) $(OBIWAN_01A_PETG_GF_GCODE) \
        $(OBIWAN_01A_PETG_GF_AUDIT): | $(OBIWAN_01A_PETG_GF_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_01A_PETG_GF_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_01A_PETG_GF_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_petg_gf_01a: $(OBIWAN_01A_PETG_GF_PROJECT) \
        $(OBIWAN_01A_PETG_GF_AUDIT)

obiwan_no_floor_petg_gf_01a_validate:
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01A_ARTIFACT_ID)" \
		--emit-ready-projects
	@test -s "$(OBIWAN_01A_PETG_GF_PROJECT)"
	@test -s "$(OBIWAN_01A_PETG_GF_AUDIT)"
	@touch "$(OBIWAN_01A_PETG_GF_STAMP)"

$(OBIWAN_01B_PETG_GF_STAMP): \
        $(OBIWAN_01B_RELEASED_STL) $(OBIWAN_01B_RELEASED_SIDECAR) \
        $(PETG_GF_PROFILE) $(SLICER_SRCS) \
        tests/test_slice_captive_magnets.py \
        $(abspath $(CAPTIVE_MAGNET_CATALOG))
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_01B_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01B_ARTIFACT_ID)" \
		--dry-run
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_01B_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01B_ARTIFACT_ID)" \
		--emit-ready-projects
	@test -s "$(OBIWAN_01B_PETG_GF_PROJECT)"
	@test -s "$(OBIWAN_01B_PETG_GF_GCODE)"
	@test -s "$(OBIWAN_01B_PETG_GF_AUDIT)"
	@touch "$@"

$(OBIWAN_01B_PETG_GF_PROJECT) $(OBIWAN_01B_PETG_GF_GCODE) \
        $(OBIWAN_01B_PETG_GF_AUDIT): | $(OBIWAN_01B_PETG_GF_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_01B_PETG_GF_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_01B_PETG_GF_STAMP)"; \
		test -s "$@"; \
	}

obiwan_floor_petg_gf_01b: $(OBIWAN_01B_PETG_GF_PROJECT) \
        $(OBIWAN_01B_PETG_GF_AUDIT)

obiwan_floor_petg_gf_01b_validate:
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--profile "$(PETG_GF_PROFILE)" \
		--output "$(PETG_GF_01B_AUDIT_OUTPUT)" \
		--jobs 1 \
		--only "$(OBIWAN_01B_ARTIFACT_ID)" \
		--emit-ready-projects
	@test -s "$(OBIWAN_01B_PETG_GF_PROJECT)"
	@test -s "$(OBIWAN_01B_PETG_GF_AUDIT)"
	@touch "$(OBIWAN_01B_PETG_GF_STAMP)"

# The two combined plates are pure local packaging over released STLs.
# Absolute prerequisites deliberately have no matching CAD rules: if a
# released input or audit authority is absent, Make fails locally instead of
# regenerating it on the workstation or dispatching it to osado.
OBIWAN_NO_FLOOR_COMBO_RELEASED_INPUTS := $(addprefix $(CURDIR)/,\
        build/no_floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl \
        build/no_floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.print.json \
        build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_1_of_2_bottom.support_blocker.stl \
        build/no_floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.stl \
        build/no_floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.print.json \
        build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.stl \
        build/no_floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl \
        build/no_floor_stand/stl/obiwan_core_2_of_2_um_carrier.print.json \
        build/no_floor_stand/support_blockers/obiwan_core_2_of_2_um_carrier.support_blocker.stl \
        build/no_floor_stand/stl/obiwan_addon_tweeter_crescent.stl \
        build/no_floor_stand/stl/obiwan_addon_tweeter_crescent.print.json)
OBIWAN_FLOOR_COMBO_RELEASED_INPUTS := $(addprefix $(CURDIR)/,\
        build/floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl \
        build/floor_stand/stl/obiwan_optional_lm_keyed_1_of_2_bottom.print.json \
        build/floor_stand/support_blockers/obiwan_optional_lm_keyed_1_of_2_bottom.support_blocker.stl \
        build/floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.stl \
        build/floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.print.json \
        build/floor_stand/support_blockers/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.stl \
        build/floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl \
        build/floor_stand/stl/obiwan_core_2_of_2_um_carrier.print.json \
        build/floor_stand/support_blockers/obiwan_core_2_of_2_um_carrier.support_blocker.stl \
        build/floor_stand/stl/obiwan_addon_tweeter_crescent.stl \
        build/floor_stand/stl/obiwan_addon_tweeter_crescent.print.json)
OBIWAN_NO_FLOOR_COMBO_DISCOVERY_AUDITS := \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/no_floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_1_of_2_bottom/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/no_floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_2_of_2_top/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/no_floor_stand_Obi-Wan_obiwan_core_2_of_2_um_carrier/captive_magnet_slice_audit.json
OBIWAN_FLOOR_COMBO_DISCOVERY_AUDITS := \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_1_of_2_bottom/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/floor_stand_Obi-Wan-split_obiwan_optional_lm_keyed_2_of_2_top/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/floor_stand_Obi-Wan_obiwan_core_2_of_2_um_carrier/captive_magnet_slice_audit.json
OBIWAN_COMBO_SOURCE_SRCS := scripts/build_obiwan_combo_plate.py \
        scripts/composite_bambu_3mf_audit.py scripts/bambu_3mf_audit.py \
        scripts/check_manifold.py tests/test_obiwan_combo_plates.py \
        $(PRINT_CONTRACT_SRCS) $(BUILD_POLICY_SRCS)
OBIWAN_COMBO_READY_SRCS := $(OBIWAN_COMBO_SOURCE_SRCS) \
        $(SLICER_SRCS) scripts/json_schema_subset.py \
        $(PETG_GF_PROFILE) tests/test_slice_captive_magnets.py

$(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP): $(OBIWAN_COMBO_SOURCE_SRCS) \
        $(OBIWAN_NO_FLOOR_COMBO_RELEASED_INPUTS)
	@rm -f "$@"
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant no_floor_stand --source-only
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_STL)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_MANIFEST)"
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py \
		--variant no_floor_stand
	@touch "$@"

$(OBIWAN_NO_FLOOR_COMBO_STL) $(OBIWAN_NO_FLOOR_COMBO_MANIFEST): | \
        $(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_combo_plate_source: $(OBIWAN_NO_FLOOR_COMBO_STL) \
        $(OBIWAN_NO_FLOOR_COMBO_MANIFEST)

$(OBIWAN_NO_FLOOR_COMBO_READY_STAMP): \
        $(OBIWAN_NO_FLOOR_COMBO_SOURCE_STAMP) \
        $(OBIWAN_COMBO_READY_SRCS) \
        $(OBIWAN_01A_MODIFIER_STL) $(OBIWAN_01A_MODIFIER_CONTRACT) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_NO_FLOOR_COMBO_DISCOVERY_AUDITS)
	@rm -f "$@"
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py \
		--variant no_floor_stand
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant no_floor_stand \
		--workspace "$(OBIWAN_NO_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--dry-run
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant no_floor_stand \
		--workspace "$(OBIWAN_NO_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_PROJECT)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_GCODE)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_RESULT)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_AUDIT)"
	@touch "$@"

$(OBIWAN_NO_FLOOR_COMBO_READY_OUTPUTS): | \
        $(OBIWAN_NO_FLOOR_COMBO_READY_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_NO_FLOOR_COMBO_READY_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_NO_FLOOR_COMBO_READY_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_combo_plate: $(OBIWAN_NO_FLOOR_COMBO_PROJECT) \
        $(OBIWAN_NO_FLOOR_COMBO_AUDIT)

obiwan_no_floor_combo_plate_validate: obiwan_no_floor_combo_plate_source \
        $(OBIWAN_COMBO_READY_SRCS) \
        $(OBIWAN_01A_MODIFIER_STL) $(OBIWAN_01A_MODIFIER_CONTRACT) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_NO_FLOOR_COMBO_DISCOVERY_AUDITS)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py \
		--variant no_floor_stand
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant no_floor_stand \
		--workspace "$(OBIWAN_NO_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_PROJECT)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_AUDIT)"
	@touch "$(OBIWAN_NO_FLOOR_COMBO_READY_STAMP)"

$(OBIWAN_FLOOR_COMBO_SOURCE_STAMP): $(OBIWAN_COMBO_SOURCE_SRCS) \
        $(OBIWAN_FLOOR_COMBO_RELEASED_INPUTS)
	@rm -f "$@"
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant floor_stand --source-only
	@test -s "$(OBIWAN_FLOOR_COMBO_STL)"
	@test -s "$(OBIWAN_FLOOR_COMBO_MANIFEST)"
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py --variant floor_stand
	@touch "$@"

$(OBIWAN_FLOOR_COMBO_STL) $(OBIWAN_FLOOR_COMBO_MANIFEST): | \
        $(OBIWAN_FLOOR_COMBO_SOURCE_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_FLOOR_COMBO_SOURCE_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_FLOOR_COMBO_SOURCE_STAMP)"; \
		test -s "$@"; \
	}

obiwan_floor_combo_plate_source: $(OBIWAN_FLOOR_COMBO_STL) \
        $(OBIWAN_FLOOR_COMBO_MANIFEST)

$(OBIWAN_FLOOR_COMBO_READY_STAMP): \
        $(OBIWAN_FLOOR_COMBO_SOURCE_STAMP) \
        $(OBIWAN_COMBO_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_FLOOR_COMBO_DISCOVERY_AUDITS)
	@rm -f "$@"
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py --variant floor_stand
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant floor_stand \
		--workspace "$(OBIWAN_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--dry-run
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant floor_stand \
		--workspace "$(OBIWAN_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing
	@test -s "$(OBIWAN_FLOOR_COMBO_PROJECT)"
	@test -s "$(OBIWAN_FLOOR_COMBO_GCODE)"
	@test -s "$(OBIWAN_FLOOR_COMBO_RESULT)"
	@test -s "$(OBIWAN_FLOOR_COMBO_AUDIT)"
	@touch "$@"

$(OBIWAN_FLOOR_COMBO_READY_OUTPUTS): | \
        $(OBIWAN_FLOOR_COMBO_READY_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_FLOOR_COMBO_READY_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_FLOOR_COMBO_READY_STAMP)"; \
		test -s "$@"; \
	}

obiwan_floor_combo_plate: $(OBIWAN_FLOOR_COMBO_PROJECT) \
        $(OBIWAN_FLOOR_COMBO_AUDIT)

obiwan_floor_combo_plate_validate: obiwan_floor_combo_plate_source \
        $(OBIWAN_COMBO_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_FLOOR_COMBO_DISCOVERY_AUDITS)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_obiwan_combo_plates.py --variant floor_stand
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/build_obiwan_combo_plate.py \
		--variant floor_stand \
		--workspace "$(OBIWAN_FLOOR_COMBO_WORKSPACE)" \
		--profile "$(PETG_GF_PROFILE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only
	@test -s "$(OBIWAN_FLOOR_COMBO_PROJECT)"
	@test -s "$(OBIWAN_FLOOR_COMBO_AUDIT)"
	@touch "$(OBIWAN_FLOOR_COMBO_READY_STAMP)"

# Backward-compatible no-floor aliases; all new documentation uses the
# state-explicit targets above.
obiwan_combo_plate_source: obiwan_no_floor_combo_plate_source
obiwan_combo_plate: obiwan_no_floor_combo_plate
obiwan_combo_plate_validate: obiwan_no_floor_combo_plate_validate

# This alternative plate is local packaging over already released Ac B-wing
# STLs.  Absolute inputs prevent either local CAD regeneration or dispatch.
OBIWAN_AC_WING_PLATE_RELEASED_INPUTS := $(addprefix $(CURDIR)/,\
        build/wings/ac/stl/obiwan_wing_ac_left_split2_1_of_2_lm_lower.stl \
        build/wings/ac/stl/obiwan_wing_ac_left_split2_1_of_2_lm_lower.print.json \
        build/wings/ac/stl/obiwan_wing_ac_left_split2_2_of_2_lm_um_upper.stl \
        build/wings/ac/stl/obiwan_wing_ac_left_split2_2_of_2_lm_um_upper.print.json \
        build/wings/ac/stl/obiwan_wing_ac_right_split2_1_of_2_lm_lower.stl \
        build/wings/ac/stl/obiwan_wing_ac_right_split2_1_of_2_lm_lower.print.json \
        build/wings/ac/stl/obiwan_wing_ac_right_split2_2_of_2_lm_um_upper.stl \
        build/wings/ac/stl/obiwan_wing_ac_right_split2_2_of_2_lm_um_upper.print.json)
OBIWAN_AC_WING_PLATE_DISCOVERY_AUDITS := \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_split2_1_of_2_lm_lower/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_split2_2_of_2_lm_um_upper/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_split2_1_of_2_lm_lower/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_split2_2_of_2_lm_um_upper/captive_magnet_slice_audit.json
OBIWAN_WING_PLATE_SOURCE_SRCS := scripts/build_obiwan_wing_plate.py \
        scripts/bambu_3mf_audit.py scripts/check_manifold.py \
        src/lx521_baffle/obiwan/wings.py $(PRINT_CONTRACT_SRCS) \
        tests/test_obiwan_wing_plates.py $(BUILD_POLICY_SRCS)
OBIWAN_WING_PLATE_READY_SRCS := $(OBIWAN_WING_PLATE_SOURCE_SRCS) \
        $(SLICER_SRCS) scripts/json_schema_subset.py \
        captive_magnet_slicing_profile.json tests/test_slice_captive_magnets.py

$(OBIWAN_AC_WING_PLATE_SOURCE_STAMP): \
        $(OBIWAN_WING_PLATE_SOURCE_SRCS) \
        $(OBIWAN_AC_WING_PLATE_RELEASED_INPUTS)
	@rm -f "$@"
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py \
		--variant ac --source-only
	@test -s "$(OBIWAN_AC_WING_PLATE_STL)"
	@test -s "$(OBIWAN_AC_WING_PLATE_MANIFEST)"
	@test -s "$(OBIWAN_AC_WING_PLATE_PREVIEW)"
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ac
	@touch "$@"

$(OBIWAN_AC_WING_PLATE_STL) $(OBIWAN_AC_WING_PLATE_MANIFEST) \
$(OBIWAN_AC_WING_PLATE_PREVIEW): | $(OBIWAN_AC_WING_PLATE_SOURCE_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AC_WING_PLATE_SOURCE_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_AC_WING_PLATE_SOURCE_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ac_wing_plate_source: $(OBIWAN_AC_WING_PLATE_STL) \
        $(OBIWAN_AC_WING_PLATE_MANIFEST) $(OBIWAN_AC_WING_PLATE_PREVIEW)

$(OBIWAN_AC_WING_PLATE_READY_STAMP): \
        $(OBIWAN_AC_WING_PLATE_SOURCE_STAMP) \
        $(OBIWAN_WING_PLATE_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_AC_WING_PLATE_DISCOVERY_AUDITS)
	@rm -f "$@"
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ac
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ac \
		--workspace "$(OBIWAN_AC_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--dry-run
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ac \
		--workspace "$(OBIWAN_AC_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing
	@test -s "$(OBIWAN_AC_WING_PLATE_PROJECT)"
	@test -s "$(OBIWAN_AC_WING_PLATE_GCODE)"
	@test -s "$(OBIWAN_AC_WING_PLATE_RESULT)"
	@test -s "$(OBIWAN_AC_WING_PLATE_AUDIT)"
	@touch "$@"

$(OBIWAN_AC_WING_PLATE_READY_OUTPUTS): | \
        $(OBIWAN_AC_WING_PLATE_READY_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AC_WING_PLATE_READY_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_AC_WING_PLATE_READY_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ac_wing_plate: $(OBIWAN_AC_WING_PLATE_PROJECT) \
        $(OBIWAN_AC_WING_PLATE_AUDIT)

obiwan_ac_wing_plate_validate: obiwan_ac_wing_plate_source \
        $(OBIWAN_WING_PLATE_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_AC_WING_PLATE_DISCOVERY_AUDITS)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ac
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ac \
		--workspace "$(OBIWAN_AC_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only
	@test -s "$(OBIWAN_AC_WING_PLATE_PROJECT)"
	@test -s "$(OBIWAN_AC_WING_PLATE_AUDIT)"
	@touch "$(OBIWAN_AC_WING_PLATE_READY_STAMP)"

# This alternative plate is also local packaging over already released Ae
# B-wing STLs.  Its locked Rz+XY placement keeps all four front-face-down
# pieces on one P2S plate; none of these rules can regenerate wing geometry.
OBIWAN_AE_WING_PLATE_RELEASED_INPUTS := $(addprefix $(CURDIR)/,\
        build/wings/ae/stl/obiwan_wing_ae_left_split2_1_of_2_lm_lower.stl \
        build/wings/ae/stl/obiwan_wing_ae_left_split2_1_of_2_lm_lower.print.json \
        build/wings/ae/stl/obiwan_wing_ae_left_split2_2_of_2_lm_um_upper.stl \
        build/wings/ae/stl/obiwan_wing_ae_left_split2_2_of_2_lm_um_upper.print.json \
        build/wings/ae/stl/obiwan_wing_ae_right_split2_1_of_2_lm_lower.stl \
        build/wings/ae/stl/obiwan_wing_ae_right_split2_1_of_2_lm_lower.print.json \
        build/wings/ae/stl/obiwan_wing_ae_right_split2_2_of_2_lm_um_upper.stl \
        build/wings/ae/stl/obiwan_wing_ae_right_split2_2_of_2_lm_um_upper.print.json)
OBIWAN_AE_WING_PLATE_DISCOVERY_AUDITS := \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_split2_1_of_2_lm_lower/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_split2_2_of_2_lm_um_upper/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_split2_1_of_2_lm_lower/captive_magnet_slice_audit.json \
        $(abspath $(BAMBU_AUDIT_OUTPUT))/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_split2_2_of_2_lm_um_upper/captive_magnet_slice_audit.json
$(OBIWAN_AE_WING_PLATE_SOURCE_STAMP): \
        $(OBIWAN_WING_PLATE_SOURCE_SRCS) \
        $(OBIWAN_AE_WING_PLATE_RELEASED_INPUTS)
	@rm -f "$@"
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py \
		--variant ae --source-only
	@test -s "$(OBIWAN_AE_WING_PLATE_STL)"
	@test -s "$(OBIWAN_AE_WING_PLATE_MANIFEST)"
	@test -s "$(OBIWAN_AE_WING_PLATE_PREVIEW)"
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ae
	@touch "$@"

$(OBIWAN_AE_WING_PLATE_STL) $(OBIWAN_AE_WING_PLATE_MANIFEST) \
$(OBIWAN_AE_WING_PLATE_PREVIEW): | $(OBIWAN_AE_WING_PLATE_SOURCE_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AE_WING_PLATE_SOURCE_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_AE_WING_PLATE_SOURCE_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ae_wing_plate_source: $(OBIWAN_AE_WING_PLATE_STL) \
        $(OBIWAN_AE_WING_PLATE_MANIFEST) $(OBIWAN_AE_WING_PLATE_PREVIEW)

$(OBIWAN_AE_WING_PLATE_READY_STAMP): \
        $(OBIWAN_AE_WING_PLATE_SOURCE_STAMP) \
        $(OBIWAN_WING_PLATE_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_AE_WING_PLATE_DISCOVERY_AUDITS)
	@rm -f "$@"
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ae
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ae \
		--workspace "$(OBIWAN_AE_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--dry-run
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ae \
		--workspace "$(OBIWAN_AE_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing
	@test -s "$(OBIWAN_AE_WING_PLATE_PROJECT)"
	@test -s "$(OBIWAN_AE_WING_PLATE_GCODE)"
	@test -s "$(OBIWAN_AE_WING_PLATE_RESULT)"
	@test -s "$(OBIWAN_AE_WING_PLATE_AUDIT)"
	@touch "$@"

$(OBIWAN_AE_WING_PLATE_READY_OUTPUTS): | \
        $(OBIWAN_AE_WING_PLATE_READY_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AE_WING_PLATE_READY_STAMP)"; \
		$(MAKE) --no-print-directory "$(OBIWAN_AE_WING_PLATE_READY_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ae_wing_plate: $(OBIWAN_AE_WING_PLATE_PROJECT) \
        $(OBIWAN_AE_WING_PLATE_AUDIT)

obiwan_ae_wing_plate_validate: obiwan_ae_wing_plate_source \
        $(OBIWAN_WING_PLATE_READY_SRCS) \
        $(abspath $(CAPTIVE_MAGNET_CATALOG)) \
        $(OBIWAN_AE_WING_PLATE_DISCOVERY_AUDITS)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) tests/test_obiwan_wing_plates.py --variant ae
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py
	$(PYTHON_RUN) scripts/build_obiwan_wing_plate.py --variant ae \
		--workspace "$(OBIWAN_AE_WING_PLATE_WORKSPACE)" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only
	@test -s "$(OBIWAN_AE_WING_PLATE_PROJECT)"
	@test -s "$(OBIWAN_AE_WING_PLATE_AUDIT)"
	@touch "$(OBIWAN_AE_WING_PLATE_READY_STAMP)"

# Promotion is also an ordinary artifact transaction.  It requires an existing
# complete shelf manifest and crosses the full 51/51 equivalence barrier with
# slicing disabled before refreshing only the requested delivery pair.
$(OBIWAN_01A_PETG_GF_PROMOTION_STAMP): \
        $(OBIWAN_01A_PETG_GF_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_01A_FRIENDLY_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_01A_PETG_GF_SHELF_STL)"
	@test -s "$(OBIWAN_01A_PETG_GF_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_01A_PETG_GF_SHELF_OUTPUTS): | \
        $(OBIWAN_01A_PETG_GF_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_01A_PETG_GF_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_01A_PETG_GF_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_petg_gf_01a_to_print: \
        $(OBIWAN_01A_PETG_GF_SHELF_OUTPUTS)

$(OBIWAN_01B_PETG_GF_PROMOTION_STAMP): \
        $(OBIWAN_01B_PETG_GF_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_01B_FRIENDLY_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_01B_PETG_GF_SHELF_STL)"
	@test -s "$(OBIWAN_01B_PETG_GF_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_01B_PETG_GF_SHELF_OUTPUTS): | \
        $(OBIWAN_01B_PETG_GF_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_01B_PETG_GF_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_01B_PETG_GF_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_floor_petg_gf_01b_to_print: \
        $(OBIWAN_01B_PETG_GF_SHELF_OUTPUTS)

$(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP): \
        $(OBIWAN_NO_FLOOR_COMBO_READY_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_NO_FLOOR_COMBO_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_SHELF_STL)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_NO_FLOOR_COMBO_SHELF_OUTPUTS): | \
        $(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_no_floor_combo_plate_to_print: \
        $(OBIWAN_NO_FLOOR_COMBO_SHELF_OUTPUTS)

$(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP): \
        $(OBIWAN_FLOOR_COMBO_READY_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_FLOOR_COMBO_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_FLOOR_COMBO_SHELF_STL)"
	@test -s "$(OBIWAN_FLOOR_COMBO_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_FLOOR_COMBO_SHELF_OUTPUTS): | \
        $(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_floor_combo_plate_to_print: $(OBIWAN_FLOOR_COMBO_SHELF_OUTPUTS)
obiwan_combo_plate_to_print: obiwan_no_floor_combo_plate_to_print

$(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP): \
        $(OBIWAN_AC_WING_PLATE_READY_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_AC_WING_PLATE_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_AC_WING_PLATE_SHELF_STL)"
	@test -s "$(OBIWAN_AC_WING_PLATE_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_AC_WING_PLATE_SHELF_OUTPUTS): | \
        $(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ac_wing_plate_to_print: $(OBIWAN_AC_WING_PLATE_SHELF_OUTPUTS)

$(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP): \
        $(OBIWAN_AE_WING_PLATE_READY_STAMP) \
        scripts/build_to_print_shelf.py tests/test_to_print_shelf.py \
        $(TO_PRINT_SHELF)/catalog.json \
        $(abspath $(TO_PRINT_SHELF))/release_manifest.json
	@rm -f "$@"
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only \
		--only "$(OBIWAN_AE_WING_PLATE_NAME)"
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_AE_WING_PLATE_SHELF_STL)"
	@test -s "$(OBIWAN_AE_WING_PLATE_SHELF_PROJECT)"
	@touch "$@"

$(OBIWAN_AE_WING_PLATE_SHELF_OUTPUTS): | \
        $(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP)
	@test -s "$@" || { \
		rm -f "$(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP)"; \
		$(MAKE) --no-print-directory \
			"$(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP)"; \
		test -s "$@"; \
	}

obiwan_ae_wing_plate_to_print: $(OBIWAN_AE_WING_PLATE_SHELF_OUTPUTS)

# User-facing, local-only materialization of the 51 P2S-printable entries.
# It consumes the existing authoritative magnet projects and fails closed if
# one is absent; the expensive full 58-artifact reslice remains an explicit
# ``make bambu_slice_release`` operation rather than an implicit shelf step.
to_print: $(OBIWAN_NO_FLOOR_COMBO_READY_STAMP) \
        $(OBIWAN_FLOOR_COMBO_READY_STAMP) \
        $(OBIWAN_AC_WING_PLATE_READY_STAMP) \
        $(OBIWAN_AE_WING_PLATE_READY_STAMP)
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing
	$(PYTHON_RUN) tests/test_to_print_shelf.py
	@test -s "$(OBIWAN_01A_PETG_GF_SHELF_STL)"
	@test -s "$(OBIWAN_01A_PETG_GF_SHELF_PROJECT)"
	@touch "$(OBIWAN_01A_PETG_GF_PROMOTION_STAMP)"
	@test -s "$(OBIWAN_01B_PETG_GF_SHELF_STL)"
	@test -s "$(OBIWAN_01B_PETG_GF_SHELF_PROJECT)"
	@touch "$(OBIWAN_01B_PETG_GF_PROMOTION_STAMP)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_SHELF_STL)"
	@test -s "$(OBIWAN_NO_FLOOR_COMBO_SHELF_PROJECT)"
	@touch "$(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP)"
	@test -s "$(OBIWAN_FLOOR_COMBO_SHELF_STL)"
	@test -s "$(OBIWAN_FLOOR_COMBO_SHELF_PROJECT)"
	@touch "$(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP)"
	@test -s "$(OBIWAN_AC_WING_PLATE_SHELF_STL)"
	@test -s "$(OBIWAN_AC_WING_PLATE_SHELF_PROJECT)"
	@touch "$(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP)"
	@test -s "$(OBIWAN_AE_WING_PLATE_SHELF_STL)"
	@test -s "$(OBIWAN_AE_WING_PLATE_SHELF_PROJECT)"
	@touch "$(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP)"

to_print_validate: obiwan_no_floor_combo_plate_source \
        obiwan_floor_combo_plate_source \
        obiwan_ac_wing_plate_source obiwan_ae_wing_plate_source \
        check_to_print_shelf
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only
	@touch "$(OBIWAN_NO_FLOOR_COMBO_READY_STAMP)"
	@touch "$(OBIWAN_NO_FLOOR_COMBO_PROMOTION_STAMP)"
	@touch "$(OBIWAN_01A_PETG_GF_STAMP)"
	@touch "$(OBIWAN_01A_PETG_GF_PROMOTION_STAMP)"
	@touch "$(OBIWAN_01B_PETG_GF_STAMP)"
	@touch "$(OBIWAN_01B_PETG_GF_PROMOTION_STAMP)"
	@touch "$(OBIWAN_FLOOR_COMBO_READY_STAMP)"
	@touch "$(OBIWAN_FLOOR_COMBO_PROMOTION_STAMP)"
	@touch "$(OBIWAN_AC_WING_PLATE_READY_STAMP)"
	@touch "$(OBIWAN_AC_WING_PLATE_PROMOTION_STAMP)"
	@touch "$(OBIWAN_AE_WING_PLATE_READY_STAMP)"
	@touch "$(OBIWAN_AE_WING_PLATE_PROMOTION_STAMP)"

check_clearance_suite: $(CLEARANCE_CHECK_STAMPS)
	@echo "all clearance checks passed (Make-jobserver isolated)"

check_r6f_suite: $(R6F_CASE_STAMPS)
	@echo "all final Obi-Wan R6F checks passed (Make-jobserver isolated)"

check_captive_magnets: tests/test_captive_magnets.py src/lx521_baffle/magnets.py \
        src/lx521_baffle/magnet_contract.py
	$(RUN) tests/test_captive_magnets.py

check_obiwan_lm_split_two_pin_static: \
        tests/test_obiwan_lm_split_two_pin_static.py \
        src/lx521_baffle/obiwan/lm_split.py
	$(PYTHON_RUN) tests/test_obiwan_lm_split_two_pin_static.py

# Seconds-scale topology gate for closure iteration.  It deliberately uses
# the same Make/remote guard authority as production but does not rebuild the
# two carrier stages until the analytic tangent/ownership plans pass.
check_obiwan_junction_closure_plans: \
		tests/test_obiwan_junction_closure_webs.py \
        src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/obiwan/closure_webs.py \
        src/lx521_baffle/obiwan/joints.py \
        src/lx521_baffle/obiwan/magnets.py \
        src/lx521_baffle/obiwan/attachments.py \
		$(OBIWAN_SRCS) $(EXECUTION_SRCS)
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_PLAN_ONLY=1 $(RUN) \
	    tests/test_obiwan_junction_closure_webs.py

# Full analytic and actual-BREP sections for the two independently assembled
# junctions.  The dense layer matrix is sixteen ordinary Make nodes: eight
# complementary layer shards for each floor/no-floor state.  Each process
# audits both LM-UM and T-UM, so their shared UM BREP and route section are
# computed once rather than once per junction.  GNU Make's inherited jobserver
# still keeps all sixteen remote slots busy.  The shard union retains every
# physical Bambu layer and assertion;
# Python does not own a second scheduler.  The explicit full-test flag makes a
# missing authenticated guard fail closed instead of silently running only the
# source contract.
check_obiwan_junction_closures: $(JUNCTION_CLOSURE_STAMP)

# Focused two-state exact-BREP gate used while iterating closure ownership.
# It stops before the sixteen dense physical-layer shards.
check_obiwan_junction_closure_base: $(JUNCTION_CLOSURE_BASE_STAMP)

$(JUNCTION_CLOSURE_STAMP): $(JUNCTION_CLOSURE_BASE_STAMP) \
		$(JUNCTION_CLOSURE_DENSE_STATE_STAMPS)
	@mkdir -p $(@D)
	@touch $@

$(JUNCTION_CLOSURE_BASE_STAMP): $(JUNCTION_CLOSURE_BASE_STATE_STAMPS)
	@mkdir -p $(@D)
	@touch $@

define JUNCTION_CLOSURE_BASE_STATE_RULE
$(call junction_closure_base_state_stamp,$(1)): \
		$(call r6f_check_stamp,$(3)) \
		tests/test_obiwan_junction_closure_webs.py \
		src/lx521_baffle/obiwan/carriers.py \
		src/lx521_baffle/obiwan/attachments.py \
		$(OBIWAN_SRCS) $(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		$(4)
	@mkdir -p $$(@D)
	# Each state is compared with its own route oracle in an independent Make
	# slot; the aggregate base stamp joins the two successful results.
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_BASE_ONLY=1 $$(RUN) \
	    tests/test_obiwan_junction_closure_webs.py
	@touch $$@
endef

$(eval $(call JUNCTION_CLOSURE_BASE_STATE_RULE,no_floor,0,feed_and_flush_mouth_contract,build/no_floor_stand/.obiwan_stage/manifest.json))
$(eval $(call JUNCTION_CLOSURE_BASE_STATE_RULE,floor,1,floor_feed_and_flush_mouth_contract,build/floor_stand/.obiwan_stage/manifest.json))

define JUNCTION_CLOSURE_DENSE_RULE
$(call junction_closure_dense_stamp,$(1),$(2)): \
		$$(call r6f_check_stamp,floor_feed_and_flush_mouth_contract) \
		$$(call r6f_check_stamp,feed_and_flush_mouth_contract) \
		tests/test_obiwan_junction_closure_webs.py \
		src/lx521_baffle/obiwan/carriers.py \
		src/lx521_baffle/obiwan/attachments.py \
		$$(OBIWAN_SRCS) $$(EXECUTION_SRCS) $$(CHECK_REVALIDATE) | \
		build/floor_stand/.obiwan_stage/manifest.json \
		build/no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $$(@D)
	# The route oracle is state-dependent; keep it in the same state as the
	# corresponding immutable staged BREP.  One state/shard process covers both
	# junctions and therefore reuses the UM section authority.
	LX_STAND_FOOT=$(if $(filter floor,$(1)),1,0) LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_DENSE_STATE=$(1) \
	    LX_OBIWAN_CLOSURE_DENSE_SHARD=$(2)/8 $$(RUN_LIGHT) \
	    tests/test_obiwan_junction_closure_webs.py
	@touch $$@
endef

$(foreach state,$(JUNCTION_CLOSURE_DENSE_STATES),\
        $(foreach shard,$(JUNCTION_CLOSURE_DENSE_SHARDS),\
        $(eval $(call JUNCTION_CLOSURE_DENSE_RULE,$(state),$(shard)))))

define JUNCTION_CLOSURE_DENSE_STATE_RULE
$(CHECK_STAMP_DIR)/obiwan_junction_closures/state_$(1).ok: \
        $(foreach shard,$(JUNCTION_CLOSURE_DENSE_SHARDS),\
        $(call junction_closure_dense_stamp,$(1),$(shard)))
	@mkdir -p $$(@D)
	@touch $$@
endef

$(foreach state,$(JUNCTION_CLOSURE_DENSE_STATES),\
        $(eval $(call JUNCTION_CLOSURE_DENSE_STATE_RULE,$(state))))

# Focused remote iteration gate.  Reuse the ordinary one-selector-per-node
# Make DAG; do not re-enter the Python suite's direct-CLI thread scheduler.
check_obiwan: validate_obiwan_stages check_r6f_suite

# Focused final-BREP shell gate.  The four independent state/route checks are
# separate prerequisites so the large remote host can run them concurrently.
check_obiwan_shells: check_floor_um_shell check_floor_t_shell \
                   check_no_floor_um_shell check_no_floor_t_shell

check_obiwan_t_shells: check_floor_t_shell check_no_floor_t_shell

# Focused two-state UM/tweeter joint and terminal-service gate.  Keep this as
# a normal Make join over the existing isolated selector nodes so osado can
# validate both stand states concurrently without re-entering a Python-owned
# scheduler or rebuilding unrelated release artifacts.
check_obiwan_service: _check_r6f_tweeter_and_service \
                    _check_r6f_floor_tweeter_and_service

# One public, remotely whitelisted regression join for closure iteration.
# These are the exact selector/stage nodes affected by junction ownership,
# route handoff, service-ear, and optional split changes; GNU Make retains
# sole staleness and jobserver authority across the otherwise independent
# workers.
check_obiwan_closure_focus: check_obiwan_shells check_obiwan_service \
                          check_obiwan_lm_split \
                          check_obiwan_junction_closures

# Focused native-mouth/burial gates.  These retain the same isolated guarded
# workers as the release suite while allowing the large remote host to check
# both stand states concurrently during route-boundary iteration.
check_obiwan_mouths: check_floor_obiwan_mouths check_no_floor_obiwan_mouths

check_obiwan_burial: check_floor_obiwan_burial check_no_floor_obiwan_burial \
                    check_obiwan_um_burial

check_obiwan_um_burial: check_floor_obiwan_um_burial \
                       check_no_floor_obiwan_um_burial

check_obiwan_backfills: check_floor_obiwan_backfills \
                      check_no_floor_obiwan_backfills

check_obiwan_route_boundaries: check_obiwan_mouths check_obiwan_burial \
                             check_obiwan_backfills

# Exact one-state iteration gate for the no-floor lower carrier.  The generic
# R6F stamp graph intentionally validates both native stages for a release,
# but a local 01a geometry correction must not rebuild the unaffected floor
# carrier merely to inspect its own final BREP.
check_no_floor_obiwan_mouths_focused: validate_no_floor_obiwan_stage \
		tests/test_obiwan_r6f.py tests/test_harness.py $(OBIWAN_SRCS)
	$(R6F_CLEAN_ENV) LX_R6F_CASE_ID=feed_and_flush_mouth_contract \
	    $(RUN) tests/test_obiwan_r6f.py

# Public focused targets are aliases of the ordinary content-addressed R6F
# nodes. This keeps the per-check stamp and its authenticated native-stage
# prerequisites single-owned instead of duplicating selector recipes.
define FOCUSED_R6F
$(1): _check_r6f_$(2)
endef

$(eval $(call FOCUSED_R6F,check_floor_obiwan_mouths,floor_feed_and_flush_mouth_contract))
$(eval $(call FOCUSED_R6F,check_no_floor_obiwan_mouths,feed_and_flush_mouth_contract))
$(eval $(call FOCUSED_R6F,check_floor_obiwan_burial,floor_lm_burial_web_contract))
$(eval $(call FOCUSED_R6F,check_no_floor_obiwan_burial,lm_burial_web_contract))
$(eval $(call FOCUSED_R6F,check_floor_obiwan_um_burial,floor_um_burial_web_contract))
$(eval $(call FOCUSED_R6F,check_no_floor_obiwan_um_burial,um_burial_web_contract))
$(eval $(call FOCUSED_R6F,check_floor_obiwan_backfills,floor_bump_backfill_contract))
$(eval $(call FOCUSED_R6F,check_no_floor_obiwan_backfills,bump_backfill_contract))
$(eval $(call FOCUSED_R6F,check_floor_um_shell,floor_um_shell))
$(eval $(call FOCUSED_R6F,check_floor_t_shell,floor_t_shell))
$(eval $(call FOCUSED_R6F,check_no_floor_um_shell,no_floor_um_shell))
$(eval $(call FOCUSED_R6F,check_no_floor_t_shell,no_floor_t_shell))

# Focused exact-BREP integral floor-load-path diagnostic. This target is
# intentionally single-check so mount/root regressions can be measured
# without rebuilding or rerunning the full Obi-Wan service matrix.
$(eval $(call FOCUSED_R6F,check_floor_integrated_mount,floor_integrated_mount))

# Focused source-BREP plus strict-STL gate for the monolithic no-floor LM.
# It catches feature-coincidence slivers before a complete candidate export.
check_no_floor_lm_mesh: _check_r6f_no_floor_lm_core \
        build/no_floor_stand/stl/.stamp_obiwan

# Focused two-state gate for the optional no-extra-fastener LM subdivision.
# State-specific prerequisites keep both exact split BREP/STL checks eligible
# for separate osado jobserver slots while sharing their release-stage nodes.
check_obiwan_lm_split: check_floor_obiwan_lm_split \
                     check_no_floor_obiwan_lm_split

check_floor_obiwan_lm_split: validate_floor_obiwan_stage \
        build/floor_stand/stl/.stamp_obiwan \
        build/floor_stand/obiwan_lm_split.step \
        _check_r6f_floor_lm_keyed_split

check_no_floor_obiwan_lm_split: validate_no_floor_obiwan_stage \
        build/no_floor_stand/stl/.stamp_obiwan \
        build/no_floor_stand/obiwan_lm_split.step \
        _check_r6f_no_floor_lm_keyed_split

# Fast source-plan plus native-BREP regression for the universal LM-lower
# wing-contact profile.  Keep the two guarded workers independent so osado can
# validate the analytic union and its blind-bored no-floor realization in
# parallel before a complete carrier/wing export.
check_obiwan_lm_profile: check_obiwan_lm_profile_plan \
                       check_obiwan_lm_profile_brep

$(eval $(call FOCUSED_R6F,check_obiwan_lm_profile_plan,bridge_contract))
$(eval $(call FOCUSED_R6F,check_obiwan_lm_profile_brep,bridge_geometry))

# Lightweight focused gate for deterministic route dimensions and policy.
# Keep it remotely guarded because importing the final CAD modules initializes
# OCC even though this check itself performs no heavy Boolean construction.
$(eval $(call FOCUSED_R6F,check_route_contract,route_contract))

# The 238-degree UM driver mount is intentionally offset by 5 degrees so its
# required support does not fuse into the adjacent LM--UM M3 heat-set ear.
# Keep that small visual/structural correction inside every focused release.
$(eval $(call FOCUSED_R6F,check_obiwan_um_pilot_spoke,um_driver_spoke_is_separate_from_lm_um_insert_ear))

# Focused exact-BREP route-to-insert clearance check for both stand states.
check_bump_brep: \
        _check_r6f_no_floor_route_smoothness \
        _check_r6f_bump_backfill_contract \
        _check_r6f_bump_brep_clearance \
        _check_r6f_floor_bump_brep_clearance

build/common/attachments.step: $(GEOM_SRCS) scripts/export_steps.py
	@mkdir -p $(@D)
	LX_ROUTING_PROFILE=proud $(RUN) scripts/export_steps.py \
	    lx521_baffle.proud.attachments \
	    --output $@

# B2 and V1 share one export skeleton; V1L and Obi-Wan stay hand-written
# because their local/remote staging behavior is genuinely different.
STL_VARIANTS := b2 v1
stl_variant_stamp = $(if $(filter b2,$(1)),.stamp,.stamp_$(1))
stl_variant_arg = $(if $(filter b2,$(1)),, --variant $(1))
stl_variant_step = $(1)_split.step

# $(1) = output root, $(2) = LX_STAND_FOOT, $(3) = variant.
define STATE_STL_VARIANT_RULE
$(1)/stl/$$(call stl_variant_stamp,$(3)): \
        $(1)/$$(call stl_variant_step,$(3)) $$(STL_EXPORT_SRCS)
	@rm -f $$@
	$$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py$$(call stl_variant_arg,$(3)) \
	    --source-step $(1)/$$(call stl_variant_step,$(3)) \
	    --outdir $(1)/stl
	@touch $$@
endef

$(foreach variant,$(STL_VARIANTS),\
    $(eval $(call STATE_STL_VARIANT_RULE,build/floor_stand,1,$(variant))))
$(foreach variant,$(STL_VARIANTS),\
    $(eval $(call STATE_STL_VARIANT_RULE,build/no_floor_stand,0,$(variant))))

# Recover a missing member of a multi-output stamp transaction without using
# grouped targets, which are unavailable in GNU Make 3.81.
# $(1) = output target list, $(2) = producer stamp.
define recover_stamp
$(1): $(2)
	@test -f $$@ || { rm -f $(2); make --no-print-directory $(2); test -f $$@; }
endef

# $(1) = variant dir, $(2) = LX_STAND_FOOT value
define RULES
$(1)/stl/.stamp_v1l: $(1)/v1l_split.step \
        $(STL_EXPORT_SRCS)
	@rm -f $$@
	# STEP is the geometry authority on every host.  Import its four labeled
	# physical children once, then mesh them and the lightweight grommet pair.
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l \
	    --source-step $(1)/v1l_split.step \
	    --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_obiwan: $(OBIWAN_SRCS) $(STL_EXPORT_SRCS) \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	# Mesh each hash-verified staged group in a fresh OCC process on every
	# host. This makes tessellation independent of previously meshed parts and
	# keeps the keyed registration bores deterministic after bed rotation.
	$(call RUN_STATE,$(2),obiwan) scripts/export_piece_stls.py --variant obiwan --obiwan-part lm --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	$(call RUN_STATE,$(2),obiwan) scripts/export_piece_stls.py --variant obiwan --obiwan-part lm_split --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	$(call RUN_STATE,$(2),obiwan) scripts/export_piece_stls.py --variant obiwan --obiwan-part um --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	$(call RUN_STATE,$(2),obiwan) scripts/export_piece_stls.py --variant obiwan --obiwan-part tweeter --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_coupon: scripts/export_coupon.py scripts/check_manifold.py \
		$(PRINT_CONTRACT_SRCS) \
		scripts/write_obiwan_release_manifest.py $(OBIWAN_SRCS) \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	$(call RUN_STATE,$(2),proud) scripts/export_coupon.py --outdir $(1)/stl
	@touch $$@

# R6F STEP targets import only the state-local, hash-verified native BREPs.
# Direct generator-module export is fail-closed in scripts/export_steps.py.
$(1)/obiwan_%.step: \
		$(1)/.obiwan_stage/manifest.json scripts/export_obiwan_staged.py \
		scripts/export_steps.py $(OBIWAN_SRCS)
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),obiwan,.) \
	    scripts/export_obiwan_staged.py step \
	    --manifest $(1)/.obiwan_stage/manifest.json \
	    --kind $$* --output $$@

$(1)/um_fit.step: src/lx521_baffle/um_fit.py \
        $(GEOM_SRCS) scripts/export_steps.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_steps.py \
	    lx521_baffle.um_fit --output $$@

# Osado builds V1L's shared baffle/duct tree once and emits the same four
# physical children. Explicit local mode retains one-piece-per-child staging.
$(1)/v1l_split.step: scripts/export_v1l_staged_step.py \
        scripts/export_steps.py $(GEOM_SRCS)
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_v1l_staged_step.py --output $$@

$(1)/%.step: src/lx521_baffle/proud/%.py $(GEOM_SRCS) scripts/export_steps.py
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_steps.py \
	    lx521_baffle.proud.$$* --output $$@

$(1)/baffle_cable_routing_proud.png: $(GEOM_SRCS) scripts/gen_cable_routing.py \
        scripts/gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE_LIGHT,$(2),proud) \
	    ../../scripts/gen_cable_routing.py

$(1)/baffle_cable_routing_obiwan.png: $(OBIWAN_SRCS) scripts/gen_cable_routing.py \
        scripts/gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE_LIGHT,$(2),obiwan) \
	    ../../scripts/gen_cable_routing.py

$(1)/obiwan_release_manifest.json: $(OBIWAN_RELEASE_SRCS) \
        $(1)/stl/.stamp_obiwan $(1)/stl/.stamp_coupon \
        $(1)/obiwan_split.step \
        $(1)/obiwan_lm_split.step \
        $(1)/obiwan_attachments.step \
        $(1)/obiwan_assembled.step \
        $(1)/um_fit.step \
        $(1)/baffle_cable_routing_proud.png \
        $(1)/baffle_cable_routing_obiwan.png \
        $(if $(filter 1,$(2)),$(FLOOR_STRENGTH_REPORTS)) \
        $(addprefix $(1)/,$(OVERLAY_PNGS))
	@rm -f $$@
	$(call RUN_STATE_LIGHT,$(2),obiwan) \
	    scripts/write_obiwan_release_manifest.py --state-dir $(1)

$(1)/.stamp_overlays: $(GEOM_SRCS) scripts/gen_driver_overlay.py \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE_LIGHT,$(2),proud) \
	    ../../scripts/gen_driver_overlay.py
	@test -f $(1)/baffle_variants_drivers.png
	@test -f $(1)/baffle_b1_drivers.png
	@test -f $(1)/baffle_b2_drivers.png
	@touch $$@
endef

$(eval $(call RULES,build/floor_stand,1))
$(eval $(call RULES,build/no_floor_stand,0))
$(eval $(call recover_stamp,$(addprefix build/floor_stand/,$(OVERLAY_PNGS)),build/floor_stand/.stamp_overlays))
$(eval $(call recover_stamp,$(addprefix build/no_floor_stand/,$(OVERLAY_PNGS)),build/no_floor_stand/.stamp_overlays))

build/floor_stand/.stamp_obiwan_integrated_floor_strength: \
        src/lx521_baffle/obiwan/floor_strength.py \
        build/floor_stand/obiwan_split.step \
        build/floor_stand/obiwan_lm_split.step
	@rm -f $@ $(FLOOR_STRENGTH_REPORTS)
	$(RUN) -m lx521_baffle.obiwan.floor_strength \
	    --json build/floor_stand/obiwan_integrated_floor_strength.json \
	    --markdown build/floor_stand/obiwan_integrated_floor_strength.md \
	    --step build/floor_stand/obiwan_split.step \
	    --step build/floor_stand/obiwan_lm_split.step
	@touch $@

$(eval $(call recover_stamp,$(FLOOR_STRENGTH_REPORTS),build/floor_stand/.stamp_obiwan_integrated_floor_strength))

clean:
	rm -rf build/floor_stand build/no_floor_stand build/wings \
	    build/vase_TEBM35C10-4 \
	    .manifold-stamps .check-stamps __pycache__ \
	    build/common/attachments.step \
	    build/common/obiwan_wing_design_map.png

endif
