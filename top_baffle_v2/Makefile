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
#   make bambu_profiles      local Bambu preset resolution; no CAD/printer I/O
#   make bambu_dry_run       local per-STL command/provenance preview
#   make bambu_slice_obiwan  local filtered ready-project diagnostic
#   make bambu_slice_release local authoritative ready-project slice/audit
#   make to_print            local complete P2S shelf: audit + 39 friendly files
#   make to_print_validate   validate an existing P2S shelf without slicing
#
# Compatible with the macOS-provided GNU make 3.81.

# Package geometry and importable CLIs have one canonical absolute search
# path.  It remains valid inside recipes that render from a build subdirectory.
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/scripts

# Bambu slicing is workstation-only.  Detect these public goals before the
# remote-first dispatcher captures MAKECMDGOALS, force even an explicitly
# supplied remote/remote-worker mode back to local, and reject ambiguous mixed
# CAD+slicer invocations.  Multiple Bambu goals in one invocation are allowed.
LOCAL_BAMBU_GOALS := bambu_profiles bambu_dry_run \
                     bambu_slice_obiwan bambu_slice_release \
                     check_to_print_shelf to_print to_print_validate
BAMBU_GOALS_REQUESTED := $(filter $(LOCAL_BAMBU_GOALS),$(MAKECMDGOALS))
ifneq ($(strip $(BAMBU_GOALS_REQUESTED)),)
ifneq ($(strip $(filter-out $(LOCAL_BAMBU_GOALS),$(MAKECMDGOALS))),)
$(error local-only Bambu goals cannot be mixed with CAD or other Make goals)
endif
ifneq ($(shell uname -s),Darwin)
$(error local-only Bambu goals require a Darwin/macOS host and cannot run on osado)
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
# preventing a carrier/web/split-only edit from invalidating every B/C/V0/V1/
# V1L STEP and STL.  GNU Make remains the dependency authority.
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
        src/lx521_baffle/obiwan/split.py
SHARED_CAD_SRCS := src/lx521_baffle/assembly.py src/lx521_baffle/magnet_contract.py \
        src/lx521_baffle/geom.py src/lx521_baffle/io.py
SRCS := $(filter-out $(OBIWAN_EXCLUSIVE_CAD_SRCS),$(CAD_SRCS_ALL)) \
        src/lx521_baffle/magnets.py $(SHARED_CAD_SRCS) \
        Makefile $(MU10_REFS) $(W22_REFS)
OBIWAN_SRCS := $(SRCS) $(OBIWAN_EXCLUSIVE_CAD_SRCS)
EXECUTION_SRCS := scripts/run_memory_guarded.py
PRINT_CONTRACT_SRCS := src/lx521_baffle/print_contract.py src/lx521_baffle/io.py
SLICER_SRCS := scripts/slice_captive_magnets.py \
        scripts/release_validation.py scripts/gcode_analysis.py \
        scripts/artifact_emit.py
OBIWAN_RELEASE_SRCS := $(OBIWAN_SRCS) $(EXECUTION_SRCS) $(PRINT_CONTRACT_SRCS) \
        scripts/export_steps.py scripts/export_obiwan_staged.py \
        scripts/export_piece_stls.py scripts/export_coupon.py src/lx521_baffle/stl_export.py scripts/gen_cable_routing.py \
        scripts/gen_driver_overlay.py scripts/check_manifold.py tests/test_captive_magnets.py \
        tests/test_clearances.py tests/test_shared_contracts.py \
        tests/test_obiwan_r6f.py tests/test_harness.py \
        tests/test_obiwan_junction_closure_webs.py \
        tests/test_obiwan_lm_split_two_pin_static.py tests/test_remote_cad.py \
        scripts/write_obiwan_release_manifest.py docs/obiwan_physical_qualification.md \
        scripts/remote_cad.py cad-remote-requirements.lock
# export_piece_stls imports the strict checker after meshing, and that checker
# imports the release-manifest helpers at module load.  Keep every STL stamp
# source-complete so a checker/contract change cannot reuse a stale mesh stamp.
STL_EXPORT_SRCS := scripts/export_piece_stls.py scripts/check_manifold.py \
        src/lx521_baffle/stl_export.py $(PRINT_CONTRACT_SRCS) \
        scripts/write_obiwan_release_manifest.py
# Command-line Make variables must not turn a candidate/release recipe into
# an unguarded OCC process. PYTHON remains selectable, but every invocation
# is forced through this workspace's authenticated process-tree guard.
override GUARD := $(abspath scripts/run_memory_guarded.py)
override RUN := $(PYTHON_RUN) $(GUARD) -- $(PYTHON_RUN)
RUN_STATE = LX_STAND_FOOT=$(1) LX_ROUTING_PROFILE=$(2) $(RUN)

# Local builds remain strictly serial. The remote executor may opt into
# several guarded slots only under the large-host profile; the enclosing
# systemd cgroup still limits their aggregate memory to 512 GiB.
ifeq ($(LX_CAD_EXECUTION),local)
.NOTPARALLEL:
endif

GENS := top_baffle_nd25fw4_b2_split top_baffle_nd25fw4_b2 \
        top_baffle_nd25fw4_b1 top_baffle_nd25fw4_a_comp \
        top_baffle_nd25fw4_b1_assembled top_baffle_nd25fw4_a_comp_assembled \
        top_baffle_nd25fw4_c7 top_baffle_nd25fw4_c7_split \
        top_baffle_nd25fw4_v0 top_baffle_nd25fw4_v0_split \
        top_baffle_nd25fw4_v1 top_baffle_nd25fw4_v1_split \
        top_baffle_nd25fw4_v1l_split top_baffle_nd25fw4_obiwan_split \
        top_baffle_nd25fw4_obiwan_lm_split \
        top_baffle_nd25fw4_obiwan_attachments \
        top_baffle_nd25fw4_obiwan_assembled \
        top_baffle_nd25fw4_um_fit \
        top_baffle_nd25fw4_v1_attachments

OVERLAY_PNGS = baffle_b1_drivers.png baffle_b2_drivers.png \
               baffle_variants_drivers.png
FLOOR_STRENGTH_REPORTS = build/floor_stand/obiwan_integrated_floor_strength.json \
                         build/floor_stand/obiwan_integrated_floor_strength.md
OBIWAN_WING_DESIGN_MAP = build/common/obiwan_wing_design_map.png
OBIWAN_WING_SLUGS = ac ae
OBIWAN_WING_STAMPS = $(addprefix build/wings/.stamp_,$(OBIWAN_WING_SLUGS))
CAPTIVE_MAGNET_CATALOG = review/captive_magnet_release_catalog.json
OBIWAN_WING_INPUTS = cad-remote-requirements.lock \
        docs/obiwan_acoustic_wings_spec.md src/lx521_baffle/magnets.py \
        src/lx521_baffle/assembly.py src/lx521_baffle/magnet_contract.py src/lx521_baffle/geom.py \
        src/lx521_baffle/print_contract.py src/lx521_baffle/io.py src/lx521_baffle/stl_export.py \
        scripts/export_steps.py scripts/gen_driver_overlay.py scripts/export_obiwan_staged.py \
        src/lx521_baffle/base.py src/lx521_baffle/proud/top_baffle_nd25fw4_b.py \
        src/lx521_baffle/proud/top_baffle_nd25fw4_b1.py src/lx521_baffle/proud/top_baffle_nd25fw4_b2.py \
        src/lx521_baffle/cables.py src/lx521_baffle/flush.py \
        src/lx521_baffle/proud/top_baffle_nd25fw4_v1.py src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/obiwan/attachments.py \
        src/lx521_baffle/obiwan/route.py \
        src/lx521_baffle/obiwan/bumps.py \
        src/lx521_baffle/obiwan/rear_entry.py \
        src/lx521_baffle/proud/top_baffle_nd25fw4_a_comp.py \
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
        test_variant_outlines_splice \
        test_v1l_service_envelope \
        test_emboss_driver_keepouts \
        test_attachment_flushness \
        test_magnet_top_site_walls \
        test_magnet_cavities_vs_t_ducts \
        test_ts_captive_keepout_nudge \
        test_standard_captive_magnet_contract \
        test_standard_local_magnet_backing \
        test_individual_attachment_magnet_burial \
        test_duct_vs_w22_pilots \
        test_duct_duct_separation \
        test_duct_vs_um_pilots \
        test_c7_duct_corridor \
        test_seam_keys_vs_ducts \
        test_v0_duct_corridor \
        test_v0_captive_geometry \
        test_v1_field \
        test_route_smoothness \
        test_lm_exit_curvature_and_fishing \
        test_v1l_um_terminal_axis_handoff \
        test_um_eroded_outline_containment \
        test_ts_eroded_outline_containment \
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
JUNCTION_CLOSURE_BASE_STAMP := \
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/base.ok
JUNCTION_CLOSURE_DENSE_CASES := \
        no_floor_lm_um floor_lm_um no_floor_t_um floor_t_um
JUNCTION_CLOSURE_DENSE_SHARDS := 0 1 2 3
junction_closure_dense_stamp = \
        $(CHECK_STAMP_DIR)/obiwan_junction_closures/$(1)_$(2).ok
JUNCTION_CLOSURE_DENSE_STAMPS := \
        $(foreach name,$(JUNCTION_CLOSURE_DENSE_CASES),\
        $(foreach shard,$(JUNCTION_CLOSURE_DENSE_SHARDS),\
        $(call junction_closure_dense_stamp,$(name),$(shard))))

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

define MANIFOLD_STL_RULE
$(call manifold_stamp,$(1)): $(1) scripts/check_manifold.py \
        $(PRINT_CONTRACT_SRCS) scripts/write_obiwan_release_manifest.py Makefile
	@mkdir -p $$(@D)
	$(RUN) scripts/check_manifold.py --stl-only $(1)
	@touch $$@
endef

$(foreach stl,$(MANIFOLD_STLS),$(eval $(call MANIFOLD_STL_RULE,$(stl))))

define CLEARANCE_CHECK_RULE
_check_clearance_$(1): $(call clearance_check_stamp,$(1))
$(call clearance_check_stamp,$(1)): tests/test_clearances.py $(OBIWAN_SRCS) \
        $(EXECUTION_SRCS) $(CHECK_REVALIDATE)
	@mkdir -p $$(@D)
	LX_CLEARANCE_SINGLE_CHECK=$(1) $(RUN) tests/test_clearances.py
	@touch $$@
endef

define R6F_CASE_RULE
_check_r6f_$(1): $(call r6f_check_stamp,$(1))
$(call r6f_check_stamp,$(1)): tests/test_obiwan_r6f.py tests/test_harness.py $(OBIWAN_SRCS) \
		$(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		build/floor_stand/.obiwan_stage/manifest.json \
		build/no_floor_stand/.obiwan_stage/manifest.json
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
		build/floor_stand/.obiwan_stage/manifest.json \
		build/no_floor_stand/.obiwan_stage/manifest.json
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

VARIANT_TARGETS = $(1)/stl/.stamp_coupon $(1)/stl/.stamp $(1)/stl/.stamp_c7 $(1)/stl/.stamp_v0 $(1)/stl/.stamp_v1 $(1)/stl/.stamp_v1l $(1)/stl/.stamp_obiwan \
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
        no_floor_obiwan_01a obiwan_release \
        obiwan_wings obiwan_wing_artifacts check_obiwan_wings \
        common check check_static check_shared_contracts check_remote_transport \
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
        bambu_profiles bambu_dry_run \
        bambu_slice_obiwan bambu_slice_release \
        refresh_captive_magnet_catalog_existing \
        check_to_print_shelf to_print to_print_validate \
        $(CLEARANCE_CHECK_TARGETS) $(R6F_CASE_TARGETS)
all: validate_obiwan_stages floor_stand_artifacts no_floor_stand_artifacts common \
        obiwan_wings $(CAPTIVE_MAGNET_CATALOG)
	# Each state already completed its one-node-per-STL sweep while the other
	# state was still eligible to use the shared jobserver.  Keep only the
	# inexpensive cross-state manifest/review consistency gate on the join.
	$(RUN) scripts/check_manifold.py --metadata-only \
	    build/floor_stand/stl build/no_floor_stand/stl

# This catalog is generated only after all 56 released magnet-bearing meshes
# and their exact front-down print transforms exist.  It is the CAD-side
# authority consumed by the local, no-printer-contact Bambu slicing audit.
$(CAPTIVE_MAGNET_CATALOG): scripts/generate_captive_magnet_catalog.py \
        captive_magnet_release_catalog.schema.json $(SLICER_SRCS) \
        scripts/json_schema_subset.py src/lx521_baffle/magnets.py $(SHARED_CAD_SRCS) \
        src/lx521_baffle/stl_export.py \
        build/floor_stand/stl/.stamp build/floor_stand/stl/.stamp_c7 \
        build/floor_stand/stl/.stamp_v0 build/floor_stand/stl/.stamp_v1 \
        build/floor_stand/stl/.stamp_v1l build/floor_stand/stl/.stamp_obiwan \
        build/floor_stand/stl/.stamp_coupon \
        build/no_floor_stand/stl/.stamp build/no_floor_stand/stl/.stamp_c7 \
        build/no_floor_stand/stl/.stamp_v0 build/no_floor_stand/stl/.stamp_v1 \
        build/no_floor_stand/stl/.stamp_v1l build/no_floor_stand/stl/.stamp_obiwan \
        build/no_floor_stand/stl/.stamp_coupon $(OBIWAN_WING_STAMPS)
	@mkdir -p review
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/generate_captive_magnet_catalog.py --output $@
	@test -s $@

# Iteration-only catalog refresh over the already exported, remotely verified
# release inventory.  This deliberately has no mesh prerequisites: it updates
# hash/provenance bindings after a focused CAD export without rebuilding the
# unaffected floor, wing, and legacy artifacts.  The generator still applies
# the complete 56-artifact/102-station fail-closed catalog contract.
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
	$(RUN) scripts/check_manifold.py --metadata-only --require-release-authorized

manifold: scripts/check_manifold.py
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/floor_stand/stl build/no_floor_stand/stl'

_manifold_parallel: $(MANIFOLD_STAMPS)
	@test -n "$(strip $(MANIFOLD_STLS))" || { \
	    echo "no STLs found under: $(MANIFOLD_ROOTS)" >&2; exit 1; }
	$(RUN) scripts/check_manifold.py --metadata-only \
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
# without rebuilding unrelated proud/C/V0/V1/V1L print sets.
# $(1) = public state prefix, $(2) = output root.
define FOCUSED_OBIWAN_RELEASE_RULE
$(1)_obiwan: validate_$(1)_obiwan_stage \
		$(2)/obiwan_release_manifest.json
	$$(MAKE) --no-print-directory _manifold_parallel MANIFOLD_ROOTS='$(2)/stl' MANIFOLD_OBIWAN_ONLY=1
endef

$(eval $(call FOCUSED_OBIWAN_RELEASE_RULE,floor,build/floor_stand))
$(eval $(call FOCUSED_OBIWAN_RELEASE_RULE,no_floor,build/no_floor_stand))

# Focused iteration/export contract for the only no-floor part whose cable
# routing owns the D20 entry cluster and LM-recess crossing.  The native stage
# remains the STEP-first authority; only 01a is tessellated, while the keyed
# pair STEP and uncluttered routing sheet are refreshed for review.
no_floor_obiwan_01a: validate_no_floor_obiwan_stage
	$(call RUN_STATE,0,obiwan) scripts/export_piece_stls.py \
	    --variant obiwan \
	    --obiwan-key optional_lm_keyed_1of2_bottom \
	    --obiwan-stage-manifest build/no_floor_stand/.obiwan_stage/manifest.json \
	    --outdir build/no_floor_stand/stl
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/export_obiwan_staged.py step \
	    --manifest build/no_floor_stand/.obiwan_stage/manifest.json \
	    --kind lm_split \
	    --output build/no_floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	cd build/no_floor_stand && LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan \
	    $(RUN) ../../scripts/gen_cable_routing.py
	@test -s build/no_floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl
	@test -s build/no_floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.print.json
	@test -s build/no_floor_stand/support_blockers/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.support_blocker.stl
	@test -s build/no_floor_stand/support_blockers/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.support_blocker.json
	@test -s build/no_floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	@test -s build/no_floor_stand/baffle_cable_routing_obiwan.png

# One public remote target owns the complete two-state Obi-Wan release, Ac/Ae
# magnetic attachments and the focused final-BREP interface checks.  A single
# Make DAG lets osado share both native stages and fan independent state,
# wing and validation recipes across its bounded jobserver slots.
obiwan_release: floor_stand_artifacts no_floor_stand_artifacts \
        obiwan_wings \
        $(CAPTIVE_MAGNET_CATALOG) \
        $(call clearance_check_stamp,test_lm_exit_curvature_and_fishing) \
        check_route_contract check_obiwan_lm_split check_obiwan_shells \
        check_obiwan_route_boundaries check_floor_integrated_mount \
        check_obiwan_um_pilot_spoke \
        check_obiwan_lm_split_two_pin_static \
        check_obiwan_junction_closures
	# Ac/Ae already pass their own final gate through obiwan_wings.
	# State-local sweeps already ran exactly once through the artifact nodes;
	# retain only the inexpensive cross-state metadata gate on this join.
	$(RUN) scripts/check_manifold.py --metadata-only \
	    build/floor_stand/stl build/no_floor_stand/stl

# Source-linked, dimensioned Ac/Ae attachment layout for the Obi-Wan carrier.
# Ac and Ae are the complete STEP-first wing inventory. Their
# independent stamps let osado build both slug trees concurrently; each tree is
# one atomic, hash-backed transaction containing STEP, STL, facts and review
# images.  The PNG remains the shared analytic contract and is regenerated
# before either solid model.
obiwan_wings: $(OBIWAN_WING_DESIGN_MAP) obiwan_wing_artifacts

obiwan_wing_artifacts: $(OBIWAN_WING_STAMPS) check_obiwan_wings

build/wings/.stamp_%: scripts/export_obiwan_wings.py src/lx521_baffle/obiwan/wings.py \
        scripts/gen_obiwan_wing_design_map.py $(OBIWAN_WING_DESIGN_MAP) \
        scripts/check_manifold.py $(OBIWAN_WING_INPUTS)
	@rm -f $@
	@mkdir -p build/wings
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/export_obiwan_wings.py --slug $* --output-root build/wings
	@test -s build/wings/$*/top_baffle_nd25fw4_obiwan_wing_$*.step
	@test -s build/wings/$*/top_baffle_nd25fw4_obiwan_wing_$*_assembled.step
	@touch $@

check_obiwan_wings: $(OBIWAN_WING_STAMPS) \
        tests/test_obiwan_wings.py
	$(call RUN_STATE,0,obiwan,.) \
	    tests/test_obiwan_wings.py --artifact-root build/wings
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/wings/ac/stl build/wings/ae/stl'

$(OBIWAN_WING_DESIGN_MAP): scripts/gen_obiwan_wing_design_map.py \
        scripts/gen_driver_overlay.py src/lx521_baffle/proud/top_baffle_nd25fw4_a_comp.py \
        src/lx521_baffle/proud/top_baffle_nd25fw4_b.py src/lx521_baffle/obiwan/carriers.py \
        src/lx521_baffle/obiwan/bridge.py \
        src/lx521_baffle/obiwan/floor.py
	@rm -f $@
	$(call RUN_STATE,0,obiwan,.) \
	    scripts/gen_obiwan_wing_design_map.py --output $@
	@test -s $@

no_floor_stand_artifacts: validate_no_floor_obiwan_stage \
        $(call VARIANT_TARGETS,build/no_floor_stand)
	# Symmetric with the floor state: expose this sweep while independent
	# floor-state artifact work can continue under the same Make jobserver.
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='build/no_floor_stand/stl'
no_floor_stand: no_floor_stand_artifacts
common: build/common/top_baffle_nd25fw4_attachments.step

check: validate_obiwan_stages check_static check_clearance_suite check_r6f_suite \
        check_obiwan_junction_closures

check_static: check_shared_contracts check_remote_transport check_release_metadata \
        check_slice_metadata check_to_print_shelf check_captive_magnets \
        check_obiwan_lm_split_two_pin_static

check_shared_contracts: tests/test_shared_contracts.py src/lx521_baffle/io.py \
        src/lx521_baffle/magnet_contract.py src/lx521_baffle/print_contract.py src/lx521_baffle/stl_export.py \
        src/lx521_baffle/assembly.py src/lx521_baffle/geom.py scripts/run_memory_guarded.py
	$(PYTHON_RUN) tests/test_shared_contracts.py

check_remote_transport: tests/test_remote_cad.py scripts/remote_cad.py Makefile
	$(PYTHON_RUN) tests/test_remote_cad.py

check_release_metadata: $(CAPTIVE_MAGNET_CATALOG) \
        tests/test_release_metadata.py scripts/check_manifold.py \
        $(PRINT_CONTRACT_SRCS) src/lx521_baffle/stl_export.py src/lx521_baffle/geom.py \
        scripts/write_obiwan_release_manifest.py Makefile
	$(PYTHON_RUN) tests/test_release_metadata.py

check_slice_metadata: tests/test_slice_captive_magnets.py \
        $(SLICER_SRCS) scripts/json_schema_subset.py \
        captive_magnet_slicing_profile.json src/lx521_baffle/magnet_contract.py \
        $(PRINT_CONTRACT_SRCS)
	$(PYTHON_RUN) tests/test_slice_captive_magnets.py

check_to_print_shelf: $(CAPTIVE_MAGNET_CATALOG) \
        tests/test_to_print_shelf.py scripts/build_to_print_shelf.py \
        to_print/catalog.json \
        $(SLICER_SRCS) scripts/bambu_3mf_audit.py $(PRINT_CONTRACT_SRCS)
	$(PYTHON_RUN) tests/test_to_print_shelf.py

# These targets consume an existing, remotely generated catalog but never list
# it as a Make prerequisite: a missing catalog must not launch heavyweight CAD
# on this workstation.  They run Bambu Studio only as a local CLI slicer and do
# not contain any printer transport/upload/start operation.
BAMBU_JOBS ?= 1
BAMBU_AUDIT_OUTPUT ?= review/captive_magnet_slice_audit

define REQUIRE_BAMBU_CATALOG
	@test -s "$(CAPTIVE_MAGNET_CATALOG)" || { \
		echo "missing $(CAPTIVE_MAGNET_CATALOG)" >&2; \
		echo "run 'make obiwan_release' first to generate/promote it remotely, then rerun this local Bambu target" >&2; \
		exit 1; \
	}
endef

bambu_profiles: check_slice_metadata
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--prepare-profiles-only

bambu_dry_run: check_slice_metadata
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--dry-run

bambu_slice_obiwan: check_slice_metadata
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--only '*Obi-Wan*' --emit-ready-projects

bambu_slice_release: check_slice_metadata
	$(REQUIRE_BAMBU_CATALOG)
	$(PYTHON_RUN) scripts/slice_captive_magnets.py \
		--catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--output "$(BAMBU_AUDIT_OUTPUT)" \
		--jobs "$(BAMBU_JOBS)" \
		--emit-ready-projects

# User-facing, local-only materialization of the 39 P2S-printable parts.
# The full audited magnet pass is deliberately a prerequisite: a friendly
# filename must never promote a stale or discovery-only 3MF to a ready project.
TO_PRINT_SHELF ?= to_print

to_print: check_to_print_shelf bambu_slice_release
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--slice-missing

to_print_validate: check_to_print_shelf
	$(PYTHON_RUN) scripts/build_to_print_shelf.py \
		--shelf "$(TO_PRINT_SHELF)" \
		--catalog "$(TO_PRINT_SHELF)/catalog.json" \
		--release-catalog "$(CAPTIVE_MAGNET_CATALOG)" \
		--release-audit "$(BAMBU_AUDIT_OUTPUT)" \
		--validate-only

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
# junctions.  The dense layer matrix is sixteen ordinary Make nodes (the four
# floor/no-floor × LM-UM/T-UM cases, each split into four complementary layer
# shards) so GNU Make's inherited jobserver keeps all sixteen remote slots
# busy
# when the LM sections outlive the lighter T sections.  The shard union retains
# every physical Bambu layer and assertion;
# Python does not own a second scheduler.  The explicit full-test flag makes a
# missing authenticated guard fail closed instead of silently running only the
# source contract.
check_obiwan_junction_closures: $(JUNCTION_CLOSURE_STAMP)

# Focused two-state exact-BREP gate used while iterating closure ownership.
# It stops before the sixteen dense physical-layer shards.
check_obiwan_junction_closure_base: $(JUNCTION_CLOSURE_BASE_STAMP)

$(JUNCTION_CLOSURE_STAMP): $(JUNCTION_CLOSURE_BASE_STAMP) \
		$(JUNCTION_CLOSURE_DENSE_STAMPS)
	@mkdir -p $(@D)
	@touch $@

$(JUNCTION_CLOSURE_BASE_STAMP): \
		$(call r6f_check_stamp,floor_feed_and_flush_mouth_contract) \
		$(call r6f_check_stamp,feed_and_flush_mouth_contract) \
		tests/test_obiwan_junction_closure_webs.py \
		src/lx521_baffle/obiwan/carriers.py \
		src/lx521_baffle/obiwan/attachments.py \
		$(OBIWAN_SRCS) $(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		build/floor_stand/.obiwan_stage/manifest.json \
		build/no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $(@D)
	# Run both base checks so each state is compared with its own route oracle.
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_BASE_ONLY=1 $(RUN) \
	    tests/test_obiwan_junction_closure_webs.py
	LX_STAND_FOOT=1 LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_BASE_ONLY=1 $(RUN) \
	    tests/test_obiwan_junction_closure_webs.py
	@touch $@

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
	# corresponding immutable staged BREP. ``floor_%`` intentionally excludes
	# the ``no_floor_%`` cases.
	LX_STAND_FOOT=$(if $(filter floor_%,$(1)),1,0) LX_ROUTING_PROFILE=obiwan \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_DENSE_CASE=$(1) \
	    LX_OBIWAN_CLOSURE_DENSE_SHARD=$(2)/4 $$(RUN) \
	    tests/test_obiwan_junction_closure_webs.py
	@touch $$@
endef

$(foreach name,$(JUNCTION_CLOSURE_DENSE_CASES),\
        $(foreach shard,$(JUNCTION_CLOSURE_DENSE_SHARDS),\
        $(eval $(call JUNCTION_CLOSURE_DENSE_RULE,$(name),$(shard)))))

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
        build/floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step \
        _check_r6f_floor_lm_keyed_split

check_no_floor_obiwan_lm_split: validate_no_floor_obiwan_stage \
        build/no_floor_stand/stl/.stamp_obiwan \
        build/no_floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step \
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

build/common/top_baffle_nd25fw4_attachments.step: $(SRCS) scripts/export_steps.py
	@mkdir -p $(@D)
	LX_ROUTING_PROFILE=proud $(RUN) scripts/export_steps.py \
	    lx521_baffle.proud.top_baffle_nd25fw4_attachments \
	    --output $@

# B2/C7/V0/V1 share one export skeleton; V1L and Obi-Wan stay hand-written
# because their local/remote staging behavior is genuinely different.
STL_VARIANTS := b2 c7 v0 v1
stl_variant_stamp = $(if $(filter b2,$(1)),.stamp,.stamp_$(1))
stl_variant_arg = $(if $(filter b2,$(1)),, --variant $(1))

# $(1) = output root, $(2) = LX_STAND_FOOT, $(3) = variant.
define STATE_STL_VARIANT_RULE
$(1)/stl/$$(call stl_variant_stamp,$(3)): $$(SRCS) $$(STL_EXPORT_SRCS)
	@rm -f $$@
	$$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py$$(call stl_variant_arg,$(3)) --outdir $(1)/stl
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
$(1)/stl/.stamp_v1l: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
ifeq ($(LX_CAD_EXECUTION),remote-worker)
	# Osado constructs the shared V1L geometry once, then meshes the same
	# four physical print pieces and grommet pair from that one build.
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --outdir $(1)/stl
else
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --v1l-piece piece_bottom --outdir $(1)/stl
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --v1l-piece piece_mid_left --outdir $(1)/stl
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --v1l-piece piece_mid_right --outdir $(1)/stl
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --v1l-piece piece_top_b2 --outdir $(1)/stl
	$(call RUN_STATE,$(2),proud) scripts/export_piece_stls.py --variant v1l --v1l-piece grommet --outdir $(1)/stl
endif
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
$(1)/top_baffle_nd25fw4_obiwan_%.step: \
		$(1)/.obiwan_stage/manifest.json scripts/export_obiwan_staged.py \
		scripts/export_steps.py $(OBIWAN_SRCS)
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),obiwan,.) \
	    scripts/export_obiwan_staged.py step \
	    --manifest $(1)/.obiwan_stage/manifest.json \
	    --kind $$* --output $$@

$(1)/top_baffle_nd25fw4_um_fit.step: src/lx521_baffle/um_fit.py \
        $(SRCS) scripts/export_steps.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_steps.py \
	    lx521_baffle.um_fit --output $$@

# Osado builds V1L's shared baffle/duct tree once and emits the same four
# physical children. Explicit local mode retains one-piece-per-child staging.
$(1)/top_baffle_nd25fw4_v1l_split.step: scripts/export_v1l_staged_step.py \
        scripts/export_steps.py $(SRCS)
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_v1l_staged_step.py --output $$@

$(1)/%.step: src/lx521_baffle/proud/%.py $(SRCS) scripts/export_steps.py
	@mkdir -p $(1)
	$(call RUN_STATE,$(2),proud,.) scripts/export_steps.py \
	    lx521_baffle.proud.$$* --output $$@

$(1)/baffle_cable_routing_proud.png: $(SRCS) scripts/gen_cable_routing.py \
        scripts/gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE,$(2),proud) \
	    ../../scripts/gen_cable_routing.py

$(1)/baffle_cable_routing_obiwan.png: $(OBIWAN_SRCS) scripts/gen_cable_routing.py \
        scripts/gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE,$(2),obiwan) \
	    ../../scripts/gen_cable_routing.py

$(1)/obiwan_release_manifest.json: $(OBIWAN_RELEASE_SRCS) \
        $(1)/stl/.stamp_obiwan $(1)/stl/.stamp_coupon \
        $(1)/top_baffle_nd25fw4_obiwan_split.step \
        $(1)/top_baffle_nd25fw4_obiwan_lm_split.step \
        $(1)/top_baffle_nd25fw4_obiwan_attachments.step \
        $(1)/top_baffle_nd25fw4_obiwan_assembled.step \
        $(1)/top_baffle_nd25fw4_um_fit.step \
        $(1)/baffle_cable_routing_proud.png \
        $(1)/baffle_cable_routing_obiwan.png \
        $(if $(filter 1,$(2)),$(FLOOR_STRENGTH_REPORTS)) \
        $(addprefix $(1)/,$(OVERLAY_PNGS))
	@rm -f $$@
	$(call RUN_STATE,$(2),obiwan) \
	    scripts/write_obiwan_release_manifest.py --state-dir $(1)

$(1)/.stamp_overlays: $(SRCS) scripts/gen_driver_overlay.py \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	@mkdir -p $(1)
	cd $(1) && $(call RUN_STATE,$(2),proud) \
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
        build/floor_stand/top_baffle_nd25fw4_obiwan_split.step \
        build/floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	@rm -f $@ $(FLOOR_STRENGTH_REPORTS)
	$(RUN) -m lx521_baffle.obiwan.floor_strength \
	    --json build/floor_stand/obiwan_integrated_floor_strength.json \
	    --markdown build/floor_stand/obiwan_integrated_floor_strength.md \
	    --step build/floor_stand/top_baffle_nd25fw4_obiwan_split.step \
	    --step build/floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	@touch $@

$(eval $(call recover_stamp,$(FLOOR_STRENGTH_REPORTS),build/floor_stand/.stamp_obiwan_integrated_floor_strength))

clean:
	rm -rf build/floor_stand build/no_floor_stand build/wings \
	    .manifold-stamps .check-stamps __pycache__ \
	    build/common/top_baffle_nd25fw4_attachments.step \
	    build/common/obiwan_wing_design_map.png

endif
