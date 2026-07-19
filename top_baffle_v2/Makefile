# Generates every flag-dependent artifact for BOTH stand-foot states:
#
#   floor_stand/      LX_STAND_FOOT=1  fused foot + NL8 panel; Obi-Wan carries
#                     its integral floor mount in the LM carrier
#   no_floor_stand/   LX_STAND_FOOT=0  flat piece_bottom, bridge
#                     pass-throughs; Obi-Wan LM owns a front-flush solid web
#
# Each folder gets a complete print set (stl/), all review STEP models, and
# five documentation PNGs (three overlays + two routing profiles).
# Proud attachments.step is flag-independent and stays at the top level.
# Obi-Wan LM core and its attachment manifest differ by stand state; the floor
# mount is integral LM material, while the remaining attachments are optional.
#
# Self-contained: STLs/PNGs use build123d/shapely/matplotlib/numpy and
# STEPs use build123d's native exporter via export_steps.py -- no
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
#
# Compatible with the macOS-provided GNU make 3.81.

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
	$(REMOTE_CAD_PYTHON) remote_cad.py run $(REMOTE_CAD_GOALS)
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
CAD_SRCS_ALL := $(wildcard top_baffle_nd25fw4*.py)
# These modules are consumed only by the Obi-Wan release family.  Keep the
# legacy/common dependency set conservative (it still contains the shared
# Obi-Wan route/floor stack reached through top_baffle_nd25fw4_cables.py), while
# preventing a carrier/web/split-only edit from invalidating every B/C/V0/V1/
# V1L STEP and STL.  GNU Make remains the dependency authority.
OBIWAN_EXCLUSIVE_CAD_SRCS := \
        top_baffle_nd25fw4_obiwan.py \
        top_baffle_nd25fw4_obiwan_assembled.py \
        top_baffle_nd25fw4_obiwan_attachments.py \
        top_baffle_nd25fw4_obiwan_bridge.py \
        top_baffle_nd25fw4_obiwan_lm_split.py \
        top_baffle_nd25fw4_obiwan_split.py
SRCS := $(filter-out $(OBIWAN_EXCLUSIVE_CAD_SRCS),$(CAD_SRCS_ALL)) \
        captive_magnets.py Makefile $(MU10_REFS) $(W22_REFS)
OBIWAN_SRCS := $(SRCS) $(OBIWAN_EXCLUSIVE_CAD_SRCS)
EXECUTION_SRCS := run_memory_guarded.py
PRINT_CONTRACT_SRCS := front_down_contract.py
OBIWAN_RELEASE_SRCS := $(OBIWAN_SRCS) $(EXECUTION_SRCS) $(PRINT_CONTRACT_SRCS) \
        export_steps.py export_obiwan_staged.py \
        export_piece_stls.py export_coupon.py gen_cable_routing.py \
        gen_driver_overlay.py check_manifold.py test_captive_magnets.py \
        test_clearances.py \
        test_obiwan_r6f.py test_obiwan_junction_closure_webs.py \
        test_obiwan_lm_split_two_pin_static.py test_remote_cad.py \
        write_obiwan_release_manifest.py obiwan_physical_qualification.md \
        remote_cad.py cad-remote-requirements.lock
# export_piece_stls imports the strict checker after meshing, and that checker
# imports the release-manifest helpers at module load.  Keep every STL stamp
# source-complete so a checker/contract change cannot reuse a stale mesh stamp.
STL_EXPORT_SRCS := export_piece_stls.py check_manifold.py \
        $(PRINT_CONTRACT_SRCS) \
        write_obiwan_release_manifest.py
# Command-line Make variables must not turn a candidate/release recipe into
# an unguarded OCC process. PYTHON remains selectable, but every invocation
# is forced through this workspace's authenticated process-tree guard.
override GUARD := $(abspath run_memory_guarded.py)
override RUN := $(PYTHON_RUN) $(GUARD) -- $(PYTHON_RUN)

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
FLOOR_STRENGTH_REPORTS = floor_stand/obiwan_integrated_floor_strength.json \
                         floor_stand/obiwan_integrated_floor_strength.md
OBIWAN_WING_DESIGN_MAP = obiwan_wing_design_map.png
OBIWAN_WING_SLUGS = ac ae
OBIWAN_WING_STAMPS = $(addprefix wings/.stamp_,$(OBIWAN_WING_SLUGS))
CAPTIVE_MAGNET_CATALOG = review/captive_magnet_release_catalog.json
OBIWAN_WING_INPUTS = cad-remote-requirements.lock \
        obiwan_acoustic_wings_spec.md captive_magnets.py \
        front_down_contract.py gen_driver_overlay.py \
        export_obiwan_staged.py \
        top_baffle_nd25fw4.py top_baffle_nd25fw4_b.py \
        top_baffle_nd25fw4_b1.py top_baffle_nd25fw4_b2.py \
        top_baffle_nd25fw4_cables.py top_baffle_nd25fw4_flush.py \
        top_baffle_nd25fw4_v1.py top_baffle_nd25fw4_obiwan.py \
        top_baffle_nd25fw4_obiwan_attachments.py \
        top_baffle_nd25fw4_obiwan_route.py top_baffle_nd25fw4_a_comp.py \
        top_baffle_nd25fw4_obiwan_bridge.py \
        top_baffle_nd25fw4_obiwan_floor.py \
        top_baffle_nd25fw4_obiwan_floor_strength.py \
        top_baffle_nd25fw4_obiwan_lm_split.py \
        floor_stand/.obiwan_stage/manifest.json \
        no_floor_stand/.obiwan_stage/manifest.json

# Strict STL topology is embarrassingly parallel, but the former checker ran
# every mesh in one Python process.  A recursive submake is used deliberately:
# it reparses only after the producer targets have populated the requested
# roots, inherits GNU Make's jobserver, and exposes one ordinary stamp target
# per STL.  There is no second scheduler or opaque cache.  The final phony
# node retains the manifest/review/cross-state gate after all mesh stamps pass.
MANIFOLD_ROOTS ?= floor_stand/stl no_floor_stand/stl
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
        test_v1_ts_captive_keepout_nudge \
        test_standard_captive_magnet_contract \
        test_standard_local_magnet_backing \
        test_duct_vs_w22_pilots \
        test_duct_duct_separation \
        test_duct_vs_um_pilots \
        test_c7_duct_corridor \
        test_seam_keys_vs_ducts \
        test_v0_duct_corridor \
        test_v0_captive_geometry \
        test_v1_field \
        test_route_smoothness \
        test_v1l_um_terminal_axis_handoff \
        test_um_eroded_outline_containment \
        test_ts_eroded_outline_containment \
        test_bridge_inserts \
        test_margin_dashboard \
        test_cutter_health \
        test_v1l_mid_right_terminal_duct_topology \
        test_v1l_upper_dovetail_depth_mid_left \
        test_v1l_upper_dovetail_depth_mid_right

R6F_CHECK_NAMES := \
        test_route_contract \
        test_w22_reference_step_geometry \
        test_insert_bump_clearance \
        test_floor_insert_bump_clearance \
        test_no_floor_route_smoothness \
        test_floor_route_smoothness \
        test_bump_brep_clearance \
        test_floor_bump_brep_clearance \
        test_bump_backfill_contract \
        test_floor_bump_backfill_contract \
        test_lm_burial_web_contract \
        test_floor_lm_burial_web_contract \
        test_um_burial_web_contract \
        test_floor_um_burial_web_contract \
        test_feed_and_flush_mouth_contract \
        test_floor_feed_and_flush_mouth_contract \
        test_crossover_brep \
        test_floor_crossover_brep \
        test_bridge_contract \
        test_bridge_geometry \
        test_joint_load_contract \
        test_floor_lm_core \
        test_no_floor_lm_core \
        test_floor_lm_keyed_split \
        test_no_floor_lm_keyed_split \
        test_floor_um_shell \
        test_floor_t_shell \
        test_no_floor_um_shell \
        test_no_floor_t_shell \
        test_lm_cable_clearance \
        test_um_cable_clearance \
        test_floor_lm_cable_clearance \
        test_floor_um_cable_clearance \
        test_floor_integrated_mount \
        test_tweeter_and_service \
        test_floor_tweeter_and_service

R6F_SERVICE_CHECK_NAMES := test_tweeter_and_service \
                           test_floor_tweeter_and_service
R6F_PRIVATE_ENV_NAMES := \
        LX_R6F_EXPORT_CARRIER LX_R6F_EXPORT_PATH LX_R6F_EXPORT_SHELL \
        LX_R6F_EXPORT_SHELL_SEGMENT LX_R6F_EXPORT_SHELL_SEGMENT_COUNT \
        LX_R6F_EXPORT_SHELL_WALL LX_R6F_EXPORT_TWEETER \
        LX_R6F_LM_INPUT_PATH LX_R6F_SERVICE_COMPONENT_CHUNK_COUNT \
        LX_R6F_SERVICE_COMPONENT_COLLISION LX_R6F_SERVICE_COMPONENT_EXPORT \
        LX_R6F_SERVICE_COMPONENT_PATH LX_R6F_SERVICE_COMPONENT_SELECTOR \
        LX_R6F_SERVICE_PHASE LX_R6F_SERVICE_PRINTED_PATH \
        LX_R6F_SERVICE_START_PHASE LX_R6F_SERVICE_STOP_AFTER_PHASE \
        LX_R6F_SINGLE_CHECK LX_R6F_SUBTRACT_INPUT LX_R6F_SUBTRACT_OUTPUT \
        LX_R6F_SUBTRACT_PART LX_R6F_VALIDATE_CABLE \
        LX_R6F_VALIDATE_CABLE_MODE LX_R6F_VALIDATE_CABLE_SEGMENT \
        LX_R6F_VALIDATE_CABLE_SEGMENT_COUNT LX_R6F_VALIDATE_CARRIER_OWNER \
        LX_R6F_VALIDATE_CARRIER_PATH LX_R6F_VALIDATE_PART_PATHS \
        LX_R6F_VALIDATE_SHELL LX_R6F_VALIDATE_SHELL_PATH \
        LX_R6F_VALIDATE_SHELL_PATHS
R6F_CLEAN_ENV = env $(foreach name,$(R6F_PRIVATE_ENV_NAMES),-u $(name))
CLEARANCE_CHECK_TARGETS := $(addprefix _check_clearance_,$(CLEARANCE_CHECK_NAMES))
R6F_CHECK_TARGETS := $(addprefix _check_r6f_,$(R6F_CHECK_NAMES))
clearance_check_stamp = $(CHECK_STAMP_DIR)/clearance/$(1).ok
r6f_check_stamp = $(CHECK_STAMP_DIR)/r6f/$(1).ok
CLEARANCE_CHECK_STAMPS := $(foreach name,$(CLEARANCE_CHECK_NAMES),\
        $(call clearance_check_stamp,$(name)))
R6F_CHECK_STAMPS := $(foreach name,$(R6F_CHECK_NAMES),\
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
$(call manifold_stamp,$(1)): $(1) check_manifold.py \
        $(PRINT_CONTRACT_SRCS) write_obiwan_release_manifest.py Makefile
	@mkdir -p $$(@D)
	$(RUN) check_manifold.py --stl-only $(1)
	@touch $$@
endef

$(foreach stl,$(MANIFOLD_STLS),$(eval $(call MANIFOLD_STL_RULE,$(stl))))

define CLEARANCE_CHECK_RULE
_check_clearance_$(1): $(call clearance_check_stamp,$(1))
$(call clearance_check_stamp,$(1)): test_clearances.py $(OBIWAN_SRCS) \
        $(EXECUTION_SRCS) $(CHECK_REVALIDATE)
	@mkdir -p $$(@D)
	LX_CLEARANCE_SINGLE_CHECK=$(1) $(RUN) test_clearances.py
	@touch $$@
endef

define R6F_CHECK_RULE
_check_r6f_$(1): $(call r6f_check_stamp,$(1))
$(call r6f_check_stamp,$(1)): test_obiwan_r6f.py $(OBIWAN_SRCS) \
		$(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		floor_stand/.obiwan_stage/manifest.json \
		no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $$(@D)
	$(R6F_CLEAN_ENV) LX_R6F_SINGLE_CHECK=$(1) $(RUN) test_obiwan_r6f.py
	@touch $$@
endef

# On a workstation these two checks deliberately remain lightweight parent
# orchestrators so every service substep gets a fresh guarded lifecycle.  On
# osado the selected process self-wraps once and performs the full large-host
# matrix under that one Make token.
define R6F_SERVICE_CHECK_RULE
_check_r6f_$(1): $(call r6f_check_stamp,$(1))
$(call r6f_check_stamp,$(1)): test_obiwan_r6f.py $(OBIWAN_SRCS) \
		$(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		floor_stand/.obiwan_stage/manifest.json \
		no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $$(@D)
	$(R6F_CLEAN_ENV) LX_R6F_SINGLE_CHECK=$(1) \
	    $(PYTHON_RUN) test_obiwan_r6f.py
	@touch $$@
endef

$(foreach name,$(CLEARANCE_CHECK_NAMES),\
    $(eval $(call CLEARANCE_CHECK_RULE,$(name))))
$(foreach name,$(filter-out $(R6F_SERVICE_CHECK_NAMES),$(R6F_CHECK_NAMES)),\
    $(eval $(call R6F_CHECK_RULE,$(name))))
$(foreach name,$(R6F_SERVICE_CHECK_NAMES),\
    $(eval $(call R6F_SERVICE_CHECK_RULE,$(name))))

VARIANT_TARGETS = $(1)/stl/.stamp_coupon $(1)/stl/.stamp $(1)/stl/.stamp_c7 $(1)/stl/.stamp_v0 $(1)/stl/.stamp_v1 $(1)/stl/.stamp_v1l $(1)/stl/.stamp_obiwan \
                  $(addprefix $(1)/,$(addsuffix .step,$(GENS))) \
                  $(1)/baffle_cable_routing_proud.png \
                  $(1)/baffle_cable_routing_obiwan.png \
                  $(addprefix $(1)/,$(OVERLAY_PNGS)) \
                  $(1)/obiwan_release_manifest.json

.PHONY: all candidate release validate_obiwan_stages \
        validate_floor_obiwan_stage validate_no_floor_obiwan_stage \
        floor_stand floor_stand_artifacts floor_obiwan \
        no_floor_stand no_floor_stand_artifacts no_floor_obiwan obiwan_release \
        obiwan_wings obiwan_wing_artifacts check_obiwan_wings \
        common check check_static check_remote_transport \
        check_release_metadata check_slice_metadata \
        check_clearance_suite check_r6f_suite \
        check_captive_magnets check_obiwan \
        check_obiwan_shells check_obiwan_t_shells \
        check_obiwan_service check_obiwan_closure_focus \
        check_obiwan_mouths check_obiwan_burial check_obiwan_backfills \
        check_obiwan_um_burial \
        check_obiwan_route_boundaries \
        check_floor_um_shell check_floor_t_shell \
        check_no_floor_um_shell check_no_floor_t_shell \
        check_floor_obiwan_mouths check_no_floor_obiwan_mouths \
        check_floor_obiwan_burial check_no_floor_obiwan_burial \
        check_floor_obiwan_um_burial check_no_floor_obiwan_um_burial \
        check_floor_obiwan_backfills check_no_floor_obiwan_backfills \
        check_route_contract check_bump_brep check_floor_integrated_mount \
        check_no_floor_lm_mesh check_obiwan_lm_split \
        check_obiwan_lm_split_two_pin_static \
        check_obiwan_junction_closure_plans check_obiwan_junction_closures \
        check_floor_obiwan_lm_split check_no_floor_obiwan_lm_split \
        check_obiwan_lm_profile check_obiwan_lm_profile_plan \
        check_obiwan_lm_profile_brep manifold _manifold_parallel clean \
        $(CLEARANCE_CHECK_TARGETS) $(R6F_CHECK_TARGETS)
all: validate_obiwan_stages floor_stand_artifacts no_floor_stand_artifacts common \
        obiwan_wings $(CAPTIVE_MAGNET_CATALOG)
	# Each state already completed its one-node-per-STL sweep while the other
	# state was still eligible to use the shared jobserver.  Keep only the
	# inexpensive cross-state manifest/review consistency gate on the join.
	$(RUN) check_manifold.py --metadata-only \
	    floor_stand/stl no_floor_stand/stl

# This catalog is generated only after all 56 released magnet-bearing meshes
# and their exact front-down print transforms exist.  It is the CAD-side
# authority consumed by the local, no-printer-contact Bambu slicing audit.
$(CAPTIVE_MAGNET_CATALOG): generate_captive_magnet_catalog.py \
        captive_magnet_release_catalog.schema.json slice_captive_magnets.py \
        json_schema_subset.py \
        captive_magnets.py \
        floor_stand/stl/.stamp floor_stand/stl/.stamp_c7 \
        floor_stand/stl/.stamp_v0 floor_stand/stl/.stamp_v1 \
        floor_stand/stl/.stamp_v1l floor_stand/stl/.stamp_obiwan \
        floor_stand/stl/.stamp_coupon \
        no_floor_stand/stl/.stamp no_floor_stand/stl/.stamp_c7 \
        no_floor_stand/stl/.stamp_v0 no_floor_stand/stl/.stamp_v1 \
        no_floor_stand/stl/.stamp_v1l no_floor_stand/stl/.stamp_obiwan \
        no_floor_stand/stl/.stamp_coupon $(OBIWAN_WING_STAMPS)
	@mkdir -p review
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    generate_captive_magnet_catalog.py --output $@
	@test -s $@

# Runtime package versions are part of the native-stage fingerprint but not
# filesystem prerequisites. Validate first on every build; the stage command
# is a no-write fast path when current and atomically rebuilds only on drift.
validate_obiwan_stages: validate_floor_obiwan_stage validate_no_floor_obiwan_stage

validate_floor_obiwan_stage: export_obiwan_staged.py $(OBIWAN_SRCS)
	@mkdir -p floor_stand/.obiwan_stage
	LX_STAND_FOOT=1 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    export_obiwan_staged.py stage \
	    --manifest floor_stand/.obiwan_stage/manifest.json

validate_no_floor_obiwan_stage: export_obiwan_staged.py $(OBIWAN_SRCS)
	@mkdir -p no_floor_stand/.obiwan_stage
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    export_obiwan_staged.py stage \
	    --manifest no_floor_stand/.obiwan_stage/manifest.json

# The manifest files are aliases for the always-validated state transactions.
# This gives every downstream artifact one shared Make node and prevents a
# parallel build from racing the phony validator against a second file recipe.
floor_stand/.obiwan_stage/manifest.json: validate_floor_obiwan_stage
	@test -f $@
no_floor_stand/.obiwan_stage/manifest.json: validate_no_floor_obiwan_stage
	@test -f $@

candidate: check all

release: candidate
	# Candidate already completed the strict mesh sweep.  Release adds only the
	# fail-closed physical-authorization and metadata gate.
	$(RUN) check_manifold.py --metadata-only --require-release-authorized

manifold: check_manifold.py
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='floor_stand/stl no_floor_stand/stl'

_manifold_parallel: $(MANIFOLD_STAMPS)
	@test -n "$(strip $(MANIFOLD_STLS))" || { \
	    echo "no STLs found under: $(MANIFOLD_ROOTS)" >&2; exit 1; }
	$(RUN) check_manifold.py --metadata-only \
	    $(if $(filter 1,$(MANIFOLD_OBIWAN_ONLY)),--obiwan-only) \
	    $(MANIFOLD_ROOTS)
	@echo "$(words $(MANIFOLD_STLS))/$(words $(MANIFOLD_STLS)) STLs strict-manifold (Make-parallel)"

# polar-measurement turntable (mates under the floor-stand foot only)
floor_stand_artifacts: validate_floor_obiwan_stage floor_stand/stl/.stamp_polar
floor_stand/stl/.stamp_polar: polar_index_base.py
	@rm -f $@
	$(RUN) polar_index_base.py
	@touch $@
floor_stand_artifacts: $(call VARIANT_TARGETS,floor_stand) \
        $(FLOOR_STRENGTH_REPORTS)
	# Start this state's topology fan-out as soon as its final artifact lands;
	# the recursive Make inherits the enclosing GNU Make jobserver.
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='floor_stand/stl'
floor_stand: floor_stand_artifacts
# Focused, atomic Obi-Wan floor release.  This avoids rebuilding unrelated B/C/
# V0/V1/V1L pieces while still producing every Obi-Wan STEP/STL/PNG, the coupon
# set, overlays, hash-bound strength report, release manifest and strict QA.
floor_obiwan: validate_floor_obiwan_stage \
        floor_stand/obiwan_release_manifest.json
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='floor_stand/stl' MANIFOLD_OBIWAN_ONLY=1

# Symmetric focused release for the no-floor carrier.  Keep this narrower than
# no_floor_stand: it emits the complete hash-backed Obi-Wan family without
# rebuilding unrelated proud/C/V0/V1/V1L print sets.
no_floor_obiwan: validate_no_floor_obiwan_stage \
        no_floor_stand/obiwan_release_manifest.json
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='no_floor_stand/stl' MANIFOLD_OBIWAN_ONLY=1

# One public remote target owns the complete two-state Obi-Wan release, Ac/Ae
# magnetic attachments and the focused final-BREP interface checks.  A single
# Make DAG lets osado share both native stages and fan independent state,
# wing and validation recipes across its bounded jobserver slots.
obiwan_release: floor_stand_artifacts no_floor_stand_artifacts \
        obiwan_wings \
        $(CAPTIVE_MAGNET_CATALOG) \
        check_route_contract check_obiwan_lm_split check_obiwan_shells \
        check_obiwan_route_boundaries check_floor_integrated_mount \
        check_obiwan_lm_split_two_pin_static \
        check_obiwan_junction_closures
	# Ac/Ae already pass their own final gate through obiwan_wings.
	# State-local sweeps already ran exactly once through the artifact nodes;
	# retain only the inexpensive cross-state metadata gate on this join.
	$(RUN) check_manifold.py --metadata-only \
	    floor_stand/stl no_floor_stand/stl

# Source-linked, dimensioned Ac/Ae attachment layout for the Obi-Wan carrier.
# Ac and Ae are the complete STEP-first wing inventory. Their
# independent stamps let osado build both slug trees concurrently; each tree is
# one atomic, hash-backed transaction containing STEP, STL, facts and review
# images.  The PNG remains the shared analytic contract and is regenerated
# before either solid model.
obiwan_wings: $(OBIWAN_WING_DESIGN_MAP) obiwan_wing_artifacts

obiwan_wing_artifacts: $(OBIWAN_WING_STAMPS) check_obiwan_wings

wings/.stamp_ac: export_obiwan_wings.py obiwan_wings_cad.py \
        gen_obiwan_wing_design_map.py $(OBIWAN_WING_DESIGN_MAP) \
        check_manifold.py $(OBIWAN_WING_INPUTS)
	@rm -f $@
	@mkdir -p wings
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    export_obiwan_wings.py --slug ac --output-root wings
	@test -s wings/ac/top_baffle_nd25fw4_obiwan_wing_ac.step
	@test -s wings/ac/top_baffle_nd25fw4_obiwan_wing_ac_assembled.step
	@touch $@

wings/.stamp_ae: export_obiwan_wings.py obiwan_wings_cad.py \
        gen_obiwan_wing_design_map.py $(OBIWAN_WING_DESIGN_MAP) \
        check_manifold.py $(OBIWAN_WING_INPUTS)
	@rm -f $@
	@mkdir -p wings
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    export_obiwan_wings.py --slug ae --output-root wings
	@test -s wings/ae/top_baffle_nd25fw4_obiwan_wing_ae.step
	@test -s wings/ae/top_baffle_nd25fw4_obiwan_wing_ae_assembled.step
	@touch $@

check_obiwan_wings: $(OBIWAN_WING_STAMPS) \
        test_obiwan_wings.py
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    test_obiwan_wings.py --artifact-root wings
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='wings/ac/stl wings/ae/stl'

$(OBIWAN_WING_DESIGN_MAP): gen_obiwan_wing_design_map.py \
        gen_driver_overlay.py top_baffle_nd25fw4_a_comp.py \
        top_baffle_nd25fw4_b.py top_baffle_nd25fw4_obiwan.py \
        top_baffle_nd25fw4_obiwan_bridge.py \
        top_baffle_nd25fw4_obiwan_floor.py
	@rm -f $@
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    gen_obiwan_wing_design_map.py --output $@
	@test -s $@

no_floor_stand_artifacts: validate_no_floor_obiwan_stage \
        $(call VARIANT_TARGETS,no_floor_stand)
	# Symmetric with the floor state: expose this sweep while independent
	# floor-state artifact work can continue under the same Make jobserver.
	$(MAKE) --no-print-directory _manifold_parallel \
	    MANIFOLD_ROOTS='no_floor_stand/stl'
no_floor_stand: no_floor_stand_artifacts
common: top_baffle_nd25fw4_attachments.step

check: validate_obiwan_stages check_static check_clearance_suite check_r6f_suite \
        check_obiwan_junction_closures

check_static: check_remote_transport check_release_metadata \
        check_slice_metadata check_captive_magnets \
        check_obiwan_lm_split_two_pin_static

check_remote_transport: test_remote_cad.py remote_cad.py Makefile
	$(PYTHON_RUN) test_remote_cad.py

check_release_metadata: test_release_metadata.py check_manifold.py \
        $(PRINT_CONTRACT_SRCS) write_obiwan_release_manifest.py Makefile
	$(PYTHON_RUN) test_release_metadata.py

check_slice_metadata: test_slice_captive_magnets.py \
        slice_captive_magnets.py json_schema_subset.py \
        captive_magnet_slicing_profile.json $(PRINT_CONTRACT_SRCS)
	$(PYTHON_RUN) test_slice_captive_magnets.py

check_clearance_suite: $(CLEARANCE_CHECK_STAMPS)
	@echo "all clearance checks passed (Make-jobserver isolated)"

check_r6f_suite: $(R6F_CHECK_STAMPS)
	@echo "all final Obi-Wan R6F checks passed (Make-jobserver isolated)"

check_captive_magnets: test_captive_magnets.py captive_magnets.py
	$(RUN) test_captive_magnets.py

check_obiwan_lm_split_two_pin_static: \
        test_obiwan_lm_split_two_pin_static.py \
        top_baffle_nd25fw4_obiwan_lm_split.py
	$(PYTHON_RUN) test_obiwan_lm_split_two_pin_static.py

# Seconds-scale topology gate for closure iteration.  It deliberately uses
# the same Make/remote guard authority as production but does not rebuild the
# two carrier stages until the analytic tangent/ownership plans pass.
check_obiwan_junction_closure_plans: \
		test_obiwan_junction_closure_webs.py \
		top_baffle_nd25fw4_obiwan.py \
		top_baffle_nd25fw4_obiwan_attachments.py \
		$(OBIWAN_SRCS) $(EXECUTION_SRCS)
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. \
	    LX_OBIWAN_CLOSURE_PLAN_ONLY=1 $(RUN) \
	    test_obiwan_junction_closure_webs.py

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

$(JUNCTION_CLOSURE_STAMP): $(JUNCTION_CLOSURE_BASE_STAMP) \
		$(JUNCTION_CLOSURE_DENSE_STAMPS)
	@mkdir -p $(@D)
	@touch $@

$(JUNCTION_CLOSURE_BASE_STAMP): \
		$(call r6f_check_stamp,test_floor_feed_and_flush_mouth_contract) \
		$(call r6f_check_stamp,test_feed_and_flush_mouth_contract) \
		test_obiwan_junction_closure_webs.py \
		top_baffle_nd25fw4_obiwan.py \
		top_baffle_nd25fw4_obiwan_attachments.py \
		$(OBIWAN_SRCS) $(EXECUTION_SRCS) $(CHECK_REVALIDATE) | \
		floor_stand/.obiwan_stage/manifest.json \
		no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $(@D)
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_BASE_ONLY=1 $(RUN) \
	    test_obiwan_junction_closure_webs.py
	@touch $@

define JUNCTION_CLOSURE_DENSE_RULE
$(call junction_closure_dense_stamp,$(1),$(2)): \
		$$(call r6f_check_stamp,test_floor_feed_and_flush_mouth_contract) \
		$$(call r6f_check_stamp,test_feed_and_flush_mouth_contract) \
		test_obiwan_junction_closure_webs.py \
		top_baffle_nd25fw4_obiwan.py \
		top_baffle_nd25fw4_obiwan_attachments.py \
		$$(OBIWAN_SRCS) $$(EXECUTION_SRCS) $$(CHECK_REVALIDATE) | \
		floor_stand/.obiwan_stage/manifest.json \
		no_floor_stand/.obiwan_stage/manifest.json
	@mkdir -p $$(@D)
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan PYTHONPATH=. \
	    LX_OBIWAN_CLOSURE_FULL_TEST=1 \
	    LX_OBIWAN_CLOSURE_DENSE_CASE=$(1) \
	    LX_OBIWAN_CLOSURE_DENSE_SHARD=$(2)/4 $$(RUN) \
	    test_obiwan_junction_closure_webs.py
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
check_obiwan_service: _check_r6f_test_tweeter_and_service \
                    _check_r6f_test_floor_tweeter_and_service

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

check_floor_obiwan_mouths: \
        _check_r6f_test_floor_feed_and_flush_mouth_contract

check_no_floor_obiwan_mouths: \
        _check_r6f_test_feed_and_flush_mouth_contract

check_floor_obiwan_burial: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_floor_lm_burial_web_contract \
	    $(RUN) test_obiwan_r6f.py

check_no_floor_obiwan_burial: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		no_floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_lm_burial_web_contract \
	    $(RUN) test_obiwan_r6f.py

check_floor_obiwan_um_burial: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_floor_um_burial_web_contract \
	    $(RUN) test_obiwan_r6f.py

check_no_floor_obiwan_um_burial: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		no_floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_um_burial_web_contract \
	    $(RUN) test_obiwan_r6f.py

check_floor_obiwan_backfills: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_floor_bump_backfill_contract \
	    $(RUN) test_obiwan_r6f.py

check_no_floor_obiwan_backfills: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		no_floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_bump_backfill_contract \
	    $(RUN) test_obiwan_r6f.py

check_floor_um_shell check_floor_t_shell: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_$(patsubst check_%,%,$@) \
	    $(RUN) test_obiwan_r6f.py

check_no_floor_um_shell check_no_floor_t_shell: \
		test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		no_floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_$(patsubst check_%,%,$@) \
	    $(RUN) test_obiwan_r6f.py

# Focused exact-BREP integral floor-load-path diagnostic. This target is
# intentionally single-check so mount/root regressions can be measured
# without rebuilding or rerunning the full Obi-Wan service matrix.
check_floor_integrated_mount: test_obiwan_r6f.py $(OBIWAN_SRCS) | \
		floor_stand/.obiwan_stage/manifest.json
	LX_R6F_SINGLE_CHECK=test_floor_integrated_mount $(RUN) test_obiwan_r6f.py

# Focused source-BREP plus strict-STL gate for the monolithic no-floor LM.
# It catches feature-coincidence slivers before a complete candidate export.
check_no_floor_lm_mesh: test_obiwan_r6f.py $(OBIWAN_SRCS) $(STL_EXPORT_SRCS)
	LX_R6F_SINGLE_CHECK=test_no_floor_lm_core $(RUN) test_obiwan_r6f.py
	$(MAKE) --no-print-directory no_floor_stand/stl/.stamp_obiwan

# Focused two-state gate for the optional no-extra-fastener LM subdivision.
# State-specific prerequisites keep both exact split BREP/STL checks eligible
# for separate osado jobserver slots while sharing their release-stage nodes.
check_obiwan_lm_split: check_floor_obiwan_lm_split \
                     check_no_floor_obiwan_lm_split

check_floor_obiwan_lm_split: validate_floor_obiwan_stage \
        floor_stand/stl/.stamp_obiwan \
        floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	LX_R6F_SINGLE_CHECK=test_floor_lm_keyed_split $(RUN) test_obiwan_r6f.py

check_no_floor_obiwan_lm_split: validate_no_floor_obiwan_stage \
        no_floor_stand/stl/.stamp_obiwan \
        no_floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	LX_R6F_SINGLE_CHECK=test_no_floor_lm_keyed_split $(RUN) test_obiwan_r6f.py

# Fast source-plan plus native-BREP regression for the universal LM-lower
# wing-contact profile.  Keep the two guarded workers independent so osado can
# validate the analytic union and its blind-bored no-floor realization in
# parallel before a complete carrier/wing export.
check_obiwan_lm_profile: check_obiwan_lm_profile_plan \
                       check_obiwan_lm_profile_brep

check_obiwan_lm_profile_plan: test_obiwan_r6f.py $(OBIWAN_SRCS)
	LX_R6F_SINGLE_CHECK=test_bridge_contract $(RUN) test_obiwan_r6f.py

check_obiwan_lm_profile_brep: test_obiwan_r6f.py $(OBIWAN_SRCS)
	LX_R6F_SINGLE_CHECK=test_bridge_geometry $(RUN) test_obiwan_r6f.py

# Lightweight focused gate for deterministic route dimensions and policy.
# Keep it remotely guarded because importing the final CAD modules initializes
# OCC even though this check itself performs no heavy Boolean construction.
check_route_contract: test_obiwan_r6f.py $(OBIWAN_SRCS)
	LX_R6F_SINGLE_CHECK=test_route_contract $(RUN) test_obiwan_r6f.py

# Focused exact-BREP route-to-insert clearance check for both stand states.
check_bump_brep: test_obiwan_r6f.py $(OBIWAN_SRCS)
	LX_R6F_SINGLE_CHECK=test_no_floor_route_smoothness $(RUN) test_obiwan_r6f.py
	LX_R6F_SINGLE_CHECK=test_bump_backfill_contract $(RUN) test_obiwan_r6f.py
	LX_R6F_SINGLE_CHECK=test_bump_brep_clearance $(RUN) test_obiwan_r6f.py
	LX_R6F_SINGLE_CHECK=test_floor_bump_brep_clearance $(RUN) test_obiwan_r6f.py

top_baffle_nd25fw4_attachments.step: $(SRCS) export_steps.py
	LX_ROUTING_PROFILE=proud PYTHONPATH=. $(RUN) export_steps.py top_baffle_nd25fw4_attachments.py \
	    --output $@

# $(1) = variant dir, $(2) = LX_STAND_FOOT value
define RULES
$(1)/stl/.stamp: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_c7: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant c7 \
	    --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_v0: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v0 \
	    --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_v1: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1 \
	    --outdir $(1)/stl
	@touch $$@

$(1)/stl/.stamp_v1l: $(SRCS) $(STL_EXPORT_SRCS)
	@rm -f $$@
ifeq ($(LX_CAD_EXECUTION),remote-worker)
	# Osado constructs the shared V1L geometry once, then meshes the same
	# four physical print pieces and grommet pair from that one build.
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --outdir $(1)/stl
else
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --v1l-piece piece_bottom --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --v1l-piece piece_mid_left --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --v1l-piece piece_mid_right --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --v1l-piece piece_top_b2 --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_piece_stls.py --variant v1l --v1l-piece grommet --outdir $(1)/stl
endif
	@touch $$@

$(1)/stl/.stamp_obiwan: $(OBIWAN_SRCS) $(STL_EXPORT_SRCS) \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
ifeq ($(LX_CAD_EXECUTION),remote-worker)
	# The staged BREPs are already hash-verified. Osado can import and mesh
	# the complete state without the local one-group-per-process workaround.
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) export_piece_stls.py --variant obiwan --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
else
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) export_piece_stls.py --variant obiwan --obiwan-part lm --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) export_piece_stls.py --variant obiwan --obiwan-part lm_split --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) export_piece_stls.py --variant obiwan --obiwan-part um --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) export_piece_stls.py --variant obiwan --obiwan-part tweeter --obiwan-stage-manifest $(1)/.obiwan_stage/manifest.json --outdir $(1)/stl
endif
	@touch $$@

$(1)/stl/.stamp_coupon: export_coupon.py check_manifold.py \
		$(PRINT_CONTRACT_SRCS) \
		write_obiwan_release_manifest.py $(OBIWAN_SRCS) \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_coupon.py --outdir $(1)/stl
	@touch $$@

# R6F STEP targets import only the state-local, hash-verified native BREPs.
# Direct generator-module export is fail-closed in export_steps.py.
$(1)/top_baffle_nd25fw4_obiwan_%.step: \
		$(1)/.obiwan_stage/manifest.json export_obiwan_staged.py \
		export_steps.py $(OBIWAN_SRCS)
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan PYTHONPATH=. $(RUN) \
	    export_obiwan_staged.py step \
	    --manifest $(1)/.obiwan_stage/manifest.json \
	    --kind $$* --output $$@

$(1)/top_baffle_nd25fw4_um_fit.step: top_baffle_nd25fw4_um_fit.py \
        $(SRCS) export_steps.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=. $(RUN) export_steps.py top_baffle_nd25fw4_um_fit.py --output $$@

# Osado builds V1L's shared baffle/duct tree once and emits the same four
# physical children. Explicit local mode retains one-piece-per-child staging.
$(1)/top_baffle_nd25fw4_v1l_split.step: export_v1l_staged_step.py $(SRCS)
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=. $(RUN) export_v1l_staged_step.py --output $$@

$(1)/%.step: %.py $(SRCS) export_steps.py
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=. $(RUN) export_steps.py $$*.py --output $$@

$(1)/baffle_cable_routing_proud.png: $(SRCS) gen_cable_routing.py \
        gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=.. $(RUN) ../gen_cable_routing.py

$(1)/baffle_cable_routing_obiwan.png: $(OBIWAN_SRCS) gen_cable_routing.py \
        gen_driver_overlay.py $(1)/.obiwan_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan PYTHONPATH=.. $(RUN) ../gen_cable_routing.py

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
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=obiwan $(RUN) \
	    write_obiwan_release_manifest.py --state-dir $(1)

$(1)/.stamp_overlays: $(SRCS) gen_driver_overlay.py \
        $(1)/.obiwan_stage/manifest.json
	@rm -f $$@
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=.. $(RUN) ../gen_driver_overlay.py
	@test -f $(1)/baffle_variants_drivers.png
	@test -f $(1)/baffle_b1_drivers.png
	@test -f $(1)/baffle_b2_drivers.png
	@touch $$@

$(addprefix $(1)/,$(OVERLAY_PNGS)): $(1)/.stamp_overlays
	@test -f $$@ || { rm -f $(1)/.stamp_overlays; \
	    make --no-print-directory $(1)/.stamp_overlays; test -f $$@; }
endef

$(eval $(call RULES,floor_stand,1))
$(eval $(call RULES,no_floor_stand,0))

floor_stand/.stamp_obiwan_integrated_floor_strength: \
        top_baffle_nd25fw4_obiwan_floor_strength.py \
        floor_stand/top_baffle_nd25fw4_obiwan_split.step \
        floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	@rm -f $@ $(FLOOR_STRENGTH_REPORTS)
	$(RUN) top_baffle_nd25fw4_obiwan_floor_strength.py \
	    --json floor_stand/obiwan_integrated_floor_strength.json \
	    --markdown floor_stand/obiwan_integrated_floor_strength.md \
	    --step floor_stand/top_baffle_nd25fw4_obiwan_split.step \
	    --step floor_stand/top_baffle_nd25fw4_obiwan_lm_split.step
	@touch $@

$(FLOOR_STRENGTH_REPORTS): floor_stand/.stamp_obiwan_integrated_floor_strength
	@test -f $@ || { rm -f floor_stand/.stamp_obiwan_integrated_floor_strength; \
	    make --no-print-directory floor_stand/.stamp_obiwan_integrated_floor_strength; \
	    test -f $@; }

clean:
	rm -rf floor_stand no_floor_stand wings .manifold-stamps .check-stamps \
	    __pycache__ \
	    top_baffle_nd25fw4_attachments.step

endif
