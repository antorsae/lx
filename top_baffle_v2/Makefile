# Generates every flag-dependent artifact for BOTH stand-foot states:
#
#   floor_stand/      LX_STAND_FOOT=1  fused foot + NL8 panel; V1LF carries
#                     its integral floor mount in the LM carrier
#   no_floor_stand/   LX_STAND_FOOT=0  flat piece_bottom, bridge
#                     pass-throughs; V1LF LM owns a front-flush solid web
#
# Each folder gets a complete print set (stl/), all review STEP models, and
# five documentation PNGs (three overlays + two routing profiles).
# Proud attachments.step is flag-independent and stays at the top level.
# V1LF LM core and its attachment manifest differ by stand state; the floor
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
#   make floor_v1lf          focused remote V1LF floor release + QA
#   make v1lf_release        both V1LF states + Ac/Ae, concurrent on osado
#   LX_CAD_EXECUTION=local make PYTHON=/path/venv/bin/python
#   make check               proud regression + final V1LF R6F suites
#   make candidate           checks + regenerated candidate artifacts + QA
#   make release             candidate QA + fail-closed physical-release gate
#
# Compatible with the macOS-provided GNU make 3.81.

LX_CAD_EXECUTION ?= remote
ifeq ($(filter $(LX_CAD_EXECUTION),remote local remote-worker),)
$(error LX_CAD_EXECUTION must be remote, local, or remote-worker)
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
SRCS := $(wildcard top_baffle_nd25fw4*.py) run_memory_guarded.py Makefile \
        $(MU10_REFS) $(W22_REFS)
V1LF_RELEASE_SRCS := $(SRCS) export_steps.py export_v1lf_staged.py \
        export_piece_stls.py export_coupon.py gen_cable_routing.py \
        gen_driver_overlay.py check_manifold.py test_clearances.py \
        test_v1lf_r6f.py test_remote_cad.py \
        write_v1lf_release_manifest.py V1LF_PHYSICAL_QUALIFICATION.md \
        remote_cad.py cad-remote-requirements.lock
# export_piece_stls imports the strict checker after meshing, and that checker
# imports the release-manifest helpers at module load.  Keep every STL stamp
# source-complete so a checker/contract change cannot reuse a stale mesh stamp.
STL_EXPORT_SRCS := export_piece_stls.py check_manifold.py \
        write_v1lf_release_manifest.py
# Command-line Make variables must not turn a candidate/release recipe into
# an unguarded OCC process. PYTHON remains selectable, but every invocation
# is forced through this workspace's authenticated process-tree guard.
override GUARD := $(abspath run_memory_guarded.py)
override RUN := $(PYTHON_RUN) $(GUARD) -- $(PYTHON_RUN)

# Local builds remain strictly serial. The remote executor may opt into
# several guarded slots only under the large-host profile; the enclosing
# systemd cgroup still limits their aggregate memory to 512 GiB.
ifneq ($(LX_CAD_EXECUTION),remote-worker)
.NOTPARALLEL:
endif

GENS := top_baffle_nd25fw4_b2_split top_baffle_nd25fw4_b2 \
        top_baffle_nd25fw4_b1 top_baffle_nd25fw4_a_comp \
        top_baffle_nd25fw4_b1_assembled top_baffle_nd25fw4_a_comp_assembled \
        top_baffle_nd25fw4_c7 top_baffle_nd25fw4_c7_split \
        top_baffle_nd25fw4_v0 top_baffle_nd25fw4_v0_split \
        top_baffle_nd25fw4_v1 top_baffle_nd25fw4_v1_split \
        top_baffle_nd25fw4_v1l_split top_baffle_nd25fw4_v1lf_split \
        top_baffle_nd25fw4_v1lf_lm_split \
        top_baffle_nd25fw4_v1lf_attachments \
        top_baffle_nd25fw4_v1lf_assembled \
        top_baffle_nd25fw4_um_fit \
        top_baffle_nd25fw4_v1_attachments

OVERLAY_PNGS = baffle_b1_drivers.png baffle_b2_drivers.png \
               baffle_variants_drivers.png
FLOOR_STRENGTH_REPORTS = floor_stand/v1lf_integrated_floor_strength.json \
                         floor_stand/v1lf_integrated_floor_strength.md
WING_CONCEPT_SLUGS = solid_lx solid_mid perforated_4k graded_1k_4k \
                     open_honeycomb tortuous_aperiodic \
                     vertical_aperture_gradient vertical_path_gradient \
                     bimodal_apertures radial_edge_loaded \
                     radial_edge_released solid_chirped_edge \
                     perimeter_frame solid_radial_leak_slots
WING_CONCEPT_PNGS = $(addprefix review/v1lf_wing_,\
                    $(addsuffix _concept.png,$(WING_CONCEPT_SLUGS))) \
                    review/v1lf_wing_concepts_index.png
V1LF_BASIC_VARIANTS_PNG = baffle_variants_drivers_v1lf.png
V1LF_BASIC_WING_SLUGS = ac ae
V1LF_BASIC_WING_STAMPS = $(addprefix wings/.stamp_,$(V1LF_BASIC_WING_SLUGS))
V1LF_BASIC_WING_INPUTS = cad-remote-requirements.lock \
        V1LF_ACOUSTIC_WINGS_SPEC.md gen_driver_overlay.py \
        export_v1lf_staged.py \
        top_baffle_nd25fw4.py top_baffle_nd25fw4_b.py \
        top_baffle_nd25fw4_b1.py top_baffle_nd25fw4_b2.py \
        top_baffle_nd25fw4_cables.py top_baffle_nd25fw4_flush.py \
        top_baffle_nd25fw4_v1.py top_baffle_nd25fw4_v1lf.py \
        top_baffle_nd25fw4_v1lf_attachments.py \
        top_baffle_nd25fw4_v1lf_route.py top_baffle_nd25fw4_a_comp.py \
        top_baffle_nd25fw4_v1lf_bridge.py \
        top_baffle_nd25fw4_v1lf_floor.py \
        top_baffle_nd25fw4_v1lf_floor_strength.py \
        top_baffle_nd25fw4_v1lf_lm_split.py \
        floor_stand/.v1lf_stage/manifest.json \
        no_floor_stand/.v1lf_stage/manifest.json

VARIANT_TARGETS = $(1)/stl/.stamp_coupon $(1)/stl/.stamp $(1)/stl/.stamp_c7 $(1)/stl/.stamp_v0 $(1)/stl/.stamp_v1 $(1)/stl/.stamp_v1l $(1)/stl/.stamp_v1lf \
                  $(addprefix $(1)/,$(addsuffix .step,$(GENS))) \
                  $(1)/baffle_cable_routing_proud.png \
                  $(1)/baffle_cable_routing_v1lf.png \
                  $(addprefix $(1)/,$(OVERLAY_PNGS)) \
                  $(1)/v1lf_release_manifest.json

.PHONY: all candidate release validate_v1lf_stages \
        validate_floor_v1lf_stage validate_no_floor_v1lf_stage \
        floor_stand floor_v1lf no_floor_stand no_floor_v1lf v1lf_release \
        v1lf_release wing_concepts \
        v1lf_basic_variants v1lf_basic_wings check_v1lf_basic_variants \
        common check check_v1lf \
        check_v1lf_shells check_v1lf_t_shells \
        check_v1lf_mouths check_v1lf_burial check_v1lf_backfills \
        check_v1lf_um_burial \
        check_v1lf_route_boundaries \
        check_floor_um_shell check_floor_t_shell \
        check_no_floor_um_shell check_no_floor_t_shell \
        check_floor_v1lf_mouths check_no_floor_v1lf_mouths \
        check_floor_v1lf_burial check_no_floor_v1lf_burial \
        check_floor_v1lf_um_burial check_no_floor_v1lf_um_burial \
        check_floor_v1lf_backfills check_no_floor_v1lf_backfills \
        check_route_contract check_bump_brep check_floor_integrated_mount \
        check_no_floor_lm_mesh check_v1lf_lm_split \
        check_floor_v1lf_lm_split check_no_floor_v1lf_lm_split \
        check_v1lf_lm_profile check_v1lf_lm_profile_plan \
        check_v1lf_lm_profile_brep manifold clean
all: validate_v1lf_stages floor_stand no_floor_stand common \
        v1lf_basic_variants
	$(RUN) check_manifold.py

# Runtime package versions are part of the native-stage fingerprint but not
# filesystem prerequisites. Validate first on every build; the stage command
# is a no-write fast path when current and atomically rebuilds only on drift.
validate_v1lf_stages: validate_floor_v1lf_stage validate_no_floor_v1lf_stage

validate_floor_v1lf_stage: export_v1lf_staged.py $(SRCS)
	@mkdir -p floor_stand/.v1lf_stage
	LX_STAND_FOOT=1 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    export_v1lf_staged.py stage \
	    --manifest floor_stand/.v1lf_stage/manifest.json

validate_no_floor_v1lf_stage: export_v1lf_staged.py $(SRCS)
	@mkdir -p no_floor_stand/.v1lf_stage
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    export_v1lf_staged.py stage \
	    --manifest no_floor_stand/.v1lf_stage/manifest.json

# The manifest files are aliases for the always-validated state transactions.
# This gives every downstream artifact one shared Make node and prevents a
# parallel build from racing the phony validator against a second file recipe.
floor_stand/.v1lf_stage/manifest.json: validate_floor_v1lf_stage
	@test -f $@
no_floor_stand/.v1lf_stage/manifest.json: validate_no_floor_v1lf_stage
	@test -f $@

candidate: check
	# The candidate gate is a barrier: regression checks must finish before
	# artifact writers start, while the recursive all build may still use the
	# bounded remote jobserver slots internally.
	$(MAKE) --no-print-directory all

release: candidate
	$(RUN) check_manifold.py --require-release-authorized

manifold: check_manifold.py
	$(RUN) check_manifold.py

# polar-measurement turntable (mates under the floor-stand foot only)
floor_stand: validate_floor_v1lf_stage floor_stand/stl/.stamp_polar
floor_stand/stl/.stamp_polar: polar_index_base.py
	@rm -f $@
	$(RUN) polar_index_base.py
	@touch $@
floor_stand: $(call VARIANT_TARGETS,floor_stand) $(FLOOR_STRENGTH_REPORTS)
	$(RUN) check_manifold.py floor_stand/stl
# Focused, atomic V1LF floor release.  This avoids rebuilding unrelated B/C/
# V0/V1/V1L pieces while still producing every V1LF STEP/STL/PNG, the coupon
# set, overlays, hash-bound strength report, release manifest and strict QA.
floor_v1lf: validate_floor_v1lf_stage \
        floor_stand/v1lf_release_manifest.json
	$(RUN) check_manifold.py --v1lf-only floor_stand/stl

# Symmetric focused release for the no-floor carrier.  Keep this narrower than
# no_floor_stand: it emits the complete hash-backed V1LF family without
# rebuilding unrelated proud/C/V0/V1/V1L print sets.
no_floor_v1lf: validate_no_floor_v1lf_stage \
        no_floor_stand/v1lf_release_manifest.json
	$(RUN) check_manifold.py --v1lf-only no_floor_stand/stl

# One public remote target owns the complete two-state V1LF release, Ac/Ae
# magnetic attachments and the focused final-BREP interface checks.  A single
# Make DAG lets osado share both native stages and fan independent state,
# wing and validation recipes across its bounded jobserver slots.
v1lf_release: floor_v1lf no_floor_v1lf v1lf_basic_variants \
        check_route_contract check_v1lf_lm_split check_v1lf_shells \
        check_v1lf_route_boundaries check_floor_integrated_mount

# Review-only W1-W14 specimen plates.  The generator is deliberately pure 2D:
# it does not import build123d/OCC or modify released carrier geometry.
wing_concepts: $(WING_CONCEPT_PNGS)

review/.stamp_v1lf_wing_concepts: gen_v1lf_wing_concepts.py \
        V1LF_ACOUSTIC_WINGS_SPEC.md
	@rm -f $@ $(WING_CONCEPT_PNGS)
	@mkdir -p review
	$(RUN) gen_v1lf_wing_concepts.py --output-dir review
	@for png in $(WING_CONCEPT_PNGS); do test -s $$png || exit 1; done
	@touch $@

$(WING_CONCEPT_PNGS): review/.stamp_v1lf_wing_concepts
	@test -s $@ || { rm -f review/.stamp_v1lf_wing_concepts; \
	    $(MAKE) review/.stamp_v1lf_wing_concepts; test -s $@; }

# Source-linked, dimensioned A/B1/B2 attachment layout for the V1LF carrier.
# Ac and Ae are full STEP-first implementations of the approved A plan.  Their
# independent stamps let osado build both slug trees concurrently; each tree is
# one atomic, hash-backed transaction containing STEP, STL, facts and review
# images.  The PNG remains the shared analytic contract and is regenerated
# before either solid model.
v1lf_basic_variants: $(V1LF_BASIC_VARIANTS_PNG) v1lf_basic_wings

v1lf_basic_wings: $(V1LF_BASIC_WING_STAMPS) check_v1lf_basic_variants

wings/.stamp_ac: export_v1lf_basic_wings.py v1lf_basic_wings_cad.py \
        gen_v1lf_basic_variants.py $(V1LF_BASIC_VARIANTS_PNG) \
        check_manifold.py $(V1LF_BASIC_WING_INPUTS)
	@rm -f $@
	@mkdir -p wings
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    export_v1lf_basic_wings.py --slug ac --output-root wings
	@test -s wings/ac/top_baffle_nd25fw4_v1lf_wing_ac.step
	@test -s wings/ac/top_baffle_nd25fw4_v1lf_wing_ac_assembled.step
	@touch $@

wings/.stamp_ae: export_v1lf_basic_wings.py v1lf_basic_wings_cad.py \
        gen_v1lf_basic_variants.py $(V1LF_BASIC_VARIANTS_PNG) \
        check_manifold.py $(V1LF_BASIC_WING_INPUTS)
	@rm -f $@
	@mkdir -p wings
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    export_v1lf_basic_wings.py --slug ae --output-root wings
	@test -s wings/ae/top_baffle_nd25fw4_v1lf_wing_ae.step
	@test -s wings/ae/top_baffle_nd25fw4_v1lf_wing_ae_assembled.step
	@touch $@

check_v1lf_basic_variants: $(V1LF_BASIC_WING_STAMPS) \
        test_v1lf_basic_wings.py
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    test_v1lf_basic_wings.py --artifact-root wings
	$(RUN) check_manifold.py wings/ac/stl wings/ae/stl

$(V1LF_BASIC_VARIANTS_PNG): gen_v1lf_basic_variants.py \
        gen_driver_overlay.py top_baffle_nd25fw4_a_comp.py \
        top_baffle_nd25fw4_b.py top_baffle_nd25fw4_b1.py \
        top_baffle_nd25fw4_b2.py top_baffle_nd25fw4_v1lf.py \
        top_baffle_nd25fw4_v1lf_bridge.py \
        top_baffle_nd25fw4_v1lf_floor.py
	@rm -f $@
	LX_STAND_FOOT=0 LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    gen_v1lf_basic_variants.py --output $@
	@test -s $@

no_floor_stand: validate_no_floor_v1lf_stage \
        $(call VARIANT_TARGETS,no_floor_stand)
	$(RUN) check_manifold.py no_floor_stand/stl
common: top_baffle_nd25fw4_attachments.step

check: test_clearances.py test_v1lf_r6f.py test_remote_cad.py remote_cad.py \
        $(SRCS)
	$(PYTHON_RUN) test_remote_cad.py
	# Keep the lightweight clearance orchestrator outside OCC so every heavy
	# check establishes a fresh authenticated guard. Clear the private selector
	# so inherited shell state cannot silently reduce the release suite.
	env -u LX_CLEARANCE_SINGLE_CHECK $(PYTHON_RUN) test_clearances.py
	# The R6F parent is a lightweight stdlib/numpy orchestrator. It gives each
	# OCC-heavy check its own authenticated guard; wrapping the parent here
	# would collapse the long service matrix back into one accumulating group.
	env -u LX_R6F_SINGLE_CHECK -u LX_R6F_SERVICE_START_PHASE \
	    -u LX_R6F_SERVICE_STOP_AFTER_PHASE $(PYTHON_RUN) test_v1lf_r6f.py

# Focused remote iteration gate. The parent still fans out its OCC-heavy
# checks across the configured remote slots; this avoids rerunning unrelated
# proud/V1L clearance checks while refining V1LF final-BREP contracts.
check_v1lf: test_v1lf_r6f.py $(SRCS)
	env -u LX_R6F_SINGLE_CHECK -u LX_R6F_SERVICE_START_PHASE \
	    -u LX_R6F_SERVICE_STOP_AFTER_PHASE $(PYTHON_RUN) test_v1lf_r6f.py

# Focused final-BREP shell gate.  The four independent state/route checks are
# separate prerequisites so the large remote host can run them concurrently.
check_v1lf_shells: check_floor_um_shell check_floor_t_shell \
                   check_no_floor_um_shell check_no_floor_t_shell

check_v1lf_t_shells: check_floor_t_shell check_no_floor_t_shell

# Focused native-mouth/burial gates.  These retain the same isolated guarded
# workers as the release suite while allowing the large remote host to check
# both stand states concurrently during route-boundary iteration.
check_v1lf_mouths: check_floor_v1lf_mouths check_no_floor_v1lf_mouths

check_v1lf_burial: check_floor_v1lf_burial check_no_floor_v1lf_burial \
                    check_v1lf_um_burial

check_v1lf_um_burial: check_floor_v1lf_um_burial \
                       check_no_floor_v1lf_um_burial

check_v1lf_backfills: check_floor_v1lf_backfills \
                      check_no_floor_v1lf_backfills

check_v1lf_route_boundaries: check_v1lf_mouths check_v1lf_burial \
                             check_v1lf_backfills

check_floor_v1lf_mouths: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_floor_feed_and_flush_mouth_contract \
	    $(RUN) test_v1lf_r6f.py

check_no_floor_v1lf_mouths: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_feed_and_flush_mouth_contract \
	    $(RUN) test_v1lf_r6f.py

check_floor_v1lf_burial: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_floor_lm_burial_web_contract \
	    $(RUN) test_v1lf_r6f.py

check_no_floor_v1lf_burial: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_lm_burial_web_contract \
	    $(RUN) test_v1lf_r6f.py

check_floor_v1lf_um_burial: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_floor_um_burial_web_contract \
	    $(RUN) test_v1lf_r6f.py

check_no_floor_v1lf_um_burial: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_um_burial_web_contract \
	    $(RUN) test_v1lf_r6f.py

check_floor_v1lf_backfills: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_floor_bump_backfill_contract \
	    $(RUN) test_v1lf_r6f.py

check_no_floor_v1lf_backfills: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_bump_backfill_contract \
	    $(RUN) test_v1lf_r6f.py

check_floor_um_shell check_floor_t_shell \
check_no_floor_um_shell check_no_floor_t_shell: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_$(patsubst check_%,%,$@) \
	    $(RUN) test_v1lf_r6f.py

# Focused exact-BREP integral floor-load-path diagnostic. This target is
# intentionally single-check so mount/root regressions can be measured
# without rebuilding or rerunning the full V1LF service matrix.
check_floor_integrated_mount: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_floor_integrated_mount $(RUN) test_v1lf_r6f.py

# Focused source-BREP plus strict-STL gate for the monolithic no-floor LM.
# It catches feature-coincidence slivers before a complete candidate export.
check_no_floor_lm_mesh: test_v1lf_r6f.py $(SRCS) $(STL_EXPORT_SRCS)
	LX_R6F_SINGLE_CHECK=test_no_floor_lm_core $(RUN) test_v1lf_r6f.py
	$(MAKE) --no-print-directory no_floor_stand/stl/.stamp_v1lf

# Focused two-state gate for the optional no-extra-fastener LM subdivision.
# State-specific prerequisites keep both exact split BREP/STL checks eligible
# for separate osado jobserver slots while sharing their release-stage nodes.
check_v1lf_lm_split: check_floor_v1lf_lm_split \
                     check_no_floor_v1lf_lm_split

check_floor_v1lf_lm_split: validate_floor_v1lf_stage \
        floor_stand/stl/.stamp_v1lf \
        floor_stand/top_baffle_nd25fw4_v1lf_lm_split.step
	LX_R6F_SINGLE_CHECK=test_floor_lm_keyed_split $(RUN) test_v1lf_r6f.py

check_no_floor_v1lf_lm_split: validate_no_floor_v1lf_stage \
        no_floor_stand/stl/.stamp_v1lf \
        no_floor_stand/top_baffle_nd25fw4_v1lf_lm_split.step
	LX_R6F_SINGLE_CHECK=test_no_floor_lm_keyed_split $(RUN) test_v1lf_r6f.py

# Fast source-plan plus native-BREP regression for the universal LM-lower
# wing-contact profile.  Keep the two guarded workers independent so osado can
# validate the analytic union and its blind-bored no-floor realization in
# parallel before a complete carrier/wing export.
check_v1lf_lm_profile: check_v1lf_lm_profile_plan \
                       check_v1lf_lm_profile_brep

check_v1lf_lm_profile_plan: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_bridge_contract $(RUN) test_v1lf_r6f.py

check_v1lf_lm_profile_brep: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_bridge_geometry $(RUN) test_v1lf_r6f.py

# Lightweight focused gate for deterministic route dimensions and policy.
# Keep it remotely guarded because importing the final CAD modules initializes
# OCC even though this check itself performs no heavy Boolean construction.
check_route_contract: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_route_contract $(RUN) test_v1lf_r6f.py

# Focused exact-BREP route-to-insert clearance check for both stand states.
check_bump_brep: test_v1lf_r6f.py $(SRCS)
	LX_R6F_SINGLE_CHECK=test_no_floor_route_smoothness $(RUN) test_v1lf_r6f.py
	LX_R6F_SINGLE_CHECK=test_bump_backfill_contract $(RUN) test_v1lf_r6f.py
	LX_R6F_SINGLE_CHECK=test_bump_brep_clearance $(RUN) test_v1lf_r6f.py
	LX_R6F_SINGLE_CHECK=test_floor_bump_brep_clearance $(RUN) test_v1lf_r6f.py

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

$(1)/stl/.stamp_v1lf: $(SRCS) $(STL_EXPORT_SRCS) \
        $(1)/.v1lf_stage/manifest.json
	@rm -f $$@
ifeq ($(LX_CAD_EXECUTION),remote-worker)
	# The staged BREPs are already hash-verified. Osado can import and mesh
	# the complete state without the local one-group-per-process workaround.
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) export_piece_stls.py --variant v1lf --v1lf-stage-manifest $(1)/.v1lf_stage/manifest.json --outdir $(1)/stl
else
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) export_piece_stls.py --variant v1lf --v1lf-part lm --v1lf-stage-manifest $(1)/.v1lf_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) export_piece_stls.py --variant v1lf --v1lf-part lm_split --v1lf-stage-manifest $(1)/.v1lf_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) export_piece_stls.py --variant v1lf --v1lf-part um --v1lf-stage-manifest $(1)/.v1lf_stage/manifest.json --outdir $(1)/stl
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) export_piece_stls.py --variant v1lf --v1lf-part tweeter --v1lf-stage-manifest $(1)/.v1lf_stage/manifest.json --outdir $(1)/stl
endif
	@touch $$@

$(1)/stl/.stamp_coupon: export_coupon.py $(SRCS) \
        $(1)/.v1lf_stage/manifest.json
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud $(RUN) export_coupon.py --outdir $(1)/stl
	@touch $$@

# R6F STEP targets import only the state-local, hash-verified native BREPs.
# Direct generator-module export is fail-closed in export_steps.py.
$(1)/top_baffle_nd25fw4_v1lf_%.step: \
        $(1)/.v1lf_stage/manifest.json export_v1lf_staged.py \
        export_steps.py $(SRCS)
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf PYTHONPATH=. $(RUN) \
	    export_v1lf_staged.py step \
	    --manifest $(1)/.v1lf_stage/manifest.json \
	    --kind $$* --output $$@

$(1)/top_baffle_nd25fw4_um_fit.step: top_baffle_nd25fw4_um_fit.py \
        $(SRCS) export_steps.py $(1)/.v1lf_stage/manifest.json
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
        gen_driver_overlay.py $(1)/.v1lf_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=.. $(RUN) ../gen_cable_routing.py

$(1)/baffle_cable_routing_v1lf.png: $(SRCS) gen_cable_routing.py \
        gen_driver_overlay.py $(1)/.v1lf_stage/manifest.json
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf PYTHONPATH=.. $(RUN) ../gen_cable_routing.py

$(1)/v1lf_release_manifest.json: $(V1LF_RELEASE_SRCS) \
        $(1)/stl/.stamp_v1lf $(1)/stl/.stamp_coupon \
        $(1)/top_baffle_nd25fw4_v1lf_split.step \
        $(1)/top_baffle_nd25fw4_v1lf_lm_split.step \
        $(1)/top_baffle_nd25fw4_v1lf_attachments.step \
        $(1)/top_baffle_nd25fw4_v1lf_assembled.step \
        $(1)/top_baffle_nd25fw4_um_fit.step \
        $(1)/baffle_cable_routing_proud.png \
        $(1)/baffle_cable_routing_v1lf.png \
        $(if $(filter 1,$(2)),$(FLOOR_STRENGTH_REPORTS)) \
        $(addprefix $(1)/,$(OVERLAY_PNGS))
	@rm -f $$@
	LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=v1lf $(RUN) \
	    write_v1lf_release_manifest.py --state-dir $(1)

$(1)/.stamp_overlays: $(SRCS) gen_driver_overlay.py \
        $(1)/.v1lf_stage/manifest.json
	@rm -f $$@
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=proud PYTHONPATH=.. $(RUN) ../gen_driver_overlay.py
	@test -f $(1)/baffle_variants_drivers.png
	@test -f $(1)/baffle_b1_drivers.png
	@test -f $(1)/baffle_b2_drivers.png
	@touch $$@

$(addprefix $(1)/,$(OVERLAY_PNGS)): $(1)/.stamp_overlays
	@test -f $$@ || { rm -f $(1)/.stamp_overlays; \
	    $(MAKE) $(1)/.stamp_overlays; test -f $$@; }
endef

$(eval $(call RULES,floor_stand,1))
$(eval $(call RULES,no_floor_stand,0))

floor_stand/.stamp_v1lf_integrated_floor_strength: \
        top_baffle_nd25fw4_v1lf_floor_strength.py \
        floor_stand/top_baffle_nd25fw4_v1lf_split.step \
        floor_stand/top_baffle_nd25fw4_v1lf_lm_split.step
	@rm -f $@ $(FLOOR_STRENGTH_REPORTS)
	$(RUN) top_baffle_nd25fw4_v1lf_floor_strength.py \
	    --json floor_stand/v1lf_integrated_floor_strength.json \
	    --markdown floor_stand/v1lf_integrated_floor_strength.md \
	    --step floor_stand/top_baffle_nd25fw4_v1lf_split.step \
	    --step floor_stand/top_baffle_nd25fw4_v1lf_lm_split.step
	@touch $@

$(FLOOR_STRENGTH_REPORTS): floor_stand/.stamp_v1lf_integrated_floor_strength
	@test -f $@ || { rm -f floor_stand/.stamp_v1lf_integrated_floor_strength; \
	    $(MAKE) floor_stand/.stamp_v1lf_integrated_floor_strength; test -f $@; }

clean:
	rm -rf floor_stand no_floor_stand wings __pycache__ \
	    top_baffle_nd25fw4_attachments.step

endif
