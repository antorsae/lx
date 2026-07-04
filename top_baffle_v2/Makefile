# Generates every flag-dependent artifact for BOTH stand-foot states:
#
#   floor_stand/      LX_STAND_FOOT=1  fused foot + NL8 panel, no bridge
#                     holes, cables routed through the foot
#   no_floor_stand/   LX_STAND_FOOT=0  flat piece_bottom, bridge
#                     pass-throughs, rear-face cable breakouts
#
# Each folder gets a complete print set (stl/), the six STEP models, and
# the three documentation PNGs. attachments.step is flag-independent and
# stays at the top level. Only piece_bottom's STL actually differs
# between the two stl/ sets.
#
# Self-contained: STLs/PNGs use build123d/shapely/matplotlib/numpy and
# STEPs use build123d's native exporter via export_steps.py -- no
# external CAD-skill dependency. Every job writes only its own target,
# so all of them run fully parallel.
#
# Usage:
#   make -j8                 everything, parallel
#   make -j8 floor_stand     one variant
#   make PYTHON=/path/venv/bin/python -j8

PYTHON ?= python3
SRCS := $(wildcard top_baffle_nd25fw4*.py)

GENS := top_baffle_nd25fw4_b2_split top_baffle_nd25fw4_b2 \
        top_baffle_nd25fw4_b1 top_baffle_nd25fw4_a_comp \
        top_baffle_nd25fw4_b1_assembled top_baffle_nd25fw4_a_comp_assembled

VARIANT_TARGETS = $(1)/stl/.stamp $(addprefix $(1)/,$(addsuffix .step,$(GENS))) \
                  $(1)/baffle_cable_routing.png $(1)/baffle_b1_drivers.png

.PHONY: all floor_stand no_floor_stand common clean
all: floor_stand no_floor_stand common
floor_stand: $(call VARIANT_TARGETS,floor_stand)
no_floor_stand: $(call VARIANT_TARGETS,no_floor_stand)
common: top_baffle_nd25fw4_attachments.step

top_baffle_nd25fw4_attachments.step: $(SRCS) export_steps.py
	PYTHONPATH=. $(PYTHON) export_steps.py top_baffle_nd25fw4_attachments.py \
	    --output $@

# $(1) = variant dir, $(2) = LX_STAND_FOOT value
define RULES
$(1)/stl/.stamp: $(SRCS) export_piece_stls.py
	LX_STAND_FOOT=$(2) $(PYTHON) export_piece_stls.py --outdir $(1)/stl
	@touch $$@

$(1)/%.step: %.py $(SRCS) export_steps.py
	@mkdir -p $(1)
	LX_STAND_FOOT=$(2) PYTHONPATH=. $(PYTHON) export_steps.py $$*.py --output $$@

$(1)/baffle_cable_routing.png: $(SRCS) gen_cable_routing.py gen_driver_overlay.py
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) PYTHONPATH=.. $(PYTHON) ../gen_cable_routing.py

$(1)/baffle_b1_drivers.png: $(SRCS) gen_driver_overlay.py
	@mkdir -p $(1)
	cd $(1) && LX_STAND_FOOT=$(2) PYTHONPATH=.. $(PYTHON) ../gen_driver_overlay.py
endef

$(eval $(call RULES,floor_stand,1))
$(eval $(call RULES,no_floor_stand,0))

clean:
	rm -rf floor_stand no_floor_stand
