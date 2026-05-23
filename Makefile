# LX521 Polar Analysis - Build Automation
# ========================================
# Full pipeline: .mdat → REW API → HDF5 → viz → docs/ → GitHub Pages
#
# Usage:
#   make all        - Full rebuild (data + viz + sync)
#   make viz        - Regenerate visualizations only (uses existing HDF5)
#   make sync       - Sync output/ to docs/
#   make deploy     - sync + commit + push to GitHub
#   make help       - Show all targets

.PHONY: all data viz sync deploy clean clean-all help rew-check rew-wait \
        data-% data-run-% viz-% sync-% docs-pages commit push

# Allow pattern-specific prerequisites using $$* etc.
.SECONDEXPANSION:

# Configuration
PYTHON := .venv/bin/python
REW_API := http://127.0.0.1:4735
REW_READY_ENDPOINT := $(REW_API)/measurements
REW_APP := /Applications/REW/REW.app
JOBS ?= 8
ifneq ($(filter -j%,$(MAKEFLAGS)),)
  # jobserver already configured
else
  MAKEFLAGS += -j$(JOBS)
endif

# Measurement sets
SETS := andres juan-baffleless juan-lx521-top-raw lx521-system

# Directories
OUTPUT_DIR := output
DOCS_DIR := docs
DATA_DIR := $(OUTPUT_DIR)/data

# Helper: convert set name to HDF5 slug (hyphens → underscores)
slug = $(subst -,_,$(1))
hdf5_for = $(DATA_DIR)/polar_data_$(call slug,$(1)).h5

# ====================
# Main Targets
# ====================

all:
	@$(MAKE) data
	@$(MAKE) viz
	@$(MAKE) sync
	@echo "✓ Full pipeline complete"

data: rew-check $(addprefix data-run-,$(SETS))
	@echo "✓ All HDF5 data files generated"

viz: $(addprefix viz-,$(SETS))
	@echo "✓ All visualizations generated"

sync: $(addprefix sync-,$(SETS))
	@$(MAKE) docs-pages
	@echo "✓ All docs synced"

deploy:
	@$(MAKE) sync
	@$(MAKE) commit
	@$(MAKE) push
	@echo "✓ Deployed to GitHub Pages"

.NOTPARALLEL: data data-run-%

# ====================
# REW API Management
# ====================

rew-check:
	@echo "Checking REW API..."
	@curl -s --max-time 2 $(REW_READY_ENDPOINT) > /dev/null 2>&1 && \
		echo "✓ REW API is running" || \
		(echo "⚠ REW API not available. Starting REW..." && \
		 open -a "$(REW_APP)" --args -api && \
		 $(MAKE) rew-wait)

rew-wait:
	@echo "Waiting for REW API to become available..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
		sleep 2; \
		if curl -s --max-time 2 $(REW_READY_ENDPOINT) > /dev/null 2>&1; then \
			echo "✓ REW API ready"; \
			exit 0; \
		fi; \
		echo "  Waiting... ($$i/12)"; \
	done; \
	echo "✗ REW API failed to start after 24 seconds"; \
	exit 1

# ====================
# Data Loading (REW API → HDF5)
# ====================

data-%: rew-check data-run-%
	@echo "✓ $* data saved to $(call hdf5_for,$*)"

data-run-%:
	@echo "Loading $* measurements..."
	@SET_NAME="$*" HDF5_PATH="$(call hdf5_for,$*)" PYTHONPATH="$(CURDIR)" $(PYTHON) scripts/check_hdf5.py

# ====================
# Visualization Generation (HDF5 → plots)
# ====================

viz-%: $$(call hdf5_for,$$*)
	@echo "Generating $* visualizations..."
	$(PYTHON) run_pipeline.py -m $* --skip-loading
	@echo "✓ $* visualizations complete"

# ====================
# Sync to docs/ (output/ → docs/)
# ====================

sync-%:
	@echo "Syncing $* to docs/..."
	@mkdir -p $(DOCS_DIR)/$*/static_plots $(DOCS_DIR)/$*/interactive
	rsync -av --delete $(OUTPUT_DIR)/$*/static_plots/ $(DOCS_DIR)/$*/static_plots/
	rsync -av --delete $(OUTPUT_DIR)/$*/interactive/ $(DOCS_DIR)/$*/interactive/
	@echo "✓ $* synced to docs/"

docs-pages:
	@echo "Generating docs landing pages..."
	$(PYTHON) generate_docs_pages.py
	@echo "✓ Docs pages generated"

# ====================
# Git Operations
# ====================

commit:
	@echo "Committing docs/ changes..."
	@git add $(DOCS_DIR)/ && \
		git diff --cached --quiet && echo "No changes to commit" || \
		git commit -m "Update visualizations" -m "🤖 Generated with [Claude Code](https://claude.com/claude-code)" -m "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
	@echo "✓ Changes committed"

push:
	@echo "Pushing to GitHub..."
	git push origin main
	@echo "✓ Pushed to GitHub"

# ====================
# Utilities
# ====================

clean:
	@echo "Cleaning output directory..."
	rm -rf $(OUTPUT_DIR)/andres $(OUTPUT_DIR)/juan-baffleless $(OUTPUT_DIR)/juan-lx521-top-raw $(OUTPUT_DIR)/lx521-system
	@echo "✓ Output directories cleaned (HDF5 data preserved)"

clean-all: clean
	@echo "Cleaning all output including HDF5..."
	rm -rf $(DATA_DIR)/*.h5
	@echo "✓ All output cleaned"

help:
	@echo "LX521 Polar Analysis - Build Targets"
	@echo "====================================="
	@echo ""
	@echo "Main Targets:"
	@echo "  all              Full pipeline: data → viz → sync"
	@echo "  data             Load all .mdat files via REW API → HDF5"
	@echo "  viz              Generate all visualizations from HDF5"
	@echo "  sync             Sync output/ to docs/ + regen pages"
	@echo "  deploy           sync + commit + push to GitHub Pages"
	@echo ""
	@echo "Per-Set Targets:"
	@echo "  data-andres      Load andres measurements"
	@echo "  data-juan-baffleless  Load juan-baffleless measurements"
	@echo "  data-juan-lx521-top-raw  Load juan-lx521-top-raw measurements"
	@echo "  data-lx521-system     Load lx521-system measurements"
	@echo "  viz-andres       Generate andres visualizations"
	@echo "  viz-juan-baffleless   Generate juan-baffleless visualizations"
	@echo "  viz-juan-lx521-top-raw   Generate juan-lx521-top-raw visualizations"
	@echo "  viz-lx521-system      Generate lx521-system visualizations"
	@echo "  sync-andres      Sync andres to docs/"
	@echo "  sync-juan-baffleless  Sync juan-baffleless to docs/"
	@echo "  sync-juan-lx521-top-raw  Sync juan-lx521-top-raw to docs/"
	@echo "  sync-lx521-system     Sync lx521-system to docs/"
	@echo ""
	@echo "Git Operations:"
	@echo "  commit           Commit docs/ changes"
	@echo "  push             Push to origin/main"
	@echo ""
	@echo "Utilities:"
	@echo "  rew-check        Check/launch REW with API"
	@echo "  clean            Remove viz output (preserve HDF5)"
	@echo "  clean-all        Remove all output including HDF5"
	@echo "  help             Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make viz sync    Regenerate viz and sync to docs/"
	@echo "  make viz-andres  Regenerate only andres visualizations"
	@echo "  make deploy      Full deployment to GitHub Pages"
