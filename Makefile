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

.PHONY: all data viz sync deploy clean help rew-check rew-wait \
        data-andres data-juan-baffleless data-lx521-system \
        viz-andres viz-juan-baffleless viz-lx521-system \
        sync-andres sync-juan-baffleless sync-lx521-system \
        commit push

# Configuration
PYTHON := .venv/bin/python
REW_API := http://127.0.0.1:4735
REW_APP := /Applications/REW/REW.app

# Measurement sets
SETS := andres juan-baffleless lx521-system

# Directories
OUTPUT_DIR := output
DOCS_DIR := docs
DATA_DIR := $(OUTPUT_DIR)/data

# HDF5 files (intermediates)
HDF5_ANDRES := $(DATA_DIR)/polar_data_andres.h5
HDF5_JUAN := $(DATA_DIR)/polar_data_juan_baffleless.h5
HDF5_LX521 := $(DATA_DIR)/polar_data_lx521_system.h5

# ====================
# Main Targets
# ====================

all: data viz sync
	@echo "✓ Full pipeline complete"

data: data-andres data-juan-baffleless data-lx521-system
	@echo "✓ All HDF5 data files generated"

viz: viz-andres viz-juan-baffleless viz-lx521-system
	@echo "✓ All visualizations generated"

sync: sync-andres sync-juan-baffleless sync-lx521-system
	@echo "✓ All docs synced"

deploy: sync commit push
	@echo "✓ Deployed to GitHub Pages"

# ====================
# REW API Management
# ====================

rew-check:
	@echo "Checking REW API..."
	@curl -s --max-time 2 $(REW_API)/application/version > /dev/null 2>&1 && \
		echo "✓ REW API is running" || \
		(echo "⚠ REW API not available. Starting REW..." && \
		 open -a "$(REW_APP)" --args -api && \
		 $(MAKE) rew-wait)

rew-wait:
	@echo "Waiting for REW API to become available..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
		sleep 2; \
		if curl -s --max-time 2 $(REW_API)/application/version > /dev/null 2>&1; then \
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

data-andres: rew-check
	@echo "Loading andres measurements..."
	$(PYTHON) run_pipeline.py -m andres --skip-viz
	@echo "✓ andres data saved to $(HDF5_ANDRES)"

data-juan-baffleless: rew-check
	@echo "Loading juan-baffleless measurements..."
	$(PYTHON) run_pipeline.py -m juan-baffleless --skip-viz
	@echo "✓ juan-baffleless data saved to $(HDF5_JUAN)"

data-lx521-system: rew-check
	@echo "Loading lx521-system measurements..."
	$(PYTHON) run_pipeline.py -m lx521-system --skip-viz
	@echo "✓ lx521-system data saved to $(HDF5_LX521)"

# ====================
# Visualization Generation (HDF5 → plots)
# ====================

viz-andres: $(HDF5_ANDRES)
	@echo "Generating andres visualizations..."
	$(PYTHON) run_pipeline.py -m andres --skip-loading
	@echo "✓ andres visualizations complete"

viz-juan-baffleless: $(HDF5_JUAN)
	@echo "Generating juan-baffleless visualizations..."
	$(PYTHON) run_pipeline.py -m juan-baffleless --skip-loading
	@echo "✓ juan-baffleless visualizations complete"

viz-lx521-system: $(HDF5_LX521)
	@echo "Generating lx521-system visualizations..."
	$(PYTHON) run_pipeline.py -m lx521-system --skip-loading
	@echo "✓ lx521-system visualizations complete"

# ====================
# Sync to docs/ (output/ → docs/)
# ====================

sync-andres:
	@echo "Syncing andres to docs/..."
	@mkdir -p $(DOCS_DIR)/andres/static_plots $(DOCS_DIR)/andres/interactive
	rsync -av --delete $(OUTPUT_DIR)/andres/static_plots/ $(DOCS_DIR)/andres/static_plots/
	rsync -av --delete $(OUTPUT_DIR)/andres/interactive/ $(DOCS_DIR)/andres/interactive/
	@echo "✓ andres synced to docs/"

sync-juan-baffleless:
	@echo "Syncing juan-baffleless to docs/..."
	@mkdir -p $(DOCS_DIR)/juan-baffleless/static_plots $(DOCS_DIR)/juan-baffleless/interactive
	rsync -av --delete $(OUTPUT_DIR)/juan-baffleless/static_plots/ $(DOCS_DIR)/juan-baffleless/static_plots/
	rsync -av --delete $(OUTPUT_DIR)/juan-baffleless/interactive/ $(DOCS_DIR)/juan-baffleless/interactive/
	@echo "✓ juan-baffleless synced to docs/"

sync-lx521-system:
	@echo "Syncing lx521-system to docs/..."
	@mkdir -p $(DOCS_DIR)/lx521-system/static_plots $(DOCS_DIR)/lx521-system/interactive
	rsync -av --delete $(OUTPUT_DIR)/lx521-system/static_plots/ $(DOCS_DIR)/lx521-system/static_plots/
	rsync -av --delete $(OUTPUT_DIR)/lx521-system/interactive/ $(DOCS_DIR)/lx521-system/interactive/
	@echo "✓ lx521-system synced to docs/"

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
	rm -rf $(OUTPUT_DIR)/andres $(OUTPUT_DIR)/juan-baffleless $(OUTPUT_DIR)/lx521-system
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
	@echo "  sync             Sync all output/ to docs/"
	@echo "  deploy           sync + commit + push to GitHub Pages"
	@echo ""
	@echo "Per-Set Targets:"
	@echo "  data-andres      Load andres measurements"
	@echo "  data-juan-baffleless  Load juan-baffleless measurements"
	@echo "  data-lx521-system     Load lx521-system measurements"
	@echo "  viz-andres       Generate andres visualizations"
	@echo "  viz-juan-baffleless   Generate juan-baffleless visualizations"
	@echo "  viz-lx521-system      Generate lx521-system visualizations"
	@echo "  sync-andres      Sync andres to docs/"
	@echo "  sync-juan-baffleless  Sync juan-baffleless to docs/"
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
