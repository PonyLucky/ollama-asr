# ollama-asr

# Absolute path to this project (directory containing the Makefile).
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

DESKTOP_DIR := $(HOME)/.local/share/applications
DESKTOP_FILE := $(DESKTOP_DIR)/ollama-asr.desktop

.PHONY: run install uninstall

run:
	$(ROOT)/run.sh

install:
	mkdir -p $(DESKTOP_DIR)
	cp $(ROOT)/assets/ollama-asr.desktop $(DESKTOP_FILE)
	-update-desktop-database $(DESKTOP_DIR) 2>/dev/null
	@echo "Installed $(DESKTOP_FILE)"

uninstall:
	rm -f $(DESKTOP_FILE)
	-update-desktop-database $(DESKTOP_DIR) 2>/dev/null
	@echo "Removed $(DESKTOP_FILE)"
