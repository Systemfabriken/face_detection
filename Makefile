# Define variables
SHELL := /bin/bash
SCRIPT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_PATH := $(SCRIPT_PATH)/.venv/bin/activate
PROMOTIONS_PATH := $(SCRIPT_PATH)/src/qt_promotions
PYQT5_RESOURCE_PATH := $(SCRIPT_PATH)/src/ui_generated/pyqt5
SITE_PACKAGES_PATH := $(shell source $(VENV_PATH) && python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PTH_FILE_PATH := $(SITE_PACKAGES_PATH)/qt_promotions.pth
PYQT5_RESOURCE_PTH := $(SITE_PACKAGES_PATH)/pyqt5-resource-file.pth

# Default target executed when no arguments given to make.
default_target: build
.PHONY: default_target

# Target for creating environment
environment:
	# Install Python3 and pip3
	sudo apt update
	sudo apt install -y python3 python3-pip python3-venv

	# Install Qt5 and Qt Designer
	sudo apt install -y qttools5-dev-tools

	# Install other dependencies
	sudo apt-get install -y libxcb-xinerama0

	unset GTK_PATH

	# Create a Python virtual environment
	python3 -m venv .venv
	source $(VENV_PATH)

	# Install the necessary Python libraries inside the virtual environment
	/bin/bash -c "source $(VENV_PATH) && pip3 install pyqt5 pyqt5-tools opencv-python-headless"

.PHONY: environment

# Target for building
build:
	cd $(SCRIPT_PATH)/scripts && ./convert_ui_to_py.sh
	# cd $(SCRIPT_PATH)/rope-robot-drm-protocol && make python
	echo $(PROMOTIONS_PATH) > $(PTH_FILE_PATH)
	# echo $(PYQT5_RESOURCE_PATH) > $(PYQT5_RESOURCE_PTH)
.PHONY: build

# Target for running the main script
run_face_detection: build
	source $(VENV_PATH) && python3 src/face_detection_main.py
.PHONY: run_face_detection

run_face_classifier: build
	source $(VENV_PATH) && python3 src/face_classifier_main.py
.PHONY: run_face_classifier

# Target for running tests
run_tests: build
.PHONY: run_tests

run_hw_tests: build
.PHONY: run_hw_tests
