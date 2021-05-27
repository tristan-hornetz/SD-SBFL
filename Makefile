REPO_DIR := $(shell pwd)/_BugsInPy
REPO_URI="https://github.com/tristan-hornetz/BugsInPy"

PY_FILES := $(wildcard *.py)

all: BugsInPy_Repository symlinks

BugsInPy_Repository:
	git clone "$(REPO_URI)" "$(REPO_DIR)"

_root:
	mkdir _root

%.py: _root
	ln -s "$(shell pwd)/$@" "_root/$@"


symlinks: $(PY_FILES)
	ln -s "$(REPO_DIR)/framework/py/TestWrapper" "$(shell pwd)/TestWrapper"
	ln -s "$(shell pwd)/debuggingbook" "$(shell pwd)/_root/debuggingbook"
	ln -s "$(shell pwd)/_root" "$(REPO_DIR)/framework/py/TestWrapper/root"

clean:
	rm -rf "_root"
	rm -f "$(shell pwd)/TestWrapper"
	rm -rf $(REPO_DIR)
