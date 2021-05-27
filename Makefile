REPO_DIR := $(shell pwd)/_BugsInPy
REPO_URI="https://github.com/tristan-hornetz/BugsInPy"
DEBUGGER_SYMDIR=_root

PY_FILES := $(wildcard *.py)

all: BugsInPy_Repository symlinks

BugsInPy_Repository:
	git clone "$(REPO_URI)" "$(REPO_DIR)"

symdir:
	mkdir "$(DEBUGGER_SYMDIR)"

%.py: symdir
	ln -s "$(shell pwd)/$@" "$(DEBUGGER_SYMDIR)/$@"


symlinks: $(PY_FILES)
	ln -s "$(REPO_DIR)/framework/py/TestWrapper" "$(shell pwd)/TestWrapper"
	ln -s "$(shell pwd)/debuggingbook" "$(shell pwd)/$(DEBUGGER_SYMDIR)/debuggingbook"
	ln -s "$(shell pwd)/$(DEBUGGER_SYMDIR)" "$(REPO_DIR)/framework/py/TestWrapper/root"

clean:
	rm -rf "$(DEBUGGER_SYMDIR)"
	rm -f "$(shell pwd)/TestWrapper"
	rm -rf $(REPO_DIR)
