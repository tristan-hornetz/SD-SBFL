REPO_DIR := $(shell pwd)/_BugsInPy
REPO_URI="https://github.com/tristan-hornetz/BugsInPy"

all: BugsInPy_Repository symlinks

BugsInPy_Repository:
	git clone "$(REPO_URI)" "$(REPO_DIR)"

symlinks:
	ln -s "$(REPO_DIR)/framework/py/TestWrapper" "$(shell pwd)/TestWrapper"
	ln -s "$(shell pwd)" "$(REPO_DIR)/framework/py/TestWrapper/root"

clean:
	rm -f "$(shell pwd)/bugsinpy_binaries"
	rm -f "$(shell pwd)/TestWrapper"
	rm -rf $(REPO_DIR)
