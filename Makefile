REPO_DIR := $(shell pwd)/_BugsInPy
REPO_URI="https://github.com/tristan-hornetz/BugsInPy"
DEBUGGER_SYMDIR=_root
ENV_SCRIPT=.env.sh
PY_FILES := $(wildcard ./Recorder/*.py)

sh_start=10
sh_end=59
define create_environment =
#!/bin/bash
###Initialize pyenv

python_version=3.9.2
env_name=statistical_debugger

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


###Install required python version if not already installed
python_installed="FALSE"
while read -r line
do
  if [[ $line == "${python_version}" ]]; then
    python_installed="TRUE"
    echo "Python Version ${python_version} is already installed."
    break
  fi
done < <(pyenv versions)

if [[ $python_installed == "FALSE" ]]; then
  pyenv install $python_version
fi

###Check if environment already present
env_present="0"
while read -r line
do
  if [[ $line == "${env_name}"* ]]; then
    env_present="1"
    break
  fi
done < <(pyenv virtualenvs)


if [[ $env_present == "0" ]]; then
  ###Create environment
  echo "Creating environment ${env_name}..."
  pyenv virtualenv "$python_version" "$env_name"
fi

###Activate environment
pyenv activate "$env_name"

### Install requirements
pip install -r requirements.txt
echo "pyenv activate $env_name" > activate
pyenv deactivate

endef


all: BugsInPy_Repository symlinks environment

BugsInPy_Repository:
	git clone "$(REPO_URI)" "$(REPO_DIR)"

symdir:
	mkdir "$(DEBUGGER_SYMDIR)"

%.py: symdir
	ln -s "$(shell pwd)/$@" "$(DEBUGGER_SYMDIR)/$(shell basename $@)"

environment:


symlinks: $(PY_FILES)
	ln -s "$(REPO_DIR)/framework/py/TestWrapper" "$(shell pwd)/TestWrapper"
	ln -s "$(shell pwd)/debuggingbook" "$(shell pwd)/$(DEBUGGER_SYMDIR)/debuggingbook"
	ln -s "$(shell pwd)/$(DEBUGGER_SYMDIR)" "$(REPO_DIR)/framework/py/TestWrapper/root"
	ln -s "$(shell pwd)/run_single_test.py" "$(shell pwd)/$(DEBUGGER_SYMDIR)/run.py"


environment:
	@sed '$(sh_start),$(sh_end)!d' Makefile > $(ENV_SCRIPT)
	@chmod +x $(ENV_SCRIPT)
	@./$(ENV_SCRIPT)
	@rm $(ENV_SCRIPT)

clean:
	rm -rf "$(DEBUGGER_SYMDIR)"
	rm -f "$(shell pwd)/TestWrapper"
	rm -rf $(REPO_DIR)
	rm -f $(ENV_SCRIPT)
	rm -f activate
	rm -rf .temp
	rm -rf ./Evaluator/.temp
	pyenv virtualenv-delete -f statistical_debugger
