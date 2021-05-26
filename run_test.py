from TestWrapper.root.Debugger import debugger
import os

dump_file = os.path.curdir + "/TestWrapper/results.txt"
test_id = "tests/test_utils.py::test_get_all_executables_pathsep"

if __name__ == '__main__':
    binary_dir = "./_BugsInPy/framework/bin"
    project = "thefuck"
    bug_id = 2

    work_dir = os.path.abspath(binary_dir + "/temp/" + project)
    debugger_module = os.path.abspath("./run_test.py")

    os.system(f"{binary_dir}/bugsinpy-checkout -p {project} -i {bug_id} -v 0")
    os.system(f"{binary_dir}/bugsinpy-compile -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-instrument -c {debugger_module} -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-test -a -w {work_dir}")
else:
    debugger.dump_file = dump_file
