from TestWrapper.root.Debugger import debugger
import os
import sys

dump_file = os.path.curdir + "/TestWrapper/results.pickle.gz"
test_id = ""


def run_test(root_dir, project, bug_id):
    binary_dir = root_dir + "/_BugsInPy/framework/bin"
    work_dir = os.path.abspath(binary_dir + "/temp/" + project)
    debugger_module = os.path.abspath(root_dir + "/run_test.py")

    os.system(f"{binary_dir}/bugsinpy-checkout -p {project} -i {bug_id} -v 0")
    os.system(f"{binary_dir}/bugsinpy-compile -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-instrument -c {debugger_module} -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-test -a -w {work_dir}")


if __name__ == '__main__':
    from evaluate import SFL_Evaluation
    import argparse

    arg_parser = argparse.ArgumentParser(description='Run the debugger on a specific bug')
    arg_parser.add_argument("-p", "--project_name", required=True, type=str, default="thefuck",
                            help="The name of the target project")
    arg_parser.add_argument("-i", "--bug_id", required=True, type=int, default=2,
                            help="The numerical ID of the target bug")

    args = arg_parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    project = args.project_name
    bug_id = args.bug_id
    run_test(root_dir, project, bug_id)

    print(SFL_Evaluation(dump_file))

else:
    debugger.dump_file = dump_file
