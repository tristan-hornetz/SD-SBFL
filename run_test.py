import os
import sys
if __name__ == '__main__' and (not os.path.islink(os.path.abspath(os.path.dirname(sys.argv[0])) + '/TestWrapper')):
    print('Symlinks not found. Did you run make?')
    exit(-1)
from TestWrapper.root.Debugger import debugger, SFL_Results
from TestWrapper.root.evaluate import BugInfo
dump_file = os.path.curdir + '/TestWrapper/results.pickle.gz'
test_ids = []


def get_test_ids():
    """
    Get a list of Test IDs of tests failing because of the bug currently investigated
    :return: A list of Test IDs of tests failing because of the bug currently investigated
    """
    _results = SFL_Results(debugger)
    _info = BugInfo(_results)
    ret = []
    with open(_info.info_dir + '/run_test.sh', 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            run_test_sh = line.strip(' \n')
            if run_test_sh.startswith('python') or run_test_sh.startswith('pytest ') or run_test_sh.startswith('tox'):
                ret.append(list(filter(lambda s: not s.startswith('-'), run_test_sh.split(' '))).pop())
    return ret


def run_test(root_dir: str, project: str, bug_id: int, output_file=dump_file):
    """
    Start a test run for a specific bug
    :param root_dir: The StatisticalDebugger Repository's absoltue path
    :param project: The project name
    :param bug_id: The numerical Bug ID
    :return: None
    """
    binary_dir = root_dir + '/_BugsInPy/framework/bin'
    work_dir = os.path.abspath(binary_dir + '/temp/' + project)
    debugger_module = os.path.abspath(root_dir + '/run_test.py')
    os.system(f'{binary_dir}/bugsinpy-checkout -p {project} -i {bug_id} -v 0')
    os.system(f'{binary_dir}/bugsinpy-compile -w {work_dir}')
    os.system(f'{binary_dir}/bugsinpy-instrument -c {debugger_module} -w {work_dir}')
    with open(work_dir + "/output_file.info", "wt") as f:
        f.write(output_file)
    os.system(f'{binary_dir}/bugsinpy-test -a -w {work_dir}')


if __name__ == '__main__':
    from evaluate import SFL_Evaluation
    import argparse
    arg_parser = argparse.ArgumentParser(description='Run the debugger on a specific bug')
    arg_parser.add_argument('-p', '--project_name', required=True, type=str, default='thefuck', help='The name of the target project')
    arg_parser.add_argument('-i', '--bug_id', required=True, type=int, default=2, help='The numerical ID of the target bug')
    arg_parser.add_argument('-o', '--output_file', required=False, type=str, default=dump_file, help='The file to dump the results to')

    args = arg_parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    project = args.project_name
    bug_id = args.bug_id

    run_test(root_dir, project, bug_id, os.path.abspath(args.output_file))
    print(SFL_Evaluation(os.path.abspath(args.output_file)))
else:
    if os.path.exists("./output_file.info"):
        with open("./output_file.info", "rt") as f:
            dump_file = f.read()
    debugger.dump_file = dump_file
    test_ids.extend(get_test_ids())
