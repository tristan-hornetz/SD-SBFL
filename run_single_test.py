import os
import subprocess
import sys
from multiprocessing.connection import Connection

if __name__ == '__main__' and (not os.path.islink(os.path.abspath(os.path.dirname(sys.argv[0])) + '/TestWrapper')):
    print('Symlinks not found. Did you run make?')
    exit(-1)
dump_file = os.path.curdir + '/TestWrapper/results.pickle.gz'
test_ids = []


def get_info_directory(_results):
    """
    Get the absolute path of the BugsInPy directory containing information on the bug tested in _results
    :param _results: The SFL_Results of a test run
    :return: The absolute path of the BugsInPy directory containing information on the bug tested in _results
    """
    base = _results.work_dir.split("_BugsInPy")[0]
    return os.path.abspath(f"{base}_BugsInPy/projects/{_results.project_name}/bugs/{_results.bug_id}")


def get_test_ids():
    """
    Get a list of Test IDs of tests failing because of the bug currently investigated
    :return: A list of Test IDs of tests failing because of the bug currently investigated
    """
    _results = SFL_Results(debugger)
    info_dir = get_info_directory(_results)
    ret = []
    with open(info_dir + '/run_test.sh', 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            run_test_sh = line.strip(' \n')
            if run_test_sh.startswith('python') or run_test_sh.startswith('pytest ') or run_test_sh.startswith(
                    'tox') or run_test_sh.startswith('py.test'):
                ret.append(list(filter(lambda s: not s.startswith('-'), run_test_sh.split(' '))).pop().split("[")[0])
    if str(os.path.realpath(info_dir + "/../..")).rstrip("/").endswith("/tqdm"):
        rc = ret.copy()
        ret = []
        for id in rc:
            ret.append(id.replace("tqdm/tests/tests_", "tqdm/tests/test_tests_"))
    return ret


def test_execute(command: str, connection: Connection):
    class PFile:
        def write(self, s):
            connection.send(s)

    sys.stdout = PFile()
    sys.stderr = PFile()
    os.system(command)


def get_subprocesses(pid: int):
    sp = subprocess.Popen(['ps', '-opid', '--no-headers', '--ppid', str(pid)], encoding='utf8', stdout=subprocess.PIPE)
    ret = list()
    try:
        ret = [int(line) for line in sp.stdout.read().splitlines()]
    except Exception as e:
        print(e)
    if len(ret) > 0:
        for i in ret:
            ret.extend(set(get_subprocesses(i)))
    print(ret)
    return ret


def run_test(root_dir: str, project: str, bug_id: int, output_file=dump_file, work_dir=""):
    """
    Start a test run for a specific bug
    :param root_dir: The StatisticalDebugger Repository's absoltue path
    :param project: The project name
    :param bug_id: The numerical Bug ID
    :return: None
    """
    binary_dir = root_dir + '/_BugsInPy/framework/bin'
    if work_dir == "":
        work_dir = os.path.abspath(binary_dir + '/temp/' + project)
    else:
        work_dir += "/" + project
    debugger_module = os.path.abspath(root_dir + '/_root/run.py')
    os.system(f'{binary_dir}/bugsinpy-checkout -p {project} -i {bug_id} -v 0 -w {os.path.dirname(work_dir)}')
    os.system(f'{binary_dir}/bugsinpy-compile -w {work_dir}')
    if os.path.exists(f'{work_dir}/bugsinpy_setup.sh'):
        print("\n=========setup============\n")
        os.system(f"cp {binary_dir}/alltest_template.sh {work_dir}/bugsinpy_modified_setup.sh")
        os.system(f"cat {work_dir}/bugsinpy_setup.sh >> {work_dir}/bugsinpy_modified_setup.sh")
        os.system(f'chmod +x {work_dir}/bugsinpy_modified_setup.sh')
        os.system(f'(cd \'{work_dir}\' && ./bugsinpy_modified_setup.sh)')
    os.system(f'{binary_dir}/bugsinpy-instrument -c {debugger_module} -w {work_dir}')
    with open(work_dir + "/output_file.info", "wt") as f:
        f.write(output_file)

    info_dir = root_dir + '/_BugsInPy/projects/' + project

    if os.path.isfile(info_dir + "/alltest.sh"):
        os.system(f"cp {binary_dir}/alltest_template.sh {work_dir}/alltest.sh")
        os.system(f"cat {info_dir + '/alltest.sh'} >> {work_dir}/alltest.sh")
        os.system(f"chmod +x {work_dir}/alltest.sh")
        os.system(f"echo '\n\npyenv deactivate' >> {work_dir}/alltest.sh")
        command = f"(cd \'{work_dir}\' && {work_dir}/alltest.sh)"
    else:
        command = f'{binary_dir}/bugsinpy-test -a -w {work_dir}'
    os.system(command)


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Run the debugger on a specific bug')
    arg_parser.add_argument('-p', '--project_name', required=True, type=str, default='thefuck',
                            help='The name of the target project')
    arg_parser.add_argument('-i', '--bug_id', required=True, type=int, default=2,
                            help='The numerical ID of the target bug')
    arg_parser.add_argument('-o', '--output_file', required=False, type=str, default=dump_file,
                            help='The file to dump the results to')

    args = arg_parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    project = args.project_name
    bug_id = args.bug_id

    run_test(root_dir, project, bug_id, os.path.abspath(args.output_file))
else:
    try:
        if os.path.exists("./output_file.info"):
            with open("./output_file.info", "rt") as f:
                dump_file = f.read()
        from TestWrapper.root.Debugger import debugger, SFL_Results

        debugger.dump_file = dump_file
        test_ids.extend(get_test_ids())
    except:
        pass
