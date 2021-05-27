from TestWrapper.root.Debugger import debugger
import os

dump_file = os.path.curdir + "/TestWrapper/results.txt"
#@@ -2574,8 +2574,8 @@ def _match_one(filter_part, dct):
test_id = "tornado.test.httpclient_test.HTTPClientCommonTestCase.test_redirect_put_without_body"


if __name__ == '__main__':
    binary_dir = "./_BugsInPy/framework/bin"
    project = "tornado"
    bug_id = 2

    work_dir = os.path.abspath(binary_dir + "/temp/" + project)
    debugger_module = os.path.abspath("./run_test.py")

    os.system(f"{binary_dir}/bugsinpy-checkout -p {project} -i {bug_id} -v 0")
    os.system(f"{binary_dir}/bugsinpy-compile -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-instrument -c {debugger_module} -w {work_dir}")
    os.system(f"{binary_dir}/bugsinpy-test -a -w {work_dir}")
else:
    debugger.dump_file = dump_file
