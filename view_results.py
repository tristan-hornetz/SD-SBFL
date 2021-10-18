
if __name__ == '__main__':
    import os
    import argparse
    from evaluate import EvaluationRun

    arg_parser = argparse.ArgumentParser(description='View evaluation results.')
    arg_parser.add_argument("-r", "--result_file", required=True, type=str,
                            help="The file containing test results")
    args = arg_parser.parse_args()
    result_file: str = args.result_file
    if not os.path.exists(result_file):
        print("The file you specified does not exist.")
        exit(1)
    evr = EvaluationRun.load(result_file)
    os.system(f"echo '{str(evr)}' | less")
    exit(0)
