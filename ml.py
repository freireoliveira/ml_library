from ml_lib import k_nearest_neighbors as knn, stats, metrics
import sys

def args() -> str|str|int|str:
    arg_error_message  = 'Incorrect arguments...\n'\
                    'Input as follows:\n'\
                    '--file str --method str {knn}'\
                    ' --fold int --metric str {acc}'

    if len(sys.argv) < 9:
        print(arg_error_message)
        sys.exit()

    methods_lib = {'knn'}
    metrics_lib = {'acc'}


    for idx, arg in enumerate(sys.argv):
        if arg in ('--file' or '-f'):  #error message not needed
            get_file = str(sys.argv[idx + 1])
        elif arg in ('--method'):
            get_method = str(sys.argv[idx + 1])
            if get_method not in methods_lib:
                print(arg_error_message)
                print(f'ERROR: Method {get_method} not available')
                sys.exit()
        elif arg in ('--fold'):
            try:
                get_fold = int(sys.argv[idx + 1])
            except ValueError:
                print(arg_error_message)
                print('ERROR: Fold should be an integer')
                sys.exit()
        elif arg in ('--metric'):
            get_metric = str(sys.argv[idx + 1])
            if get_metric not in metrics_lib:
                print(arg_error_message)
                print(f'ERROR: Metric {get_metric} not available')
                sys.exit()

    return get_file, get_method, get_fold, get_metric


def import_data(file_name:str) -> list|int|int:

    data = []
    count = 0

    with open(f'{file_name}', 'r') as f:
        for l in f:
                line= l.strip().split(' ')
                if count == 0:   #first line with matrix dimensions
                    rows, cols = int(line[0]), int(line[1])
                else:
                    data += [[float(x) for x in line]]
                count += 1

    return data, rows, cols


def main() -> None:
    file_name, method, fold, metric = args()

    data, rows, _ = import_data(file_name)

    if fold > rows:
        raise TypeError(f'Fold up to {rows}') #allows LOO
    
    incorrect = True
    while incorrect:
        z = input('Z-score? (y|n): ')
        if z not in ('y','n'):
            pass
        else:
            incorrect = False

    if z == 'y':
        data = stats(data).z_score()
    else:
        pass

    if method == 'knn':
        try:
            nn = int(input('k-nn: '))
        except ValueError:
            print("ERROR: k should be an integer")
            sys.exit()

        predicted, original = knn(data, nn, fold).run()
        
        if metric == 'acc':
            print('Accuracy:\n'\
            f'{metrics(predicted, original, fold).accuracy()}')

if __name__ == '__main__':
    main()