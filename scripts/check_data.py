import argparse, os, glob, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')

CHECKS = {
'mnist': lambda: os.path.isdir(os.path.join(DATA, 'mnist')),
'cifar10': lambda: os.path.isdir(os.path.join(DATA, 'cifar10')),
'chestxray14': lambda: os.path.isfile(os.path.join(DATA, 'chestxray14', 'Data_Entry_2017.csv')) \
and len(glob.glob(os.path.join(DATA, 'chestxray14', 'images', '**', '*.*'), recursive=True)) > 100000,
}

if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=list(CHECKS.keys()) + ['all'], default='all')
args = p.parse_args()

datasets = CHECKS.keys() if args.dataset == 'all' else [args.dataset]
ok = True
for d in datasets:
present = CHECKS[d]()
status = 'OK' if present else 'MISSING'
print(f'{d}: {status}')
ok = ok and present
sys.exit(0 if ok else 1)
