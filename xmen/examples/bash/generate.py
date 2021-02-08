import argparse

parser = argparse.ArgumentParser(prog='Generate script.sh and defaults.yml examples for ')
parser.add_argument('root', help='The root directory in which to instantiate the bash experiment')

if __name__ == '__main__':
    import os
    from xmen.utils import get_version
    from ruamel.yaml import YAML
    yaml = YAML()
    folder, _ = os.path.split(__file__)

    args = parser.parse_args()
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    version = get_version(path=__file__)
    defaults = {
        '_version': {'path': __file__},
        '_created': '8/2/2021',
        'a': 'Hello',
        'b': 'Planet'}
    with open(os.path.join(args.root, 'defaults.yml'), 'w') as f:
        yaml.dump(defaults, f)

    with open(os.path.join(args.root, 'script.sh'), 'w') as f:
        f.write('\n'.join(
            ['#! /bin/bash',
             'echo "#######################################"',
             'echo "contents of params.yml file ${1}"',
             'echo "#######################################"',
             'cat ${1}',
             'echo " "']))
