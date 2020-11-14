from xmen.tests.experiment import AnExperiment
import subprocess

x = AnExperiment()
x.register('/tmp/test', 'an_experiment')

x.update_version()
code = x.get_run_script()

with open('/tmp/test_script.sh', 'w') as file:
    file.write(code)

# subprocess.call(['chmod +x /tmp/test_script.sh'])
# subprocess.call('/tmp/test_script.sh')
