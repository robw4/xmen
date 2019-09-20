from xmen.experiment import Experiment, experiment_parser
import os

class AnExperiment(Experiment):
    """The doc string of AnExperiment class"""
    
    def __init__(self):
        """The doc string of __init__ experiment class."""
        super(AnExperiment, self).__init__()
        self.a: str = 'h'     # A parameter
        self.b: int = 17      # Another parameter
        
        # Private attributes will not be treated as parameters
        self._c = None
    
    def run(self):
        print(f'The experiment state inside run is {self.status} for experiment {self.name}, a = {self.a}, b = {self.b}')
        self.leave_message({'Hi': 'I am running!'})
        
        with open(os.path.join(self.directory, 'logs.txt'), 'w') as f:
            f.write('This was written from a running experiment')

if __name__ == '__main__':
    args = experiment_parser.parse_args()
    exp = AnExperiment()
    exp.main(args)