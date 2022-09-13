from controller import Controller
import neat


class NeatController(Controller):

    def __int__(self):

        config_filepath = 'neat_config.txt'
        configuration = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_filepath
        )
        self.configuration = configuration

    def control(self, sensor_values, genome):
        neuralnet = neat.nn.FeedForwardNetwork.create(genome, self.configuration)
        output = neuralnet.activate(sensor_values)

        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]