from ParticleSwarmOptimization import ParticleSwarmOptimization
from ClusteredParticleFilteringOptimization import ClusteredParticleFilteringOptimization
import json



def runPSO(configFile):
    fp = open(configFile)
    config = json.load(fp)
    fp.close()
    controllerConfig = config['PSO_Plant']['Controller']
    plantConfig = config['PSO_Plant']
    databaseConfig = config['Database']
    PSOConfig = config['PSO']

    Optimizer = ParticleSwarmOptimization(PSOConfig, controllerConfig, plantConfig, databaseConfig, True)

    print('Starting PSO Controller Optimization')
    Optimizer.run()


def runCPF(configFile):
    fp = open(configFile)
    config = json.load(fp)
    fp.close()
    controllerConfig = config['CPF_Plant']['Controller']
    plantConfig = config['CPF_Plant']
    databaseConfig = config['Database']
    CPFConfig = config['CPF']

    Optimizer = ClusteredParticleFilteringOptimization(CPFConfig, controllerConfig, plantConfig, databaseConfig, render=True)

    print('Starting CPF Controller Optimization')
    Optimizer.run()



if __name__ == "__main__":
    PSO_config = 'PSO_Config.json'
    CPF_config = 'CPF_Config.json'
    # runPSO(PSO_config)
    runCPF(CPF_config)