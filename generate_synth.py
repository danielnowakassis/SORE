


def agrawal(functions, abrupt = True, instances = 1000000):
    drift_points = len(functions)
    drift_length = int(instances / drift_points)
    dataset_init = f"-i {instances} -f {(int(instances/100))} -s "
    dataset_corpus = ""
    for i in range(drift_points - 1, -1, -1):
        if i == drift_points - 1:
            dataset_corpus = f"generators.AgrawalGenerator -f {functions[i]}"
        else:
            if abrupt:
                dataset_corpus = f"ConceptDriftStream -s (generators.AgrawalGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w 50 -p {drift_length}"
            else:
                dataset_corpus = f"ConceptDriftStream -s (generators.AgrawalGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w {int(min(drift_length/5, 50000))} -p {drift_length}"
    return dataset_init + "(" + dataset_corpus +  ")"

def rbf(n_classes = 2, n_atributos = 10, rate = 0.001, instances = 50000):
    dataset_init = f"-i {instances} -f {(int(instances/100))} -s "
    #dataset_init + "(" + ")"
    return f"generators.RandomRBFGeneratorDrift -s {rate} -c {n_classes} -a {n_atributos}" 
#-i {instances} -f {(int(instances/100))} 
def assets_negotiator(functions, abrupt = True, instances = 1000000):
    drift_points = len(functions)
    drift_length = int(instances / drift_points)
    dataset_init = f"-i {instances} -f {(int(instances/100))} -s "
    dataset_corpus = ""
    for i in range(drift_points - 1, -1, -1):
        if i == drift_points - 1:
            dataset_corpus = f"generators.AssetNegotiationGenerator -f {functions[i]}"
        else:
            if abrupt:
                dataset_corpus = f"ConceptDriftStream -s (generators.AssetNegotiationGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w 50 -p {drift_length}"
            else:
                dataset_corpus = f"ConceptDriftStream -s (generators.AssetNegotiationGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w {int(min(drift_length/5, 50000))} -p {drift_length}"
    return dataset_init + "(" + dataset_corpus +  ")"

def sea(functions, abrupt = True, instances = 1000000): 
    drift_points = len(functions)
    drift_length = int(instances / drift_points)
    dataset_init = f"-i {instances} -f {(int(instances/100))} -s "
    dataset_corpus = ""
    for i in range(drift_points - 1, -1, -1):
        if i == drift_points - 1:
            dataset_corpus = f"generators.SEAGenerator -f {functions[i]}"
        else:
            if abrupt:
                dataset_corpus = f"ConceptDriftStream -s (generators.SEAGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w 50 -p {drift_length}"
            else:
                dataset_corpus = f"ConceptDriftStream -s (generators.SEAGenerator -f {functions[i]}) -d (" + dataset_corpus + f") -w {int(min(drift_length/5, 50000))} -p {drift_length}"
    return dataset_init + "(" + dataset_corpus +  ")"

def led(functions, abrupt = True, instances = 1000000): 
    drift_points = len(functions)
    drift_length = int(instances / drift_points)
    dataset_init = f"-i {instances} -f {(int(instances/100))} -s "
    dataset_corpus = ""
    for i in range(drift_points - 1, -1, -1):
        if i == drift_points - 1:
            dataset_corpus = f"generators.LEDGeneratorDrift -d {functions[i]}"
        else:
            if abrupt:
                dataset_corpus = f"ConceptDriftStream -s (generators.LEDGeneratorDrift -d {functions[i]}) -d (" + dataset_corpus + f") -w 50 -p {drift_length}"
            else:
                dataset_corpus = f"ConceptDriftStream -s (generators.LEDGeneratorDrift -d {functions[i]}) -d (" + dataset_corpus + f") -w {int(min(drift_length/5, 50000))} -p {drift_length}"
    return dataset_init + "(" + dataset_corpus +  ")"

