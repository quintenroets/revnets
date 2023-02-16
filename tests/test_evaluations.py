from revnets import evaluations, networks, reconstructions


def test_cheat_evaluations():
    test_networks = networks.get_all_networks()
    for network_module in test_networks:
        network = network_module.Network()
        original = network.get_trained_network()
        reconstruction = network.get_architecture()
        reconstructions.cheat.reconstruct(original, reconstruction, network)
        evaluation_metrics = evaluations.evaluate(original, reconstruction, network)

        # cheat should give perfect metrics
        for value in evaluation_metrics.dict().values():
            assert value == 0
