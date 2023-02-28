from revnets import evaluations, networks, reconstructions


def test_cheat_evaluations():
    test_networks = networks.get_all_networks()
    for network_module in test_networks:
        network = network_module.Network()
        original = network.get_trained_network()
        reconstructor = reconstructions.cheat.Reconstructor(original, network)
        reconstruction = reconstructor.reconstruct()
        evaluation_metrics = evaluations.evaluate(original, reconstruction, network)

        # cheat should give perfect metrics
        perfect_values = (
            evaluation_metrics.weights_mse,
            evaluation_metrics.train_outputs_mse,
            evaluation_metrics.val_outputs_mse,
            evaluation_metrics.train_outputs_mse,
        )
        for value in perfect_values:
            assert value in (0, None)
