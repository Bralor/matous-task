from dependency_injector import containers, providers

from clusterer.clusterer import DataLoader


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    read_json = providers.Factory(DataLoader.from_json,
                                  source=config.input.filepath)

    read_numpy = providers.Factory(DataLoader.from_numpy,
                                   source=config.input.filepath)

    # TODO: Add algorithms factories
