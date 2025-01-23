import numpy.typing as npt
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from clusterer.clusterer import DataLoader, KMeansProcessor


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    kmeans = providers.Factory(
            KMeansProcessor,
            n_clusters=config.hyperparameters.n_clusters,
            random_state=config.hyperparameters.random_state,
            max_iter=config.hyperparameters.max_iter
    )

    # TODO: added more clustering algorithms

    algorithm = providers.Selector(config.algorithm,
                                   kmeans=kmeans)
    # Source:
    # https://python-dependency-injector.ets-labs.org/
    # tutorials/cli.html#selector

    read_json = providers.Factory(lambda source: DataLoader
                                  .from_json(source)
                                  .data,
                                  source=config.input.filepath)

    read_numpy = providers.Factory(lambda source: DataLoader
                                   .from_numpy(source)
                                   .data,
                                   source=config.input.filepath)

    read_source = providers.Selector(config.input.format,
                                     json=read_json,
                                     npy=read_numpy)

    # TODO: Save the results


@inject
def create_clusters(source: npt.NDArray,
                    algorithm=Provide[Container.algorithm]
                    ) -> npt.NDArray[int]:
    """
    Run the clustering algorithm processor on the provided data.
    """
    model = algorithm
    return model.fit(source)
