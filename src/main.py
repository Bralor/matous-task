import argparse

import clusterer.containers
from clusterer.containers import Container, create_clusters


def main(source: str) -> None:
    """
    The main entry point of the wrapper.
    """
    container = Container()
    container.config.from_yaml(source)
    data = container.read_source()

    if data is None:
        raise ValueError(
            'Loaded data is empty or None. Please check the input data source.'
        )

    container.wire(modules=[clusterer.containers])
    labels = create_clusters(data)

    print(labels)

    # TODO: Store the outputs


def parse_arguments() -> str:
    """
    Parse command-line arguments and return the config file path.

    Returns:
         str: Parsed configuration filename.
    """
    parser = argparse.ArgumentParser(description='Provide the configuration'
                                                 ' file.')

    parser.add_argument("config_file",
                        type=str,
                        help="Filepath to your configuration.")

    parsed_args = parser.parse_args()

    return parsed_args.config_file


if __name__ == "__main__":
    config_filepath = parse_arguments()
    main(source=config_filepath)
