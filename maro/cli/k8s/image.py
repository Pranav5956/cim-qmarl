# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.operation_lock_wrapper import operation_lock


@check_details_validity
@operation_lock
def push_image(cluster_name: str, image_name: str, **kwargs):
    # Late import.
    from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name,
        )
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def list_image(cluster_name: str, **kwargs):
    # Late import.
    from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.list_image()
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")
