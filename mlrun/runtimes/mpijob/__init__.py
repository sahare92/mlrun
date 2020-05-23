# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .v1 import MpiRuntimeV1
from .v1alpha1 import MpiRuntimeV1Alpha1
from mlrun.k8s_utils import K8sHelper
from mlrun.runtimes.utils import RunError

from kubernetes import client


def infer_mpi_runtime():
    mpijob_crd_version = _get_mpijob_crd_version()

    version_to_runtime = {
        'v1alpha1': MpiRuntimeV1Alpha1,
        'v1': MpiRuntimeV1
    }

    return version_to_runtime[mpijob_crd_version]


def _get_mpijob_crd_version():
    k8s = K8sHelper()
    try:
        mpijob_crd = k8s.get_custom_resource_definition('mpijobs.kubeflow.org')
        return mpijob_crd.spec.version
    except client.rest.ApiException as e:
        raise RunError("Exception occurred while getting MpiJob crd: %s" % e)
