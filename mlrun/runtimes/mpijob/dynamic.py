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

from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.kubejob import KubejobRuntime


class DynamicallyInferredMpiRuntime(KubejobRuntime):
    def _run(self, runobj: RunObject, execution: MLClientCtx):
        inferred_runtime = self._infer_mpi_runtime()
        return inferred_runtime._run(runobj, execution)

    def _infer_mpi_runtime(self):
        namespace = self._get_k8s().ns()

        # default to v1alpha1 for backwards compatibility
        mpi_job_crd_version = 'v1alpha1'

        # get mpi-operator pod
        res = self._k8s.list_pods(namespace=namespace, selector='release=mpi-operator')
        if len(res) > 0:
            mpi_operator_pod = res[0]
            mpi_job_crd_version = mpi_operator_pod.metadata.labels.get('crd-version', mpi_job_crd_version)

        crd_version_to_runtime = {
            'v1alpha1': MpiRuntimeV1Alpha1,
            'v1': MpiRuntimeV1
        }
        return crd_version_to_runtime[mpi_job_crd_version]
