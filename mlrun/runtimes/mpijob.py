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
import time
from copy import deepcopy
import shlex

from .utils import AsyncLogWriter, RunError
from ..config import config
from ..model import RunObject
from .kubejob import KubejobRuntime
from ..utils import update_in, logger, get_in
from ..execution import MLClientCtx

from kubernetes import client

mpi_group = 'kubeflow.org'
mpi_version = 'v1'
mpi_plural = 'mpijobs'

_mpijob_pod_template = {
    'spec': {
        'containers': [{
            'image': 'mlrun/mpijob',
            'name': 'base',
            'command': [],
            'env': [],
            'volumeMounts': [],
            'securityContext': {
                'capabilities': {'add': ['IPC_LOCK']}},
            'resources': {
                'limits': {}}}],
        'volumes': []
    },
    'metadata': {}
}


def _generate_mpi_job(launcher_pod_template, worker_pod_template):
    return {
     'apiVersion': 'kubeflow.org/{0}'.format(mpi_version),
     'kind': 'MPIJob',
     'metadata': {
         'name': '',
         'namespace': 'default-tenant'
     },
     'spec': {
         'mpiReplicaSpecs': {
             'Launcher': {
                 'template': launcher_pod_template
             }, 'Worker': {
                 'replicas': 1,
                 'template': worker_pod_template
             }
         }
     }}


def _update_container(struct, key, value):
    struct['spec']['containers'][0][key] = value


class MpiRuntime(KubejobRuntime):
    kind = 'mpijob'
    _is_nested = False

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        if runobj.metadata.iteration:
            self.store_run(runobj)

        meta = self._get_meta(runobj, True)
        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name

        # Populate an mpijob object

        # start by populating pod templates
        launcher_pod_template = deepcopy(_mpijob_pod_template)
        worker_pod_template = deepcopy(_mpijob_pod_template)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if self.spec.image:
                _update_container(pod_template, 'image', self.full_image_path())
            _update_container(pod_template, 'volumeMounts', self.spec.volume_mounts)
            extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
            if self.spec.rundb:
                extra_env['MLRUN_DBPATH'] = self.spec.rundb
            extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]
            _update_container(pod_template, 'env', extra_env + self.spec.env)
            if self.spec.image_pull_policy:
                _update_container(
                    pod_template, 'imagePullPolicy', self.spec.image_pull_policy)
            if self.spec.workdir:
                _update_container(pod_template, 'workingDir', self.spec.workdir)
            if self.spec.image_pull_secret:
                update_in(pod_template, 'spec.imagePullSecrets',
                          [{'name': self.spec.image_pull_secret}])
            update_in(pod_template, 'metadata.labels', pod_labels)
            update_in(pod_template, 'spec.volumes', self.spec.volumes)

        # configuration for workers
        # update resources only for workers because the launcher doesn't require
        # special resources (like GPUs, Memory, etc..)
        if self.spec.resources:
            _update_container(worker_pod_template, 'resources', self.spec.resources)

        # configuration for launcher
        quoted_args = []
        for arg in self.spec.args:
            quoted_args.append(shlex.quote(arg))
        if self.spec.command:
            _update_container(
                launcher_pod_template, 'command',
                ['mpirun', 'python', shlex.quote(self.spec.command)] + quoted_args)

        # generate mpi job using the above job_pod_template
        job = _generate_mpi_job(launcher_pod_template, worker_pod_template)

        # update the replicas only for workers
        update_in(job, 'spec.mpiReplicaSpecs.Worker.replicas', self.spec.replicas or 1)

        update_in(job, 'metadata', meta.to_dict())

        resp = self._submit_mpijob(job, meta.namespace)
        launcher_state = None

        timeout = int(config.submit_timeout) or 120
        for _ in range(timeout):
            resp = self.get_job(meta.name, meta.namespace)
            launcher_state = get_in(resp, 'status.replicaStatuses.Launcher')
            if resp and launcher_state:
                break
            time.sleep(1)

        if resp:
            logger.info('MpiJob {} state={}'.format(
                meta.name, launcher_state or 'unknown'))
            if launcher_state:
                state = 'active' if launcher_state['active'] == 1 else 'error'
                launcher, status = self._get_launcher(meta.name,
                                                      meta.namespace)
                execution.set_hostname(launcher)
                execution.set_state('running' if state == 'active' else state)
                if self.kfp:
                    writer = AsyncLogWriter(self._db_conn, runobj)
                    status = self._get_k8s().watch(
                        launcher, meta.namespace, writer=writer)
                    logger.info(
                        'MpiJob {} finished with state {}'.format(
                            meta.name, status))
                    if status == 'succeeded':
                        execution.set_state('completed')
                    else:
                        execution.set_state('error', 'MpiJob {} finished with state {}'.format(meta.name, status))
                else:
                    txt = 'MpiJob {} launcher pod {} state {}'.format(
                        meta.name, launcher, state)
                    logger.info(txt)
                    runobj.status.status_text = txt
            else:
                txt = 'MpiJob status unknown or failed, check pods: {}'.format(
                    self.get_pods(meta.name, meta.namespace))
                logger.warning(txt)
                runobj.status.status_text = txt
                if self.kfp:
                    execution.set_state('error', txt)

        return None

    def _submit_mpijob(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                mpi_group, mpi_version, namespace=namespace,
                plural=mpi_plural, body=job)
            name = get_in(resp, 'metadata.name', 'unknown')
            logger.info('MpiJob {} created'.format(name))
            return resp
        except client.rest.ApiException as e:
            logger.error("Exception when creating MPIJob: %s" % e)
            raise RunError("Exception when creating MPIJob: %s" % e)

    def delete_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            # delete the mpi job\
            body = client.V1DeleteOptions()
            resp = k8s.crdapi.delete_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name, body)
            logger.info('del status: {}'.format(
                get_in(resp, 'status', 'unknown')))
        except client.rest.ApiException as e:
            print("Exception when deleting MPIJob: %s" % e)

    def list_jobs(self, namespace=None, selector='', show=True):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.list_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural,
                watch=False, label_selector=selector)
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)

        items = []
        if resp:
            items = resp.get('items', [])
            if show and items:
                print('{:10} {:20} {:21} {}'.format(
                    'status', 'name', 'start', 'end'))
                for i in items:
                    print('{:10} {:20} {:21} {}'.format(
                        get_in(i, 'status.launcherStatus', ''),
                        get_in(i, 'metadata.name', ''),
                        get_in(i, 'status.startTime', ''),
                        get_in(i, 'status.completionTime', ''),
                    ))
        return items

    def get_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name)
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)
        return resp

    def get_pods(self, name=None, namespace=None, launcher=False):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        selector = ''
        if name:
            selector += ',mpi_job_name={}'.format(name)
        if launcher:
            selector += ',mpi-job-role=launcher'
        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def _get_launcher(self, name, namespace=None):
        pods = self.get_pods(name, namespace, launcher=True)
        if not pods:
            logger.error('no pod matches that job name')
            return
        # TODO: Why was this here?
        # k8s = self._get_k8s()
        return list(pods.items())[0]
