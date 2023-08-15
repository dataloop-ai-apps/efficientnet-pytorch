import dtlpy as dl
from model_adapter import Adapter
import os

dl.login()

project = dl.projects.get(project_name='Demo Saarah')
codebase: dl.GitCodebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/efficientnet-pytorch',
                                          git_tag='1.0')

metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                      default_configuration={'model_name': 'b0'},
                                      output_type=dl.AnnotationType.CLASSIFICATION
                                      )
module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

package = project.packages.push(package_name='efficientnet_pytorch',
                                src_path=os.getcwd(),
                                package_type='ml',
                                modules=[module],
                                service_config={
                                    'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                    runnerImage='gcr.io/viewo-g/piper/agent/runner'
                                                                                '/gpu/main:1.81.4.latest',
                                                                    autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                        min_replicas=0,
                                                                        max_replicas=1),
                                                                    concurrency=1).to_json()},
                                metadata=metadata)

model = package.models.create(model_name='efficientnet_pytorch',
                          description='efficientnet',
                          tags=['pretrained', 'efficientnet', 'pytorch'],
                        configuration={"model_name": "b0", "no_classes": 10},
                          dataset_id=None,
                          project_id=package.project.id
                          )

model.status = 'trained'
model.update()
model.deploy()