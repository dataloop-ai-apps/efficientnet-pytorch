import dtlpy as dl
from efficientnet_pytorch import EfficientNet


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for my model',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def load(self, config, **kwargs):
        print('loading a model')
        self.model = EfficientNet.from_pretrained('efficientnet-' + config['model_name'],
                                                  num_classes=config['no_classes'])
        self.model.eval()

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        preds = self.model(batch)
        batch_annotations = list()
        for i_img, predicted_class in enumerate(preds):  # annotations per image
            image_annotations = dl.AnnotationCollection()
            image_annotations.add(annotation_definition=dl.Classification(label=predicted_class),
                                  model_info={'name': self.model_name})
            batch_annotations.append(image_annotations)
        return batch_annotations
