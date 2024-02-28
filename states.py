"""
This app watermakrs given CNN as follows:
1. All hyperparameters and the target model are read from the config file.
2. A trigger set is consstructed based on the hyperparameters values.
3. The cordinator trains a global model on a trigger set.
4. The coordinator broadcasts watermarked model to all participants
"""
import bios

from FeatureCloud.app.engine.app import AppState, app_state, Role

from watermarking.make_trigger_set import get_dataset
from watermarking.model_loader import load_model
from watermarking.embed_wm import embed
from watermarking.savers import save_model, save_trigger_set


@app_state('initial', Role.BOTH)  
class InitialState(AppState):
    """
    Initial State
    """

    def register(self):
        self.register_transition('read_input', Role.COORDINATOR)  
        self.register_transition('output', Role.PARTICIPANT)

    def run(self):      
        if self.is_coordinator:
            return 'read_input'
        else:
            return 'output'

@app_state('read_input', Role.COORDINATOR)
class InputState(AppState):
    """
    Input data reading
    """

    def register(self):
        self.register_transition('embed_wm', Role.COORDINATOR)  

    def run(self):
        self.update(message='reading config file', progress=0.2)  
        self.read_config()
        return 'embed_wm'

    def read_config(self):
        config = bios.read('/mnt/input/config.yml')['fc_cnn_wm'] 
        self.store('model', config['model'])
        self.store('data', config['data'])
        self.store('wm_settings', config['wm_settings'])
        

@app_state('embed_wm', Role.COORDINATOR)
class EmbeddingState(AppState):
    """
    Watermark embedding
    """

    def register(self):
        self.register_transition('output', Role.COORDINATOR)

    def run(self):
        self.update(message='processing data', progress=0.3)

        model_info = self.load('model')
        data_info = self.load('data')
        wm_settings = self.load('wm_settings')

        model_path = 'mnt/input/' + model_info['model_name']
        architecture = model_info['architecture']
        turn_off_bn = model_info['turn_off_batch_norm']
        
        dataset_name = data_info.get('dataset_name')
        dataset_folder = data_info.get('dataset_folder')
        data_extensions = data_info.get('data_extensions')
        trigger_set_size = data_info.get('trigger_set_size', 100)
        image_size = data_info.get('image_size', {'height': 224, 'width': 224})
        height = image_size.get('height', 224)
        width = image_size.get('width', 224)
        num_channels = data_info.get('num_channels', 3)
        num_classes = data_info.get('num_claases', 10)
        mean = data_info.get('mean')
        std = data_info.get('std')

        wm_type = wm_settings.get('wm_type', "odd_abstract") 
        wm_classes = wm_settings.get('wm_classes')
        wm_th = wm_settings.get('wm_th', 0.8) 
        batch_size = wm_settings.get('batch_size', 32)
        optimizer = wm_settings.get('optimizer', 'sgd')
        lr = wm_settings.get('lr', 0.001)
        momentum = wm_settings.get('momentum', 0.9)
        max_epochs = wm_settings.get('max_epochs', 100)
        
        dataset_path = dataset_folder if not dataset_folder else '/mnt/input/datasets/' + dataset_folder
        self.update(message='creating trigger set', progress=0.35)
        trigger_set = get_dataset(wm_type, trigger_set_size, num_classes, num_channels, wm_classes,
                                height, width, mean, std, dataset_name=dataset_name, 
                                dataset_path=dataset_path, extensions=data_extensions)

        save_trigger_set(trigger_set, '/mnt/output')
        model = load_model(model_path, architecture, num_channels, num_classes)
        self.update(message='embedding a watermark', progress=0.4)
        wm_model, epochs, acc = embed(model, trigger_set, wm_th, batch_size, optimizer, 
                                      lr, momentum, max_epochs, turn_off_bn)
        save_model(wm_model, 'mnt/output')
        self.store('wm_accuracy', acc)
        self.store('wm_epochs', epochs) 
        return 'output'


@app_state('output', Role.BOTH)
class OutputState(AppState):
    """
    Final state
    """
    def register(self):
        self.register_transition('terminal')

    def run(self):
        epochs = self.load('wm_epochs')
        wm_acc = self.load('wm_accuracy')
        self.update(message=f'finished watermarking', progress=0.9)
        self.log(f'The model was trained for {epochs} epochs and reached {wm_acc} % accuracy on the trigger set')
        return 'terminal' 